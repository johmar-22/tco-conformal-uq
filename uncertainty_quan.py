# -*- coding: utf-8 -*-
"""
Conformal Uncertainty Quantification for Transparent Conductive Oxide
Property Prediction: Coverage Diagnostics Under Crystal Symmetry Heterogeneity

Dataset: NOMAD 2018 benchmark (Al_x Ga_y In_{1-x-y})_2O_3, DFT-PBE
  Used as a controlled testbed for the UQ framework, not for materials discovery.
  Band-gap values carry DFT-PBE systematic underestimate (~0.5-1.5 eV vs HSE/exp).
  The benchmark contains exactly 6 crystal lattice symmetries (spacegroups 12, 33,
  167, 194, 206, 227), enabling leave-one-lattice-out generalization tests.


=============================================================================
REPRODUCIBILITY INSTRUCTIONS
=============================================================================
1. Install dependencies:
       pip install mapie==0.8.6 optuna dscribe shap lightgbm ase tqdm joblib

2. The CSV (train.csv) is downloaded automatically from GitHub on first run.

3. For Section 6 outlier visualization and SOAP computation you need the
   NOMAD 2018 XYZ geometry files. Download them from:
     https://www.kaggle.com/c/nomad2018-predict-transparent-conductors/data
   and extract so that DATA_ROOT = 'data/train/' contains one sub-folder
   per structure ID, each with geometry.xyz inside.
   If you only want to run from the CSV (no XYZ), the SOAP section will
   skip gracefully and you can load a pre-computed SOAP cache instead.

4. All outputs (figures, CSVs, model checkpoints) are written to RESULTS_DIR.
=============================================================================
"""

# =============================================================================
# SECTION 0: CONFIGURATION  (edit these paths before running)
# =============================================================================

# --- Output / checkpoint directories (created automatically) ---
RESULTS_DIR    = 'results'                    # figures, CSVs, summary
CHECKPOINT_DIR = 'results/checkpoints'        # model .pkl files, SOAP cache
DATA_DIR       = 'data'                       # CSV cache lives here

# --- NOMAD geometry files (only needed for SOAP + outlier visualization) ---
# Download from Kaggle and extract so geometry.xyz files live at:
#   DATA_ROOT/<id>/geometry.xyz
DATA_ROOT = 'data/train'

# --- Remote CSV (downloaded automatically on first run) ---
TRAIN_URL = ('https://raw.githubusercontent.com/csutton7/'
             'nomad_2018_kaggle_dataset/master/train.csv')

# --- Cached paths (derived from the directories above) ---
import os
SOAP_CACHE     = os.path.join(CHECKPOINT_DIR, 'soap_features_v5.npz')
DATA_CSV_CACHE = os.path.join(DATA_DIR,       'train_cache.pkl')
PREP_CACHE     = os.path.join(CHECKPOINT_DIR, 'data_prep_v5.npz')
HPO_CACHE      = os.path.join(CHECKPOINT_DIR, 'best_params_v5.json')
STAGE_FILE     = os.path.join(RESULTS_DIR,    'pipeline_stage_v5.json')

# =============================================================================
# SECTION 0B: IMPORTS AND GLOBAL CONSTANTS
# =============================================================================

import json, gc, warnings, shutil, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.stats import spearmanr, kendalltau, pearsonr
from collections import Counter

from sklearn.model_selection import (
    StratifiedShuffleSplit, GroupKFold, KFold, train_test_split
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import lightgbm as lgb
import optuna
import joblib

from mapie.regression import MapieRegressor
from mapie.metrics import regression_coverage_score

from ase.io import read as ase_read, write as ase_write
from ase.visualize.plot import plot_atoms
from dscribe.descriptors import SOAP
from tqdm.auto import tqdm
import shap

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

SEED  = 42
ALPHA = 0.10   # target coverage = 1 - ALPHA = 90%
np.random.seed(SEED)

LATTICE_MAP = {
    12: 'C/2m', 33: 'Pna21', 167: 'R3c',
    194: 'P63/mmc', 206: 'Ia-3', 227: 'Fd-3m',
}

# Create all directories
for d in [RESULTS_DIR, CHECKPOINT_DIR, DATA_DIR]:
    os.makedirs(d, exist_ok=True)

# ---- Stage tracker (crash-safe resume) ----
def load_stage():
    if os.path.exists(STAGE_FILE):
        with open(STAGE_FILE) as f:
            return json.load(f)
    return {}

def save_stage(key, value=True):
    s = load_stage()
    s[key] = value
    with open(STAGE_FILE, 'w') as f:
        json.dump(s, f, indent=2)
    print(f"  [checkpoint] stage '{key}' saved.")

stage = load_stage()
print(f"Stage tracker loaded. Completed stages: {list(stage.keys())}")

# ---- GPU detection ----
USE_GPU = False
GPU_PARAMS = {}
try:
    import subprocess
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        USE_GPU = True
        GPU_PARAMS = {
            'device_type': 'gpu',
            'gpu_platform_id': 0,
            'gpu_device_id': 0,
        }
        print(f"GPU detected. LightGBM will use device_type='gpu'.")
        print(result.stdout.split('\n')[8])
    else:
        print("No GPU detected. Running on CPU.")
except Exception:
    print("nvidia-smi not available. Running on CPU.")

# n_jobs: use -1 on CPU (all cores), 1 on GPU (GPU manages parallelism)
LGBM_JOBS = 1 if USE_GPU else -1



NATURE_DPI    = 600
NATURE_FONT   = 'Arial'
NATURE_FS_SM  = 7
NATURE_FS_MD  = 8
NATURE_FS_LG  = 9
NATURE_LW     = 0.75
NATURE_SINGLE = 3.5
NATURE_DOUBLE = 7.2
NATURE_HEIGHT = 2.8

# Colour palette (colour-blind safe, Nature Materials style)
PALETTE = {
    'blue':   '#2166AC',
    'red':    '#D6604D',
    'orange': '#F4A582',
    'green':  '#4DAC26',
    'purple': '#762A83',
    'grey':   '#878787',
}

def set_nature_style():
    """Apply Nature-journal matplotlib rcParams globally."""
    plt.rcParams.update({
        'font.family':        'sans-serif',
        'font.sans-serif':    [NATURE_FONT, 'Helvetica', 'DejaVu Sans'],
        'font.size':          NATURE_FS_MD,
        'axes.titlesize':     NATURE_FS_LG,
        'axes.labelsize':     NATURE_FS_MD,
        'xtick.labelsize':    NATURE_FS_SM,
        'ytick.labelsize':    NATURE_FS_SM,
        'legend.fontsize':    NATURE_FS_SM,
        'legend.frameon':     False,
        'axes.linewidth':     NATURE_LW,
        'axes.spines.top':    False,
        'axes.spines.right':  False,
        'xtick.major.width':  NATURE_LW,
        'ytick.major.width':  NATURE_LW,
        'xtick.major.size':   3,
        'ytick.major.size':   3,
        'lines.linewidth':    1.2,
        'figure.dpi':         NATURE_DPI,
        'savefig.dpi':        NATURE_DPI,
        'savefig.bbox':       'tight',
        'savefig.transparent':False,
        'pdf.fonttype':       42,   # embed fonts in PDF
        'ps.fonttype':        42,
    })

set_nature_style()

def save_fig(fig, fname, subdir=RESULTS_DIR):
    """Save figure locally (PDF + JPG at 600 DPI)."""
    os.makedirs(subdir, exist_ok=True)
    local_path = os.path.join(subdir, fname)
    fig.savefig(local_path, dpi=NATURE_DPI, bbox_inches='tight', facecolor='white')
    # Also save a JPG companion
    jpg_path = local_path.replace('.pdf', '.jpg')
    fig.savefig(jpg_path, dpi=NATURE_DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Figure saved: {local_path}")

print(f"\nSetup complete  [v5 | GPU={USE_GPU} | DPI={NATURE_DPI}]")

# =============================================================================
# SECTION 1: DATA LOADING
# =============================================================================
print("\n--- Section 1: Loading Data ---")

if os.path.exists(DATA_CSV_CACHE):
    full_df = pd.read_pickle(DATA_CSV_CACHE)
    print(f"Loaded CSV from cache: {len(full_df)} structures.")
else:
    print(f"Downloading CSV from {TRAIN_URL} ...")
    full_df = pd.read_csv(TRAIN_URL)
    os.makedirs(DATA_DIR, exist_ok=True)
    full_df.to_pickle(DATA_CSV_CACHE)
    print(f"Loaded {len(full_df)} structures; CSV cache saved.")

COMP_COLS = ['percent_atom_al', 'percent_atom_ga', 'percent_atom_in',
             'spacegroup', 'number_of_total_atoms']
missing = [c for c in COMP_COLS if c not in full_df.columns]
if missing:
    print(f"WARNING: Missing columns: {missing}")

# =============================================================================
# SECTION 2: SOAP DESCRIPTORS
# =============================================================================
print("\n--- Section 2: SOAP Features ---")

if os.path.exists(SOAP_CACHE):
    X_soap = np.load(SOAP_CACHE)['features']
    print(f"Loaded cached SOAP features: {X_soap.shape}")
else:
    if not os.path.isdir(DATA_ROOT):
        raise FileNotFoundError(
            f"SOAP cache not found and DATA_ROOT does not exist: {DATA_ROOT}\n"
            "Please download the NOMAD 2018 train geometry files from Kaggle\n"
            "and extract them so that DATA_ROOT/<id>/geometry.xyz exists."
        )
    print("Computing SOAP from scratch (~20 min on CPU).")
    all_atoms = []
    for fid in tqdm(full_df['id'].values, desc='Reading .xyz'):
        path = os.path.join(DATA_ROOT, str(fid), 'geometry.xyz')
        try:
            all_atoms.append(ase_read(path, format='aims'))
        except Exception:
            all_atoms.append(None)

    species_set = set()
    for a in all_atoms:
        if a is not None:
            species_set.update(a.get_chemical_symbols())

    soap_gen = SOAP(species=sorted(species_set), periodic=True,
                    r_cut=6.0, n_max=9, l_max=8, sparse=False)

    def aggregate(atomic_soap):
        if atomic_soap is None or len(atomic_soap) == 0:
            return None
        return np.concatenate([
            np.mean(atomic_soap, axis=0), np.std(atomic_soap,  axis=0),
            np.min(atomic_soap,  axis=0), np.max(atomic_soap,  axis=0),
        ])

    X_soap = np.array([
        aggregate(soap_gen.create(a) if a is not None else None)
        for a in tqdm(all_atoms, desc='SOAP')
    ])
    np.savez_compressed(SOAP_CACHE, features=X_soap)
    print(f"SOAP cache saved: {X_soap.shape}")
    save_stage('soap_computed')

# =============================================================================
# SECTION 3: DATA PREPARATION  (fully checkpointed)
# =============================================================================
print("\n--- Section 3: Data Preparation ---")

if os.path.exists(PREP_CACHE):
    print("  Loading prepared data from checkpoint...")
    d = np.load(PREP_CACHE, allow_pickle=True)
    X_train_scaled = d['X_train_scaled']
    X_test_scaled  = d['X_test_scaled']
    y_train_form   = d['y_train_form']
    y_test_form    = d['y_test_form']
    y_train_band   = d['y_train_band']
    y_test_band    = d['y_test_band']
    in_frac_train  = d['in_frac_train']
    in_frac_test   = d['in_frac_test']
    sg_train       = d['sg_train']
    sg_test        = d['sg_test']
    groups_train   = d['groups_train']
    groups_test    = d['groups_test']
    train_idx      = d['train_idx']
    test_idx       = d['test_idx']
    SOAP_DIM       = int(d['SOAP_DIM'])
    scaler         = joblib.load(os.path.join(CHECKPOINT_DIR, 'scaler_v5.pkl'))
    df_valid       = pd.read_pickle(os.path.join(CHECKPOINT_DIR, 'df_valid_v5.pkl'))
    sg_names       = np.array([LATTICE_MAP.get(int(sg), f'SG{sg}')
                               for sg in sg_test])
    print(f"  Train: {X_train_scaled.shape}  Test: {X_test_scaled.shape}")
else:
    y_form_raw = full_df['formation_energy_ev_natom'].values.astype(np.float32)
    y_band_raw = full_df['bandgap_energy_ev'].values.astype(np.float32)

    valid_mask = (
        np.isfinite(y_form_raw) & np.isfinite(y_band_raw) &
        ~np.isnan(X_soap).any(axis=1)
    )
    X        = X_soap[valid_mask].astype(np.float32)
    y_form   = y_form_raw[valid_mask]
    y_band   = y_band_raw[valid_mask]
    df_valid = full_df[valid_mask].reset_index(drop=True)
    df_valid.to_pickle(os.path.join(CHECKPOINT_DIR, 'df_valid_v5.pkl'))

    # Spacegroup labels
    if 'spacegroup' in df_valid.columns:
        sg_labels = df_valid['spacegroup'].values.astype(int)
    else:
        sg_labels = np.zeros(len(df_valid), dtype=int)

    # Indium fraction
    in_frac_all = (df_valid['percent_atom_in'].values.astype(np.float32)
                   if 'percent_atom_in' in df_valid.columns
                   else np.zeros(len(df_valid), dtype=np.float32))

    # Composition groups (Task C)
    if 'spacegroup' in df_valid.columns and 'percent_atom_al' in df_valid.columns:
        def make_group(row):
            return (f"SG{int(row.get('spacegroup',0))}_"
                    f"Al{round(row.get('percent_atom_al',0),1)}_"
                    f"Ga{round(row.get('percent_atom_ga',0),1)}")
        comp_groups = df_valid.apply(make_group, axis=1).values
    else:
        comp_groups = sg_labels.astype(str)

    print("Spacegroup distribution:")
    for sg, name in LATTICE_MAP.items():
        print(f"  {name:10s} (SG {sg:3d}): {np.sum(sg_labels==sg)}")

    # Stratified split (matches Sutton et al. equal-lattice distribution)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_idx, test_idx = next(sss.split(X, sg_labels))

    X_train, X_test           = X[train_idx], X[test_idx]
    y_train_form, y_test_form = y_form[train_idx], y_form[test_idx]
    y_train_band, y_test_band = y_band[train_idx], y_band[test_idx]
    in_frac_train             = in_frac_all[train_idx]
    in_frac_test              = in_frac_all[test_idx]
    groups_train              = comp_groups[train_idx]
    groups_test               = comp_groups[test_idx]
    sg_train                  = sg_labels[train_idx]
    sg_test                   = sg_labels[test_idx]

    scaler         = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_test_scaled  = scaler.transform(X_test).astype(np.float32)
    SOAP_DIM       = X_train_scaled.shape[1] // 4

    np.savez_compressed(PREP_CACHE,
        X_train_scaled=X_train_scaled, X_test_scaled=X_test_scaled,
        y_train_form=y_train_form, y_test_form=y_test_form,
        y_train_band=y_train_band, y_test_band=y_test_band,
        in_frac_train=in_frac_train, in_frac_test=in_frac_test,
        sg_train=sg_train, sg_test=sg_test,
        groups_train=groups_train, groups_test=groups_test,
        train_idx=train_idx, test_idx=test_idx,
        SOAP_DIM=np.array(SOAP_DIM),
    )
    joblib.dump(scaler, os.path.join(CHECKPOINT_DIR, 'scaler_v5.pkl'), compress=3)
    save_stage('data_prep')
    del X
    gc.collect()

    sg_names = np.array([LATTICE_MAP.get(int(sg), f'SG{sg}') for sg in sg_test])
    print(f"Train: {X_train_scaled.shape}  Test: {X_test_scaled.shape}")

# Feature names (rebuilt from SOAP_DIM)
FEATURE_NAMES = (
    [f'Mean_{i:04d}' for i in range(SOAP_DIM)] +
    [f'Std_{i:04d}'  for i in range(SOAP_DIM)] +
    [f'Min_{i:04d}'  for i in range(SOAP_DIM)] +
    [f'Max_{i:04d}'  for i in range(SOAP_DIM)]
)

# =============================================================================
# SECTION 4: HPO  (Optuna TPE + GroupKFold + GPU)
# =============================================================================
print("\n--- Section 4: Hyperparameter Optimisation (GroupKFold + GPU) ---")

if os.path.exists(HPO_CACHE):
    with open(HPO_CACHE) as f:
        hp_cache = json.load(f)
    best_params_form = hp_cache['formation']
    best_params_band = hp_cache['bandgap']
    print(f"  Loaded HPO params from checkpoint.")
    print(f"  Formation: {best_params_form}")
    print(f"  Band Gap:  {best_params_band}")
else:
    def optimize_lgbm_gkf(X, y, groups, study_name, n_trials=20):
        """
        Optuna TPE + GroupKFold(n=6) + GPU.
        SQLite storage in CHECKPOINT_DIR = resumable if the process is interrupted.
        """
        storage = f"sqlite:///{os.path.join(CHECKPOINT_DIR, study_name)}.db"
        gkf     = GroupKFold(n_splits=6)

        def objective(trial):
            gc.collect()
            base_params = {
                'objective':         'regression',
                'metric':            'mae',
                'verbosity':         -1,
                'boosting_type':     'gbdt',
                'n_jobs':            LGBM_JOBS,
                'seed':              SEED + trial.number,
                'max_bin':           63,
                'learning_rate':     trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
                'num_leaves':        trial.suggest_int('num_leaves', 31, 127),
                'max_depth':         trial.suggest_int('max_depth', 5, 9),
                'min_child_samples': trial.suggest_int('min_child_samples', 20, 60),
                'subsample':         trial.suggest_float('subsample', 0.65, 0.95),
                'colsample_bytree':  trial.suggest_float('colsample_bytree', 0.65, 0.95),
                'reg_alpha':         trial.suggest_float('reg_alpha', 1e-4, 1.0, log=True),
                'reg_lambda':        trial.suggest_float('reg_lambda', 1e-4, 1.0, log=True),
            }
            base_params.update(GPU_PARAMS)

            fold_maes = []
            for tr_i, val_i in gkf.split(X, y, groups=groups):
                m = lgb.LGBMRegressor(**base_params, n_estimators=700)
                m.fit(X[tr_i], y[tr_i],
                      eval_set=[(X[val_i], y[val_i])],
                      callbacks=[lgb.early_stopping(40, verbose=False),
                                 lgb.log_evaluation(-1)])
                fold_maes.append(mean_absolute_error(y[val_i], m.predict(X[val_i])))
                del m
                gc.collect()
            return float(np.mean(fold_maes))

        study = optuna.create_study(
            study_name=study_name, storage=storage,
            direction='minimize', load_if_exists=True,
            sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=10),
        )
        done      = sum(1 for t in study.trials
                        if t.state == optuna.trial.TrialState.COMPLETE)
        remaining = max(0, n_trials - done)
        if remaining > 0:
            print(f"    Running {remaining} remaining trials for {study_name}...")
            study.optimize(objective, n_trials=remaining, gc_after_trial=True)
        best = study.best_params if study.trials else {
            'learning_rate': 0.05, 'num_leaves': 63, 'max_depth': 8,
            'min_child_samples': 30, 'subsample': 0.8, 'colsample_bytree': 0.8,
            'reg_alpha': 0.01, 'reg_lambda': 0.01,
        }
        print(f"    Best MAE ({study_name}): {study.best_value:.5f}")
        return best

    best_params_form = optimize_lgbm_gkf(
        X_train_scaled, y_train_form, sg_train, 'lgbm_form_v5', n_trials=25)
    gc.collect()
    best_params_band = optimize_lgbm_gkf(
        X_train_scaled, y_train_band, sg_train, 'lgbm_band_v5', n_trials=25)

    with open(HPO_CACHE, 'w') as f:
        json.dump({'formation': best_params_form, 'bandgap': best_params_band}, f, indent=2)
    save_stage('hpo_complete')
    print(f"  HPO params saved.")

print(f"  Formation: {best_params_form}")
print(f"  Band Gap:  {best_params_band}")
print(f"  GPU used for HPO: {USE_GPU}")

# =============================================================================
# SECTION 5: CONFORMAL PREDICTION TRAINING  (CV+, GroupKFold + GPU)
# =============================================================================
print("\n--- Section 5: Training Conformal Predictors (CV+ GroupKFold + GPU) ---")

def build_lgbm(params):
    """Build LightGBM regressor with GPU params if available."""
    clean = {k: v for k, v in params.items()
             if k not in ('metric', 'objective')}
    clean.update(GPU_PARAMS)
    return lgb.LGBMRegressor(
        **clean, n_estimators=700, n_jobs=LGBM_JOBS,
        verbose=-1, max_bin=63, force_col_wise=True,
    )

def train_conformal_gkf(X, y, params, groups, target_name, ckpt_name):
    """Train CV+ with GroupKFold(n=6). Checkpoint auto-saved to CHECKPOINT_DIR."""
    ckpt_path = os.path.join(CHECKPOINT_DIR, ckpt_name)
    if os.path.exists(ckpt_path):
        print(f"  Loading checkpoint: {ckpt_name}")
        return joblib.load(ckpt_path)

    mapie = MapieRegressor(
        estimator=build_lgbm(params),
        cv=GroupKFold(n_splits=6),
        method='plus',
        n_jobs=1,
    )
    print(f"  Training {target_name} (CV+ GroupKFold 6 folds, GPU={USE_GPU})...")
    t0 = time.time()
    mapie.fit(X, y, groups=groups)
    print(f"  Done in {(time.time()-t0)/60:.1f} min.")
    joblib.dump(mapie, ckpt_path, compress=3)
    print(f"  Checkpoint saved: {ckpt_path}")
    return mapie

mapie_form = train_conformal_gkf(
    X_train_scaled, y_train_form, best_params_form, sg_train,
    'Formation Energy', 'mapie_form_v5.pkl')
gc.collect()
mapie_band = train_conformal_gkf(
    X_train_scaled, y_train_band, best_params_band, sg_train,
    'Band Gap', 'mapie_band_v5.pkl')
save_stage('cp_training_complete')
print("Conformal predictors trained and checkpointed.")

# =============================================================================
# SECTION 6: EVALUATION  (Nature figures, 600 DPI)
# =============================================================================
print("\n--- Section 6: Evaluation ---")

def rmsle(y_true, y_pred):
    y_true_p = np.clip(y_true, 0, None)
    y_pred_p = np.clip(y_pred, 0, None)
    return float(np.sqrt(mean_squared_error(np.log1p(y_true_p), np.log1p(y_pred_p))))

def evaluate_target(mapie_model, X_te, y_te, target_name, alpha=ALPHA):
    y_pred, y_pis = mapie_model.predict(X_te, alpha=alpha)
    lo, hi = y_pis[:, 0, 0], y_pis[:, 1, 0]
    widths  = hi - lo
    abs_err = np.abs(y_te - y_pred)
    r_p, p_p = pearsonr(widths, abs_err)
    metrics = {
        'y_pred': y_pred, 'lo': lo, 'hi': hi, 'widths': widths,
        'mae':      mean_absolute_error(y_te, y_pred),
        'rmse':     float(np.sqrt(mean_squared_error(y_te, y_pred))),
        'rmsle':    rmsle(y_te, y_pred),
        'r2':       r2_score(y_te, y_pred),
        'coverage': regression_coverage_score(y_te, lo, hi),
        'mean_width': float(np.mean(widths)),
        'pearson_r': r_p, 'pearson_p': p_p,
    }
    print(f"\n  [{target_name}] alpha={alpha}")
    for k in ('mae','rmse','rmsle','r2','coverage','mean_width','pearson_r'):
        print(f"    {k:<14} = {metrics[k]:.5f}")
    return metrics

res_form = evaluate_target(mapie_form, X_test_scaled, y_test_form,
                           'Formation Energy (eV/atom)')
res_band = evaluate_target(mapie_band, X_test_scaled, y_test_band,
                           'Band Gap (eV)')

# ---- Parity plots (Nature double-column) ----
def parity_plot_nature(y_true, y_pred, lo, hi, xlabel, ylabel, panel_label, fname):
    """Nature-style parity plot: 600 DPI, no top/right spine, 8pt fonts."""
    fig, ax = plt.subplots(figsize=(NATURE_SINGLE, NATURE_SINGLE))

    abs_err = np.abs(y_true - y_pred)
    sc = ax.scatter(y_true, y_pred, c=abs_err, cmap='plasma',
                    s=6, alpha=0.6, linewidths=0, rasterized=True)
    cb = fig.colorbar(sc, ax=ax, pad=0.02, shrink=0.85)
    cb.set_label('|Error|', fontsize=NATURE_FS_SM)
    cb.ax.tick_params(labelsize=NATURE_FS_SM)

    order = np.argsort(y_true)
    ax.fill_between(y_true[order], lo[order], hi[order],
                    alpha=0.15, color=PALETTE['blue'],
                    label='90% CP interval', zorder=0)

    lim = [min(y_true.min(), y_pred.min()) - 0.05,
           max(y_true.max(), y_pred.max()) + 0.05]
    ax.plot(lim, lim, color=PALETTE['grey'], lw=NATURE_LW, ls='--', zorder=5)
    ax.set_xlim(lim); ax.set_ylim(lim)

    ax.set_xlabel(xlabel, fontsize=NATURE_FS_MD)
    ax.set_ylabel(ylabel, fontsize=NATURE_FS_MD)
    ax.text(0.0, 1.04, panel_label, transform=ax.transAxes,
            fontsize=NATURE_FS_LG, fontweight='bold', va='bottom', ha='left')
    ax.legend(fontsize=NATURE_FS_SM, loc='upper left')

    r2  = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    cov = regression_coverage_score(y_true, lo, hi)
    ax.text(0.97, 0.06,
            f'MAE = {mae:.4f}\nR² = {r2:.4f}\nCov. = {cov:.3f}',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=NATURE_FS_SM,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8,
                      ec=PALETTE['grey'], lw=0.5))

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, fname)

parity_plot_nature(
    y_test_form, res_form['y_pred'], res_form['lo'], res_form['hi'],
    'DFT formation energy (eV atom⁻¹)',
    'LGBM+CV+ prediction (eV atom⁻¹)',
    '(a)', 'parity_formation_v5.pdf')

parity_plot_nature(
    y_test_band, res_band['y_pred'], res_band['lo'], res_band['hi'],
    'DFT band gap (eV)',
    'LGBM+CV+ prediction (eV)',
    '(b)', 'parity_bandgap_v5.pdf')

# ---- 6.1 Conditional coverage (Task B) ----
print("\n--- Section 6.1: Conditional Coverage (Task B) ---")

def cond_coverage_table(y_te, lo, hi, y_pred, subset_defs, tag):
    rows = []
    for lbl, msk in subset_defs:
        if msk.sum() == 0: continue
        rows.append({
            'Subset': lbl, 'N': int(msk.sum()),
            'Coverage': round(regression_coverage_score(y_te[msk], lo[msk], hi[msk]), 4),
            'MAE':      round(mean_absolute_error(y_te[msk], y_pred[msk]), 4),
            'MeanWidth':round(float(np.mean(hi[msk]-lo[msk])), 4),
        })
        print(f"  {lbl:<45s} n={int(msk.sum()):4d}  "
              f"cov={rows[-1]['Coverage']:.3f}  "
              f"MAE={rows[-1]['MAE']:.4f}  width={rows[-1]['MeanWidth']:.4f}")
    df = pd.DataFrame(rows)
    path = os.path.join(RESULTS_DIR, f'cond_coverage_{tag}_v5.csv')
    df.to_csv(path, index=False)
    return df

form_subsets = [
    ('Stable (E_f <= 0.20 eV/atom)', y_test_form <= 0.20),
    ('Unstable (E_f > 0.20 eV/atom)', y_test_form > 0.20),
    ('Strongly stable (E_f < -1.0 eV/atom)', y_test_form < -1.0),
]
band_subsets = [
    ('Metal (E_g < 0.10 eV)', y_test_band < 0.10),
    ('Semiconductor (0.10 <= E_g < 3.0 eV)',
     (y_test_band >= 0.10) & (y_test_band < 3.0)),
    ('Insulator (E_g >= 3.0 eV)', y_test_band >= 3.0),
    ('TCO target window (2.0-4.0 eV)',
     (y_test_band >= 2.0) & (y_test_band <= 4.0)),
]

print("\n  Formation energy subsets:")
df_form_cond = cond_coverage_table(
    y_test_form, res_form['lo'], res_form['hi'],
    res_form['y_pred'], form_subsets, 'formation')

print("\n  Band gap subsets:")
df_band_cond = cond_coverage_table(
    y_test_band, res_band['lo'], res_band['hi'],
    res_band['y_pred'], band_subsets, 'bandgap')

# ---- 6.2 Leakage defence (Task C) ----
print("\n--- Section 6.2: Leakage Defence (Task C) ---")

gc_df = pd.DataFrame({
    'group':    groups_test,
    'err_form': np.abs(y_test_form - res_form['y_pred']),
    'err_band': np.abs(y_test_band - res_band['y_pred']),
})
group_stats = gc_df.groupby('group').agg(
    n=('err_form','count'),
    mae_form=('err_form','mean'),
    mae_band=('err_band','mean'),
).reset_index()
focal = group_stats.sort_values('n', ascending=False).iloc[0]
rare  = group_stats.sort_values('n').iloc[0]
ratio = focal['mae_form'] / rare['mae_form'] if rare['mae_form'] > 0 else np.nan
print(f"  MAE ratio frequent/rare: {ratio:.2f}  [<2.0 = no severe leakage]")
group_stats.to_csv(os.path.join(RESULTS_DIR, 'leakage_defence_v5.csv'), index=False)

# ---- 6.3 Spacegroup-stratified coverage ----
print("\n--- Section 6.3: Spacegroup-Stratified Coverage (Novel) ---")

sg_rows_form, sg_rows_band = [], []
print(f"  {'Lattice':<10} {'N':>5}  {'Cov_form':>9} {'W_form':>8}  "
      f"{'Cov_band':>9} {'W_band':>8}  {'MAE_form':>9} {'MAE_band':>9}")
print("  " + "-"*80)

for sg_int, sg_name in LATTICE_MAP.items():
    msk = sg_test == sg_int
    if not msk.any(): continue
    n     = int(msk.sum())
    cov_f = regression_coverage_score(y_test_form[msk], res_form['lo'][msk], res_form['hi'][msk])
    w_f   = float(np.mean(res_form['hi'][msk] - res_form['lo'][msk]))
    mae_f = mean_absolute_error(y_test_form[msk], res_form['y_pred'][msk])
    cov_b = regression_coverage_score(y_test_band[msk], res_band['lo'][msk], res_band['hi'][msk])
    w_b   = float(np.mean(res_band['hi'][msk] - res_band['lo'][msk]))
    mae_b = mean_absolute_error(y_test_band[msk], res_band['y_pred'][msk])
    print(f"  {sg_name:<10} {n:>5}  {cov_f:>9.3f} {w_f:>8.4f}  "
          f"{cov_b:>9.3f} {w_b:>8.4f}  {mae_f:>9.4f} {mae_b:>9.4f}")
    sg_rows_form.append({'Lattice':sg_name,'SG':sg_int,'N':n,
                         'Coverage':round(cov_f,4),'MeanWidth':round(w_f,4),'MAE':round(mae_f,4)})
    sg_rows_band.append({'Lattice':sg_name,'SG':sg_int,'N':n,
                         'Coverage':round(cov_b,4),'MeanWidth':round(w_b,4),'MAE':round(mae_b,4)})

df_sg_form = pd.DataFrame(sg_rows_form)
df_sg_band = pd.DataFrame(sg_rows_band)
df_sg_form.to_csv(os.path.join(RESULTS_DIR,'sg_coverage_formation_v5.csv'), index=False)
df_sg_band.to_csv(os.path.join(RESULTS_DIR,'sg_coverage_bandgap_v5.csv'), index=False)

# Nature figure: spacegroup-stratified coverage
fig, axes = plt.subplots(1, 2, figsize=(NATURE_DOUBLE, NATURE_HEIGHT))

for i, (ax, df_sg, tit) in enumerate(zip(axes,
        [df_sg_form, df_sg_band],
        ['Formation energy', 'Band gap'])):
    x    = np.arange(len(df_sg))
    bars = ax.bar(x, df_sg['Coverage'], color=PALETTE['blue'],
                  alpha=0.75, width=0.6, zorder=3)
    ax.axhline(1 - ALPHA, color=PALETTE['red'], ls='--',
               lw=NATURE_LW, label=f'Nominal {(1-ALPHA)*100:.0f}%', zorder=4)

    bold_color = '#d95f02'
    ax2 = ax.twinx()
    ax2.plot(x, df_sg['MeanWidth'], 'o-', color=bold_color,
             lw=NATURE_LW, ms=4, label='Mean width', zorder=5)
    ax2.set_ylabel('Mean interval width', fontsize=NATURE_FS_SM, color=bold_color)
    ax2.tick_params(axis='y', labelcolor=bold_color, labelsize=NATURE_FS_SM)
    ax2.spines['right'].set_visible(True)
    ax2.spines['right'].set_linewidth(NATURE_LW)

    ax.set_xticks(x)
    ax.set_xticklabels(df_sg['Lattice'], rotation=45, ha='right', va='top',
                       fontsize=NATURE_FS_SM)
    ax.tick_params(axis='x', pad=2)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel('Empirical coverage', fontsize=NATURE_FS_MD)

    for xi, (_, row) in zip(x, df_sg.iterrows()):
        ax.text(xi, row['Coverage'] + 0.02, f'n={row["N"]}',
                ha='center', va='bottom', fontsize=5, zorder=10,
                bbox=dict(facecolor='white', alpha=0.85, edgecolor='none', pad=1))

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2,
              fontsize=NATURE_FS_SM, loc='lower right',
              bbox_to_anchor=(1.0, 1.04), frameon=False, ncol=2)

    panel_label = '(a)' if i == 0 else '(b)'
    ax.text(0.0, 1.04, panel_label, transform=ax.transAxes,
            fontsize=NATURE_FS_LG, fontweight='bold', va='bottom', ha='left')
    ax.grid(axis='y', lw=0.4, alpha=0.4, zorder=0)

plt.suptitle('Spacegroup-stratified CP coverage\n'
             '(GroupKFold calibration -- each bar = one held-out lattice)',
             fontsize=NATURE_FS_LG, y=1.08)
plt.tight_layout()
save_fig(fig, 'sg_coverage_v5.pdf')

# ---- 6.4 Interval-width informativeness ----
print("\n--- Section 6.4: Interval Width Informativeness ---")
#
# Novel analysis: are wider CP intervals a reliable signal that prediction
# errors will be larger? If Pearson(interval_width, |error|) > 0 and
# statistically significant, the intervals are *informative* (not just valid).
# This goes beyond the marginal coverage guarantee to show the intervals have
# practical diagnostic value.

def width_informativeness_plot(y_te, y_pred, widths, target_name, panel_label_1, panel_label_2, fname_tag):
    abs_err   = np.abs(y_te - y_pred)
    r_p, p_p  = pearsonr(widths, abs_err)
    q_labels  = pd.qcut(widths, q=4,
                         labels=['Q1 (narrow)', 'Q2', 'Q3', 'Q4 (wide)'])
    q_stats   = (pd.DataFrame({'q': q_labels, 'e': abs_err})
                 .groupby('q', observed=True)['e'].mean())

    fig, (ax1, ax2) = plt.subplots(1, 2,
        figsize=(NATURE_DOUBLE, NATURE_HEIGHT))

    # Scatter: width vs |error|
    ax1.scatter(widths, abs_err, s=4, alpha=0.25, c=PALETTE['blue'],
                linewidths=0, rasterized=True)
    m, b = np.polyfit(widths, abs_err, 1)
    xfit = np.linspace(widths.min(), widths.max(), 300)
    ax1.plot(xfit, m*xfit + b, color=PALETTE['red'], lw=1.2,
             label=f'Pearson r = {r_p:.3f}\np = {p_p:.1e}')
    ax1.set_xlabel('CP interval width', fontsize=NATURE_FS_MD)
    ax1.set_ylabel('|prediction error|', fontsize=NATURE_FS_MD)
    ax1.text(0.0, 1.04, panel_label_1, transform=ax1.transAxes,
             fontsize=NATURE_FS_LG, fontweight='bold', va='bottom', ha='left')
    ax1.legend(fontsize=NATURE_FS_SM)
    ax1.grid(lw=0.4, alpha=0.4)

    # Bar: mean |error| per width quartile
    ax2.bar(range(4), q_stats.values, color=PALETTE['blue'],
            alpha=0.75, width=0.6)
    ax2.set_xticks(range(4))
    ax2.set_xticklabels(q_stats.index.tolist(), fontsize=NATURE_FS_SM,
                         rotation=20, ha='right')
    ax2.set_ylabel('Mean |error|', fontsize=NATURE_FS_MD)
    ax2.text(0.0, 1.04, panel_label_2, transform=ax2.transAxes,
             fontsize=NATURE_FS_LG, fontweight='bold', va='bottom', ha='left')
    ax2.grid(axis='y', lw=0.4, alpha=0.4)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, f'width_informativeness_{fname_tag}_v5.pdf')
    print(f"  [{target_name}] Pearson r={r_p:.4f}  p={p_p:.2e}")

width_informativeness_plot(
    y_test_form, res_form['y_pred'], res_form['widths'],
    'Formation energy', '(a)', '(b)', 'form')

width_informativeness_plot(
    y_test_band, res_band['y_pred'], res_band['widths'],
    'Band gap', '(c)', '(d)', 'band')

save_stage('evaluation_complete')

# ---- Conditional Coverage Figure (Task B -- paper-ready) ----
print("\n--- Section 6.5: Conditional Coverage Figure (Task B) ---")

df_form_cond['Target'] = 'Formation energy (eV/atom)'
df_band_cond['Target'] = 'Band gap (eV)'

col_order = ['Target', 'Subset', 'N', 'Coverage', 'MeanWidth', 'MAE']
combined = pd.concat([df_form_cond[col_order], df_band_cond[col_order]], ignore_index=True)
combined['BelowNominal'] = combined['Coverage'] < (1 - ALPHA)

print("\n=== Combined Conditional Coverage Table ===")
print(combined.to_string(index=False))
combined.to_csv(os.path.join(RESULTS_DIR, 'conditional_coverage_combined.csv'), index=False)

# LaTeX table snippet
print("\n=== LaTeX snippet ===\n")
print(r"\begin{table}[ht]")
print(r"\caption{\textbf{Conditional coverage by physical subset at $\alpha = 0.10$ (90\% nominal).}")
print(r"Coverage values below the 0.90 nominal threshold are marked $^*$.}")
print(r"\label{tab:cond_coverage_subset}")
print(r"\begin{tabular}{llcccc}")
print(r"\toprule")
print(r"Target & Subset & $N$ & Coverage & Width & MAE \\")
print(r"\midrule")
prev_target = None
for _, row in combined.iterrows():
    if prev_target is not None and row['Target'] != prev_target:
        print(r"\midrule")
    prev_target = row['Target']
    cov_str = f"{row['Coverage']:.3f}"
    if row['BelowNominal']:
        cov_str = r"\textbf{" + cov_str + r"}$^*$"
    print(f"{row['Target']} & {row['Subset']} & {row['N']} "
          f"& {cov_str} & {row['MeanWidth']:.3f} & {row['MAE']:.4f} \\\\")
print(r"\bottomrule")
print(r"\end{tabular}")
print(r"\end{table}")

# Conditional coverage bar chart
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
panel_labels = ['a', 'b']
for i, (ax, (df_sub, unit)) in enumerate(zip(
    axes,
    [(df_form_cond, 'eV/atom'), (df_band_cond, 'eV')],
)):
    labels   = [str(label).split(' (')[0] for label in df_sub['Subset'].tolist()]
    coverage = df_sub['Coverage'].tolist()
    widths_  = df_sub['MeanWidth'].tolist()
    ns       = df_sub['N'].tolist()
    x        = np.arange(len(labels))
    bar_width = 0.55
    colors = [PALETTE['red'] if c < (1 - ALPHA) else PALETTE['blue'] for c in coverage]
    bars   = ax.bar(x, coverage, width=bar_width, color=colors,
                    edgecolor='white', linewidth=0.5, zorder=3)
    ax.axhline(1 - ALPHA, color='black', linestyle='--', linewidth=0.8, zorder=4)
    for bar, n in zip(bars, ns):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.004, f'n={n}',
                ha='center', va='bottom', fontsize=NATURE_FS_SM, color='black')
    ax2 = ax.twinx()
    ax2.plot(x, widths_, color=PALETTE['grey'], marker='o',
             markersize=3, linewidth=0.9, linestyle='-', zorder=5)
    ax2.set_ylabel(f'Mean width ({unit})', fontsize=NATURE_FS_SM)
    ax2.tick_params(axis='y', labelsize=NATURE_FS_SM)
    ax2.set_ylim(0, max(widths_) * 1.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, ha='center', fontsize=NATURE_FS_SM)
    ax.set_ylabel('Empirical coverage', fontsize=NATURE_FS_SM)
    ax.set_ylim(0.80, 1.02)
    ax.tick_params(axis='both', labelsize=NATURE_FS_SM)
    ax.yaxis.grid(True, linestyle=':', linewidth=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.text(-0.15, 1.05, f"({panel_labels[i]})", transform=ax.transAxes,
            fontsize=8, fontweight='bold', va='bottom', ha='left')

blue_patch  = mpatches.Patch(color=PALETTE['blue'],  label='Coverage >= 0.90')
red_patch   = mpatches.Patch(color=PALETTE['red'],   label='Coverage < 0.90')
nom_line    = plt.Line2D([0], [0], color='black', linestyle='--', linewidth=0.8, label='90% nominal')
width_line_ = plt.Line2D([0], [0], color=PALETTE['grey'], marker='o', markersize=3,
                          linewidth=0.9, label='Mean width (right axis)')
fig.legend(handles=[blue_patch, red_patch, nom_line, width_line_],
           fontsize=NATURE_FS_SM, loc='lower center',
           bbox_to_anchor=(0.5, -0.08), ncol=4, frameon=False)
plt.tight_layout()
save_fig(fig, 'fig_conditional_coverage_subset.pdf')

print("\n=== Interpretation notes for the paper ===")
for _, row in combined.iterrows():
    status = "BELOW NOMINAL" if row['BelowNominal'] else "OK"
    print(
        f"  [{status}] {row['Target']} | {row['Subset']} "
        f"(n={row['N']}): coverage={row['Coverage']:.3f}, "
        f"MAE={row['MAE']:.4f}, width={row['MeanWidth']:.3f}"
    )

# ---- High-uncertainty outlier visualization ----
print("\n--- Section 6.6: High-Uncertainty Outlier Visualization ---")

mask_sg227   = (sg_test == 227)
widths_sg227 = res_band['widths'][mask_sg227]
errors_sg227 = np.abs(y_test_band[mask_sg227] - res_band['y_pred'][mask_sg227])

# 2nd widest to avoid trivial edge cases
top_local_idx         = np.argsort(widths_sg227)[-2]
sg227_test_positions  = np.where(mask_sg227)[0]
top_test_pos          = sg227_test_positions[top_local_idx]
top_valid_idx         = test_idx[top_test_pos]

row_out       = df_valid.iloc[top_valid_idx]
structure_id  = int(row_out['id'])
true_bg       = float(y_test_band[top_test_pos])
pred_bg       = float(res_band['y_pred'][top_test_pos])
interval_lo   = float(res_band['lo'][top_test_pos])
interval_hi   = float(res_band['hi'][top_test_pos])
interval_w    = float(res_band['widths'][top_test_pos])
abs_error     = float(errors_sg227[top_local_idx])
top3_local    = np.argsort(widths_sg227)[-3:][::-1]
top3_ids      = [int(df_valid.iloc[test_idx[sg227_test_positions[i]]]['id'])
                 for i in top3_local]

print("=" * 60)
print("HIGH-UNCERTAINTY OUTLIER (Fd-3m, band gap, widest CP interval)")
print("=" * 60)
print(f"  Structure ID   : {structure_id}")
print(f"  Spacegroup     : Fd-3m (227)")
print(f"  True band gap  : {true_bg:.3f} eV")
print(f"  Predicted      : {pred_bg:.3f} eV")
print(f"  CP interval    : [{interval_lo:.3f}, {interval_hi:.3f}] eV")
print(f"  Interval width : {interval_w:.3f} eV")
print(f"  Absolute error : {abs_error:.3f} eV")
if 'percent_atom_in' in row_out.index:
    print(f"  Indium fraction: {row_out['percent_atom_in']:.3f}")
if 'percent_atom_al' in row_out.index:
    print(f"  Al fraction    : {row_out['percent_atom_al']:.3f}")
if 'number_of_total_atoms' in row_out.index:
    print(f"  Atoms in cell  : {int(row_out['number_of_total_atoms'])}")
print(f"\n  Top-3 widest Fd-3m structure IDs: {top3_ids}")
print("=" * 60)

geom_path = os.path.join(DATA_ROOT, str(structure_id), 'geometry.xyz')
if not os.path.exists(geom_path):
    print(f"WARNING: Geometry file not found at {geom_path}. "
          "Skipping outlier visualization.\n"
          "Download NOMAD XYZ files from Kaggle (see REPRODUCIBILITY INSTRUCTIONS).")
else:
    atoms = ase_read(geom_path, format='aims')
    print(f"\nLoaded structure: {len(atoms)} atoms")
    print(f"  Species: {set(atoms.get_chemical_symbols())}")
    print(f"  Cell (A):\n{atoms.get_cell()}")

    cif_path = os.path.join(RESULTS_DIR, f'outlier_sg227_{structure_id}.cif')
    ase_write(cif_path, atoms)
    print(f"\nCIF exported: {cif_path}")

    ROTATIONS = {
        'a-axis view': ('90x,0y,0z',  '(a)'),
        'b-axis view': ('90x,90y,0z', '(b)'),
        'c-axis view': ('0x,0y,0z',   '(c)'),
    }
    fig, axes_out = plt.subplots(1, 3, figsize=(7.0, 2.8))
    for ax_o, (view_label, (rot, panel_label)) in zip(axes_out, ROTATIONS.items()):
        plot_atoms(atoms, ax_o, radii=0.45, rotation=rot, show_unit_cell=2, colors=None)
        ax_o.set_title(f'{panel_label} {view_label}', fontsize=NATURE_FS_SM,
                       fontweight='bold', pad=3)
        ax_o.axis('off')
    legend_elements = [
        mpatches.Patch(color='#FF6347', label='O'),
        mpatches.Patch(color='#BFC2C7', label='Al'),
        mpatches.Patch(color='#6AAFB0', label='Ga'),
        mpatches.Patch(color='#A67CB5', label='In'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4,
               fontsize=NATURE_FS_SM, frameon=False, bbox_to_anchor=(0.5, -0.05))
    title_str = (
        f'Structure {structure_id} | Fd-3m (SG 227) | '
        f'E$_g$ = {true_bg:.2f} eV (pred: {pred_bg:.2f} eV) | '
        f'CP width = {interval_w:.2f} eV'
    )
    fig.suptitle(title_str, fontsize=NATURE_FS_SM, y=1.02)
    plt.tight_layout()
    save_fig(fig, f'outlier_sg227_{structure_id}_ase.pdf')
    print(f"  Outlier figure saved.")

    print(f"""
=== SUGGESTED FIGURE CAPTION ===

Figure X | Crystal structure of a representative high-uncertainty outlier
in the Fd-3m (SG 227, spinel-type) subgroup. The structure (NOMAD ID
{structure_id}) has a DFT-PBE band gap of {true_bg:.2f} eV, with a
predicted value of {pred_bg:.2f} eV and a 90% CP interval of
[{interval_lo:.2f}, {interval_hi:.2f}] eV (width = {interval_w:.2f} eV).
Absolute error: {abs_error:.2f} eV.
Colours: O (red), Al (grey), Ga (teal), In (purple).
""")

# =============================================================================
# SECTION 7: ALPHA-GRID + ENSEMBLE BASELINE
# =============================================================================
print("\n--- Section 7: Alpha-Grid + Ensemble Baseline ---")

ALPHAS = [0.05, 0.10, 0.20]

def alpha_grid_fig(model, X_te, y_te, alphas, panel_label_1, panel_label_2, fname):
    nom_covs, emp_covs, mwidths = [], [], []
    for a in alphas:
        _, pis = model.predict(X_te, alpha=a)
        lo_a, hi_a = pis[:, 0, 0], pis[:, 1, 0]
        emp_covs.append(regression_coverage_score(y_te, lo_a, hi_a))
        nom_covs.append(1 - a)
        mwidths.append(float(np.mean(hi_a - lo_a)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(NATURE_DOUBLE, NATURE_HEIGHT))

    ax1.plot([0,1],[0,1], color=PALETTE['grey'], lw=NATURE_LW, ls='--')
    ax1.plot(nom_covs, emp_covs, 'o-', color=PALETTE['blue'], lw=1.2, ms=5)
    for nc, ec, a in zip(nom_covs, emp_covs, alphas):
        ax1.annotate(f'a={a}', (nc, ec),
                     xytext=(5, 4), textcoords='offset points',
                     fontsize=NATURE_FS_SM)
    ax1.set_xlabel('Nominal coverage (1 - a)', fontsize=NATURE_FS_MD)
    ax1.set_ylabel('Empirical coverage', fontsize=NATURE_FS_MD)
    ax1.text(0.0, 1.04, panel_label_1, transform=ax1.transAxes,
             fontsize=NATURE_FS_LG, fontweight='bold', va='bottom', ha='left')
    ax1.set_xlim(0.7, 1.02); ax1.set_ylim(0.7, 1.02)
    ax1.grid(lw=0.4, alpha=0.4)

    ax2.bar([f'a={a}' for a in alphas], mwidths,
            color=PALETTE['blue'], alpha=0.75, width=0.5)
    ax2.set_ylabel('Mean interval width', fontsize=NATURE_FS_MD)
    ax2.text(0.0, 1.04, panel_label_2, transform=ax2.transAxes,
             fontsize=NATURE_FS_LG, fontweight='bold', va='bottom', ha='left')
    ax2.tick_params(axis='x', labelsize=NATURE_FS_SM)
    ax2.grid(axis='y', lw=0.4, alpha=0.4)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, fname)

    for nc, ec, mw in zip(nom_covs, emp_covs, mwidths):
        print(f"  nom={nc:.2f}  emp={ec:.3f}  mean_width={mw:.4f}")

print("\n  [Formation Energy]")
alpha_grid_fig(mapie_form, X_test_scaled, y_test_form,
               ALPHAS, '(a)', '(b)', 'alpha_grid_form_v5.pdf')

print("\n  [Band Gap]")
alpha_grid_fig(mapie_band, X_test_scaled, y_test_band,
               ALPHAS, '(c)', '(d)', 'alpha_grid_band_v5.pdf')

# --- Ensemble variance baseline ---
print("\n  [7.2] Ensemble baseline:")

def train_ensemble_gpu(X_tr, y_tr, params, n_folds=5):
    """Train one LightGBM per KFold on CPU or GPU."""
    models, kf = [], KFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    for tr_i, _ in kf.split(X_tr):
        m = build_lgbm(params)
        m.fit(X_tr[tr_i], y_tr[tr_i])
        models.append(m)
    return models

def ensemble_coverage_matched(models, X_te, y_te, target_mw):
    preds  = np.stack([m.predict(X_te) for m in models], axis=1)
    y_mean = preds.mean(axis=1)
    y_std  = preds.std(axis=1)
    scale  = (target_mw / 2.0) / max(y_std.mean(), 1e-8)
    cov    = regression_coverage_score(y_te, y_mean - scale*y_std,
                                        y_mean + scale*y_std)
    return cov, float(np.mean(2 * scale * y_std))

ENS_CKPT = os.path.join(CHECKPOINT_DIR, 'ensemble_baseline_v5.pkl')

if os.path.exists(ENS_CKPT):
    print("  Loading ensemble baseline from checkpoint...")
    ens_form, ens_band, ens_results = joblib.load(ENS_CKPT)
    for name, row_e in ens_results.items():
        print(f"  [{name}]  CV+: cov={row_e['cp_cov']:.4f} w={row_e['cp_width']:.4f} | "
              f"Ensemble: cov={row_e['ens_cov']:.4f} w={row_e['ens_width']:.4f} | "
              f"CV+ advantage: {row_e['delta']:+.4f}")
else:
    ens_form = train_ensemble_gpu(X_train_scaled, y_train_form, best_params_form)
    ens_band = train_ensemble_gpu(X_train_scaled, y_train_band, best_params_band)
    gc.collect()

    ens_results = {}
    for res, y_te, ens, name in [
            (res_form, y_test_form, ens_form, 'Formation'),
            (res_band, y_test_band, ens_band, 'Band Gap')]:
        cov_e, w_e = ensemble_coverage_matched(ens, X_test_scaled, y_te,
                                                res['mean_width'])
        delta = res['coverage'] - cov_e
        ens_results[name] = {
            'cp_cov':    res['coverage'],
            'cp_width':  res['mean_width'],
            'ens_cov':   cov_e,
            'ens_width': w_e,
            'delta':     delta,
        }
        print(f"  [{name}]  CV+: cov={res['coverage']:.4f} w={res['mean_width']:.4f} | "
              f"Ensemble: cov={cov_e:.4f} w={w_e:.4f} | CV+ advantage: {delta:+.4f}")

    joblib.dump((ens_form, ens_band, ens_results), ENS_CKPT, compress=3)
    print(f"  Checkpoint saved: {ENS_CKPT}")

save_stage('ensemble_baseline_complete')

# =============================================================================
# SECTION 8: SHAP + KENDALL TAU (Task A.2)
# =============================================================================
print("\n--- Section 8: SHAP + Kendall Tau Stability ---")

base_form = mapie_form.estimator_.single_estimator_
base_band = mapie_band.estimator_.single_estimator_

SHAP_VAL_CKPT  = os.path.join(CHECKPOINT_DIR, 'shap_values_v5.npz')
SHAP_META_CKPT = os.path.join(CHECKPOINT_DIR, 'shap_meta_v5.pkl')

def shap_beeswarm_nature(base_model, X_te, feature_names, target_name, panel_label, fname,
                          n_top=20):
    """Nature-style SHAP beeswarm: 600 DPI, 7pt fonts."""
    print(f"  Computing SHAP for {target_name}...")
    exp = shap.TreeExplainer(base_model)
    sv  = exp.shap_values(X_te)
    mean_abs = np.abs(sv).mean(axis=0)
    top_idx  = np.argsort(mean_abs)[::-1][:n_top]
    top_names = [feature_names[i] for i in top_idx]

    print(f"  Top-5 features [{target_name}]:")
    for i in range(5):
        print(f"    {i+1}. {top_names[i]}  |SHAP|={mean_abs[top_idx[i]]:.5f}")

    fig = plt.figure(figsize=(NATURE_SINGLE, NATURE_HEIGHT * 1.4))
    shap.summary_plot(sv[:, top_idx], X_te[:, top_idx],
                      feature_names=top_names,
                      max_display=n_top, show=False,
                      plot_size=None)
    main_ax = fig.axes[0]
    main_ax.tick_params(labelsize=NATURE_FS_SM)
    main_ax.set_xlabel('SHAP value (impact on model output)', fontsize=NATURE_FS_MD)
    fig.text(0.02, 0.98, panel_label, fontsize=NATURE_FS_LG,
             fontweight='bold', va='top', ha='left')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, fname)
    return sv, top_idx, top_names

if os.path.exists(SHAP_VAL_CKPT) and os.path.exists(SHAP_META_CKPT):
    print("  Loading SHAP values from checkpoint...")
    d          = np.load(SHAP_VAL_CKPT)
    sv_form    = d['sv_form']
    sv_band    = d['sv_band']
    meta       = joblib.load(SHAP_META_CKPT)
    top_form   = meta['top_form']
    top_band   = meta['top_band']
    topnames_form = meta['topnames_form']
    topnames_band = meta['topnames_band']
    print(f"  top_form[0]={topnames_form[0]}  top_band[0]={topnames_band[0]}")
else:
    sv_form, top_form, topnames_form = shap_beeswarm_nature(
        base_form, X_test_scaled, FEATURE_NAMES,
        'Formation energy', '(a)', 'shap_beeswarm_form_v5.pdf')

    sv_band, top_band, topnames_band = shap_beeswarm_nature(
        base_band, X_test_scaled, FEATURE_NAMES,
        'Band gap', '(b)', 'shap_beeswarm_band_v5.pdf')

    np.savez_compressed(SHAP_VAL_CKPT, sv_form=sv_form, sv_band=sv_band)
    joblib.dump({'top_form': top_form, 'top_band': top_band,
                 'topnames_form': topnames_form, 'topnames_band': topnames_band},
                SHAP_META_CKPT, compress=3)
    print(f"  SHAP checkpoint saved.")

# ---- Task A.2: Kendall tau ----
print("\n  [Task A.2] SHAP ranking stability (5 seeds, Kendall tau):")
STABILITY_SEEDS = [42, 123, 456, 789, 2024]
K_TOP    = 10
K_REPORT = 20

def shap_stability(X_tr, y_tr, params, X_te, feature_names, seeds,
                   k=K_TOP, base_top_idx=None, k_report=K_REPORT):
    rank_vecs = []
    for s in seeds:
        p = dict(params)
        p.update(GPU_PARAMS)
        clean = {kk: v for kk, v in p.items() if kk not in ('metric', 'objective')}
        m = lgb.LGBMRegressor(**clean, n_estimators=700, n_jobs=LGBM_JOBS,
                              verbose=-1, max_bin=63, seed=s, force_col_wise=True)
        m.fit(X_tr, y_tr)
        sv  = shap.TreeExplainer(m).shap_values(X_te)
        ma  = np.abs(sv).mean(axis=0)
        top = np.argsort(ma)[::-1][:k]
        rv  = np.zeros(len(feature_names))
        for rank, idx in enumerate(top):
            rv[idx] = rank + 1
        rank_vecs.append(rv)
        del m, sv; gc.collect()

    pairs = [(i, j)
             for i in range(len(seeds))
             for j in range(i + 1, len(seeds))]

    taus_all  = [kendalltau(rank_vecs[i], rank_vecs[j]).statistic for i, j in pairs]
    mt_all    = float(np.mean(taus_all))
    mn_all    = float(np.min(taus_all))

    if base_top_idx is not None:
        idx_k     = base_top_idx[:k_report]
        taus_topk = [
            kendalltau(
                [rank_vecs[i][f] for f in idx_k],
                [rank_vecs[j][f] for f in idx_k]
            ).statistic
            for i, j in pairs
        ]
        mt_topk = float(np.mean(taus_topk))
        mn_topk = float(np.min(taus_topk))
        print(f"    All-feature tau : Mean={mt_all:.4f}  Min={mn_all:.4f}")
        print(f"    Top-{k_report} tau      : Mean={mt_topk:.4f}  Min={mn_topk:.4f}"
              f"  [>0.9 = stable]")
    else:
        mt_topk, mn_topk = None, None
        print(f"    All-feature tau : Mean={mt_all:.4f}  Min={mn_all:.4f}"
              f"  [>0.9 = stable]")

    return mt_all, mn_all, mt_topk, mn_topk

TAU_FORM_CKPT = os.path.join(CHECKPOINT_DIR, 'shap_tau_form_v5.pkl')
TAU_BAND_CKPT = os.path.join(CHECKPOINT_DIR, 'shap_tau_band_v5.pkl')

if os.path.exists(TAU_FORM_CKPT):
    print("  Loading Formation tau from checkpoint...")
    r_f = joblib.load(TAU_FORM_CKPT)
    tau_mean_form, tau_min_form, tau_topk_form = r_f['mt_all'], r_f['mn_all'], r_f['mt_topk']
    print(f"  [Formation]  All-feature tau: Mean={tau_mean_form:.4f}  Min={tau_min_form:.4f}")
    if tau_topk_form is not None:
        print(f"               Top-{K_REPORT} tau:       Mean={tau_topk_form:.4f}  Min={r_f['mn_topk']:.4f}")
else:
    print("  Formation energy stability:")
    mt_all_f, mn_all_f, mt_topk_f, mn_topk_f = shap_stability(
        X_train_scaled, y_train_form, best_params_form,
        X_test_scaled, FEATURE_NAMES, STABILITY_SEEDS,
        base_top_idx=top_form)
    tau_mean_form, tau_min_form, tau_topk_form = mt_all_f, mn_all_f, mt_topk_f
    r_f = {'mt_all': mt_all_f, 'mn_all': mn_all_f, 'mt_topk': mt_topk_f, 'mn_topk': mn_topk_f}
    joblib.dump(r_f, TAU_FORM_CKPT, compress=3)
    print(f"  Checkpoint saved: {TAU_FORM_CKPT}")

if os.path.exists(TAU_BAND_CKPT):
    print("  Loading Band Gap tau from checkpoint...")
    r_b = joblib.load(TAU_BAND_CKPT)
    tau_mean_band, tau_min_band, tau_topk_band = r_b['mt_all'], r_b['mn_all'], r_b['mt_topk']
    print(f"  [Band Gap]   All-feature tau: Mean={tau_mean_band:.4f}  Min={tau_min_band:.4f}")
    if tau_topk_band is not None:
        print(f"               Top-{K_REPORT} tau:       Mean={tau_topk_band:.4f}  Min={r_b['mn_topk']:.4f}")
else:
    print("  Band gap stability:")
    mt_all_b, mn_all_b, mt_topk_b, mn_topk_b = shap_stability(
        X_train_scaled, y_train_band, best_params_band,
        X_test_scaled, FEATURE_NAMES, STABILITY_SEEDS,
        base_top_idx=top_band)
    tau_mean_band, tau_min_band, tau_topk_band = mt_all_b, mn_all_b, mt_topk_b
    r_b = {'mt_all': mt_all_b, 'mn_all': mn_all_b, 'mt_topk': mt_topk_b, 'mn_topk': mn_topk_b}
    joblib.dump(r_b, TAU_BAND_CKPT, compress=3)
    print(f"  Checkpoint saved: {TAU_BAND_CKPT}")

tau_results = {
    'Formation': r_f,
    'Band Gap':  r_b,
}
save_stage('shap_complete')

# =============================================================================
# SECTION 9: TASK A -- INDIUM PROXY CHECK
# =============================================================================
print("\n--- Section 9: Indium Proxy Check (Task A) ---")

feat_band_top = X_test_scaled[:, top_band[0]]
feat_form_top = X_test_scaled[:, top_form[0]]
rho_b, p_b    = spearmanr(feat_band_top, in_frac_test)
rho_f, p_f    = spearmanr(feat_form_top, in_frac_test)

print(f"  Band Gap   top ({topnames_band[0][:22]}) vs In%: rho={rho_b:.4f}  p={p_b:.2e}")
print(f"  Formation  top ({topnames_form[0][:22]}) vs In%: rho={rho_f:.4f}  p={p_f:.2e}")
if abs(rho_b) >= 0.90:
    print("  WARNING: rho >= 0.90 -- top band-gap feature is a strong In proxy.")
else:
    print("  rho < 0.90 -- SOAP captures geometry beyond In composition.")

fig, axes = plt.subplots(1, 2, figsize=(NATURE_DOUBLE, NATURE_HEIGHT))
for i, (ax, feat, rho, p) in enumerate(zip(
        axes, [feat_form_top, feat_band_top],
        [rho_f, rho_b], [p_f, p_b])):
    ax.scatter(in_frac_test, feat, s=5, alpha=0.35, c=PALETTE['blue'],
               linewidths=0, rasterized=True)
    ax.set_xlabel('Indium atom fraction', fontsize=NATURE_FS_MD)
    ax.set_ylabel('Top SHAP feature value (scaled)', fontsize=NATURE_FS_MD)
    panel_label = '(a)' if i == 0 else '(b)'
    ax.text(0.0, 1.04, panel_label, transform=ax.transAxes,
            fontsize=NATURE_FS_LG, fontweight='bold', va='bottom', ha='left')
    ax.text(0.03, 0.95, f'rho = {rho:.3f}\np = {p:.1e}',
            transform=ax.transAxes, va='top', fontsize=NATURE_FS_SM,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8,
                      ec=PALETTE['grey'], lw=0.5))
    ax.grid(lw=0.4, alpha=0.4)

plt.tight_layout()
save_fig(fig, 'indium_proxy_v5.pdf')

# =============================================================================
# SECTION 10: CROSS-LATTICE GENERALISATION EXPERIMENT  (fully checkpointed)
# =============================================================================
print("\n--- Section 10: Cross-Lattice Generalisation (Novel) ---")

CL_CACHE = os.path.join(CHECKPOINT_DIR, 'cross_lattice_results_v5.json')

if os.path.exists(CL_CACHE):
    with open(CL_CACHE) as f:
        cl_cache = json.load(f)
    cross_lat_form = cl_cache['formation']
    cross_lat_band = cl_cache['bandgap']
    print("  Loaded cross-lattice results from checkpoint.")
else:
    all_X    = np.vstack([X_train_scaled, X_test_scaled])
    all_form = np.concatenate([y_train_form, y_test_form])
    all_band = np.concatenate([y_train_band, y_test_band])
    all_sg   = np.concatenate([sg_train, sg_test])

    cross_lat_form, cross_lat_band = [], []

    print(f"\n  {'Lattice':<12} {'N':>6}  "
          f"{'MAE_form':>9} {'Cov_form':>9} {'W_form':>8}  "
          f"{'MAE_band':>9} {'Cov_band':>9} {'W_band':>8}")
    print("  " + "-"*85)

    for held_sg, held_name in LATTICE_MAP.items():
        held_mask  = all_sg == held_sg
        train_mask = ~held_mask
        if not held_mask.any(): continue

        sg_cl_tr   = all_sg[train_mask]
        n_folds_cl = min(len(np.unique(sg_cl_tr)), 5)

        cl_ckpt = os.path.join(CHECKPOINT_DIR, f'cl_{held_sg}_v5.pkl')
        if os.path.exists(cl_ckpt):
            row_f, row_b = joblib.load(cl_ckpt)
            print(f"  {held_name:<12} (loaded from checkpoint)")
        else:
            mapie_cl_form = MapieRegressor(
                estimator=build_lgbm(best_params_form),
                cv=GroupKFold(n_splits=n_folds_cl),
                method='plus', n_jobs=1,
            )
            mapie_cl_band = MapieRegressor(
                estimator=build_lgbm(best_params_band),
                cv=GroupKFold(n_splits=n_folds_cl),
                method='plus', n_jobs=1,
            )
            mapie_cl_form.fit(all_X[train_mask], all_form[train_mask], groups=sg_cl_tr)
            mapie_cl_band.fit(all_X[train_mask], all_band[train_mask], groups=sg_cl_tr)

            pred_f, pis_f = mapie_cl_form.predict(all_X[held_mask], alpha=ALPHA)
            pred_b, pis_b = mapie_cl_band.predict(all_X[held_mask], alpha=ALPHA)

            y_te_f = all_form[held_mask]
            y_te_b = all_band[held_mask]

            row_f = {
                'Lattice': held_name, 'SG': held_sg, 'N': int(held_mask.sum()),
                'MAE':       round(mean_absolute_error(y_te_f, pred_f), 4),
                'Coverage':  round(regression_coverage_score(
                                 y_te_f, pis_f[:,0,0], pis_f[:,1,0]), 4),
                'MeanWidth': round(float(np.mean(pis_f[:,1,0]-pis_f[:,0,0])), 4),
            }
            row_b = {
                'Lattice': held_name, 'SG': held_sg, 'N': int(held_mask.sum()),
                'MAE':       round(mean_absolute_error(y_te_b, pred_b), 4),
                'Coverage':  round(regression_coverage_score(
                                 y_te_b, pis_b[:,0,0], pis_b[:,1,0]), 4),
                'MeanWidth': round(float(np.mean(pis_b[:,1,0]-pis_b[:,0,0])), 4),
            }
            joblib.dump((row_f, row_b), cl_ckpt, compress=3)
            del mapie_cl_form, mapie_cl_band; gc.collect()

        cross_lat_form.append(row_f)
        cross_lat_band.append(row_b)
        print(f"  {held_name:<12} {row_f['N']:>6}  "
              f"{row_f['MAE']:>9.4f} {row_f['Coverage']:>9.3f} {row_f['MeanWidth']:>8.4f}  "
              f"{row_b['MAE']:>9.4f} {row_b['Coverage']:>9.3f} {row_b['MeanWidth']:>8.4f}")

    with open(CL_CACHE, 'w') as f:
        json.dump({'formation': cross_lat_form, 'bandgap': cross_lat_band}, f, indent=2)
    save_stage('cross_lattice_complete')

df_cl_form = pd.DataFrame(cross_lat_form)
df_cl_band = pd.DataFrame(cross_lat_band)
df_cl_form.to_csv(os.path.join(RESULTS_DIR,'cross_lattice_formation_v5.csv'), index=False)
df_cl_band.to_csv(os.path.join(RESULTS_DIR,'cross_lattice_bandgap_v5.csv'), index=False)

# Cross-lattice figure (Nature double-column, 4-panel)
fig, axes = plt.subplots(2, 2, figsize=(NATURE_DOUBLE, NATURE_HEIGHT * 2.1))

for col, (df_cl, tit) in enumerate([(df_cl_form,'Formation energy'),
                                     (df_cl_band,'Band gap')]):
    x   = np.arange(len(df_cl))
    lbl = df_cl['Lattice'].tolist()

    axes[0, col].bar(x, df_cl['MAE'], color=PALETTE['blue'],
                     alpha=0.75, width=0.6, zorder=3)
    axes[0, col].set_xticks(x)
    axes[0, col].set_xticklabels(lbl, rotation=45, ha='right', va='top',
                                 fontsize=NATURE_FS_SM)
    axes[0, col].tick_params(axis='x', pad=2)
    axes[0, col].set_ylabel('MAE', fontsize=NATURE_FS_MD)
    axes[0, col].grid(axis='y', lw=0.4, alpha=0.4, zorder=0)
    lbl_top = '(a)' if col == 0 else '(b)'
    axes[0, col].text(0.0, 1.04, lbl_top, transform=axes[0, col].transAxes,
                      fontsize=NATURE_FS_LG, fontweight='bold', va='bottom', ha='left')

    axes[1, col].bar(x, df_cl['Coverage'], color=PALETTE['red'],
                     alpha=0.75, width=0.6, zorder=3)
    axes[1, col].axhline(1 - ALPHA, color='black', ls='--',
                          lw=NATURE_LW, label=f'Nominal {(1-ALPHA)*100:.0f}%', zorder=4)
    axes[1, col].set_xticks(x)
    axes[1, col].set_xticklabels(lbl, rotation=45, ha='right', va='top',
                                 fontsize=NATURE_FS_SM)
    axes[1, col].tick_params(axis='x', pad=2)
    axes[1, col].set_ylim(0, 1.15)
    axes[1, col].set_ylabel('CP coverage', fontsize=NATURE_FS_MD)
    axes[1, col].legend(fontsize=NATURE_FS_SM, loc='lower right',
                        bbox_to_anchor=(1.0, 1.01), frameon=False)
    axes[1, col].grid(axis='y', lw=0.4, alpha=0.4, zorder=0)
    lbl_bot = '(c)' if col == 0 else '(d)'
    axes[1, col].text(0.0, 1.04, lbl_bot, transform=axes[1, col].transAxes,
                      fontsize=NATURE_FS_LG, fontweight='bold', va='bottom', ha='left')
    for xi, (_, row_cl) in zip(x, df_cl.iterrows()):
        axes[1, col].text(xi, row_cl['Coverage'] + 0.02, f'n={row_cl["N"]}',
                          ha='center', va='bottom', fontsize=5, zorder=10,
                          bbox=dict(facecolor='white', alpha=0.85, edgecolor='none', pad=1))

plt.suptitle('Cross-lattice generalisation experiment\n'
             '(one lattice held out per run, trained on remaining 5)',
             fontsize=NATURE_FS_LG, y=1.05)
plt.tight_layout()
save_fig(fig, 'cross_lattice_v5.pdf')

# Figure: Mean Width vs MAE scatter (Cross-Lattice)
print("\n  Generating Cross-Lattice Scatter Plot...")

r_form_cl = pearsonr(df_cl_form['MeanWidth'], df_cl_form['MAE'])[0]
r_band_cl = pearsonr(df_cl_band['MeanWidth'], df_cl_band['MAE'])[0]
print(f"  Pearson(mean_width, MAE) across lattices:")
print(f"    Formation: r = {r_form_cl:.4f}")
print(f"    Band Gap:  r = {r_band_cl:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(NATURE_DOUBLE, NATURE_HEIGHT))
for i, (ax, df, r_val, title) in enumerate(zip(
    axes, [df_cl_form, df_cl_band], [r_form_cl, r_band_cl],
    ['Formation energy', 'Band gap']
)):
    x_vals = df['MeanWidth'].values
    y_vals = df['MAE'].values
    ax.scatter(x_vals, y_vals, s=30, color=PALETTE['blue'], alpha=0.8, zorder=3)
    m, b = np.polyfit(x_vals, y_vals, 1)
    x_fit = np.linspace(x_vals.min() - 0.01, x_vals.max() + 0.01, 100)
    ax.plot(x_fit, m * x_fit + b, color=PALETTE['red'], ls='--', lw=NATURE_LW,
            label=f'r = {r_val:.2f}', zorder=2)
    ax.set_xlabel('Mean interval width', fontsize=NATURE_FS_MD)
    ax.set_ylabel('MAE (eV/atom)' if i == 0 else 'MAE (eV)', fontsize=NATURE_FS_MD)
    for xi, yi, label in zip(x_vals, y_vals, df['Lattice']):
        ax.annotate(label, (xi, yi), textcoords="offset points", xytext=(5,0),
                    ha='left', fontsize=NATURE_FS_SM)
    panel_label = '(a)' if i == 0 else '(b)'
    ax.text(0.0, 1.04, panel_label, transform=ax.transAxes,
            fontsize=NATURE_FS_LG, fontweight='bold', va='bottom', ha='left')
    ax.legend(fontsize=NATURE_FS_SM, loc='best', frameon=False)
    ax.grid(lw=0.4, alpha=0.4, zorder=0)

plt.tight_layout(rect=[0, 0, 1, 0.95])
save_fig(fig, 'cross_lattice_scatter_v5.pdf')

# =============================================================================
# SECTION 11: SUMMARY TABLE + ALL OUTPUTS
# =============================================================================
print("\n--- Section 11: Summary Table ---")

summary = pd.DataFrame({
    'Metric': [
        'MAE Formation (eV/atom)', 'MAE Band Gap (eV)',
        'RMSLE Formation', 'RMSLE Band Gap',
        'R2 Formation', 'R2 Band Gap',
        'CP Coverage Formation (90% nominal)', 'CP Coverage Band Gap',
        'Mean Interval Width Formation', 'Mean Interval Width Band Gap',
        'Pearson(width,|err|) Formation', 'Pearson(width,|err|) Band Gap',
        'SHAP Kendall tau Formation (mean)', 'SHAP Kendall tau Band Gap (mean)',
        'Indium proxy rho (band gap top feature)',
    ],
    'Value': [
        round(res_form['mae'],5), round(res_band['mae'],5),
        round(res_form['rmsle'],5), round(res_band['rmsle'],5),
        round(res_form['r2'],4), round(res_band['r2'],4),
        round(res_form['coverage'],4), round(res_band['coverage'],4),
        round(res_form['mean_width'],4), round(res_band['mean_width'],4),
        round(res_form['pearson_r'],4), round(res_band['pearson_r'],4),
        round(tau_mean_form,4), round(tau_mean_band,4),
        round(rho_b,4),
    ],
    'Reference / Note': [
        'cf. Sutton et al. (2019) Table 1', 'cf. Sutton et al. (2019) Table 1',
        'NOMAD 2018 competition metric', 'NOMAD 2018 competition metric',
        '', '',
        'GroupKFold CV+ guarantee', 'GroupKFold CV+ guarantee',
        '', '',
        '>0 = informative intervals', '>0 = informative intervals',
        '>0.9 = stable feature rankings', '>0.9 = stable feature rankings',
        '<0.9 = SOAP not pure In proxy',
    ]
})

summary_csv = os.path.join(RESULTS_DIR, 'summary_metrics_v5.csv')
summary.to_csv(summary_csv, index=False)
print(summary.to_string(index=False))

# Save final models to CHECKPOINT_DIR
for name, obj in [
        ('mapie_form_v5_final.pkl', mapie_form),
        ('mapie_band_v5_final.pkl', mapie_band),
        ('scaler_v5.pkl', scaler)]:
    path = os.path.join(CHECKPOINT_DIR, name)
    joblib.dump(obj, path, compress=3)
    print(f"  Saved: {name}")

save_stage('pipeline_complete')
print(f"\nAll outputs saved to: {RESULTS_DIR}/")

# =============================================================================
# SECTION 12: MANUSCRIPT EVIDENCE CHECKLIST
# =============================================================================
print("\n--- Section 12: npj Manuscript Evidence Checklist ---")

checklist = f"""
npj Computational Materials -- Submission Checklist (v5)
=========================================================
GPU used: {USE_GPU}
Figure DPI: {NATURE_DPI}
Font: Arial/Helvetica {NATURE_FS_MD}pt (labels) / {NATURE_FS_SM}pt (ticks)

Task A   [Indium proxy]:
  Band Gap   rho = {rho_b:.4f}  p = {p_b:.2e}
  Formation  rho = {rho_f:.4f}  p = {p_f:.2e}

Task A.2 [SHAP stability, 5 seeds, top-10 Kendall tau]:
  Formation: mean tau = {tau_mean_form:.4f}  min = {tau_min_form:.4f}
  Band Gap:  mean tau = {tau_mean_band:.4f}  min = {tau_min_band:.4f}

Task B   [Conditional coverage]:
  -> cond_coverage_formation_v5.csv / cond_coverage_bandgap_v5.csv
  -> conditional_coverage_combined.csv
  -> fig_conditional_coverage_subset.pdf

Task C   [Leakage defence]:
  -> leakage_defence_v5.csv

Novel contributions vs. Sutton et al. (2019):
  1. CP guarantee: formation cov={res_form['coverage']:.3f}  band gap cov={res_band['coverage']:.3f}  [nominal=0.90]
  2. Extended SOAP (mean+std+min+max vs mean-only)
  3. GroupKFold by spacegroup in HPO + MAPIE
  4. Spacegroup-stratified coverage (sg_coverage_*_v5.csv)
  5. Cross-lattice CP coverage experiment (cross_lattice_*_v5.csv)
  6. Interval-width informativeness Pearson r

Figures saved to {RESULTS_DIR}/:
  parity_formation_v5.pdf        parity_bandgap_v5.pdf
  sg_coverage_v5.pdf             cross_lattice_v5.pdf
  shap_beeswarm_form_v5.pdf      shap_beeswarm_band_v5.pdf
  alpha_grid_form_v5.pdf         alpha_grid_band_v5.pdf
  width_informativeness_*.pdf    indium_proxy_v5.pdf
  fig_conditional_coverage_subset.pdf
  cross_lattice_scatter_v5.pdf
"""
print(checklist)
with open(os.path.join(RESULTS_DIR, 'manuscript_checklist_v5.txt'), 'w') as f:
    f.write(checklist)

# =============================================================================
# SECTION 13: FULL RESULTS DUMP
# =============================================================================
SEP  = "=" * 70
SEP2 = "-" * 70

def _hr(title=""):
    print(f"\n{SEP}")
    if title:
        pad = (70 - len(title) - 2) // 2
        print(" " * pad + f" {title} ")
    print(SEP)

def _sec(title):
    print(f"\n{SEP2}")
    print(f"  {title}")
    print(SEP2)

_hr("FULL RESULTS DUMP -- ANATOMY OF UNCERTAINTY v5")
print("Copy everything between the ==== lines and paste for verification.\n")

_sec("DATASET")
print(f"  Total structures (valid):    {X_train_scaled.shape[0] + X_test_scaled.shape[0]}")
print(f"  Training set size:           {X_train_scaled.shape[0]}")
print(f"  Test set size:               {X_test_scaled.shape[0]}")
print(f"  SOAP feature dimension:      {X_train_scaled.shape[1]}")
print(f"  SOAP_DIM (per statistic):    {SOAP_DIM}")
print(f"  Aggregation:                 mean + std + min + max  (4 x {SOAP_DIM})")
print(f"\n  Spacegroup distribution (FULL dataset):")
for sg, name in LATTICE_MAP.items():
    n_all  = int(np.sum(np.concatenate([sg_train, sg_test]) == sg))
    n_tr   = int(np.sum(sg_train == sg))
    n_te   = int(np.sum(sg_test  == sg))
    print(f"    {name:<10}  SG {sg:>3}:  total={n_all:>4}  train={n_tr:>4}  test={n_te:>3}")

_sec("HYPERPARAMETER OPTIMISATION  (Optuna TPE + GroupKFold)")
print(f"  GPU used: {USE_GPU}")
print(f"\n  Formation energy best params:")
for k, v in best_params_form.items():
    print(f"    {k:<22} = {v}")
print(f"\n  Band gap best params:")
for k, v in best_params_band.items():
    print(f"    {k:<22} = {v}")

_sec("MAIN PREDICTION METRICS  (test set, stratified split)")
print(f"  {'Metric':<35} {'Formation':>12} {'Band Gap':>12}")
print(f"  {'-'*60}")
for label, key in [
    ('MAE  (eV/atom or eV)',    'mae'),
    ('RMSE (eV/atom or eV)',    'rmse'),
    ('RMSLE (competition metric)', 'rmsle'),
    ('R2',                      'r2'),
]:
    print(f"  {label:<35} {res_form[key]:>12.5f} {res_band[key]:>12.5f}")

_sec(f"CONFORMAL PREDICTION  (CV+, GroupKFold, nominal coverage = {1-ALPHA:.0%})")
print(f"  {'Metric':<35} {'Formation':>12} {'Band Gap':>12}")
print(f"  {'-'*60}")
for label, key in [
    ('Empirical coverage',        'coverage'),
    ('Mean interval width',       'mean_width'),
    ('Pearson(width, |error|)',   'pearson_r'),
    ('Pearson p-value',           'pearson_p'),
]:
    print(f"  {label:<35} {res_form[key]:>12.5f} {res_band[key]:>12.5f}")
print(f"\n  Coverage delta (empirical - nominal):")
print(f"    Formation:  {res_form['coverage'] - (1-ALPHA):+.5f}")
print(f"    Band Gap:   {res_band['coverage'] - (1-ALPHA):+.5f}")

_sec("ALPHA-GRID  (nominal vs empirical coverage)")
print(f"  {'alpha':>6} {'nominal':>9} {'cov_form':>10} {'w_form':>9} "
      f"{'cov_band':>10} {'w_band':>9}")
print(f"  {'-'*60}")
for a in ALPHAS:
    _, pf = mapie_form.predict(X_test_scaled, alpha=a)
    _, pb = mapie_band.predict(X_test_scaled, alpha=a)
    cf = regression_coverage_score(y_test_form, pf[:,0,0], pf[:,1,0])
    wf = float(np.mean(pf[:,1,0] - pf[:,0,0]))
    cb = regression_coverage_score(y_test_band, pb[:,0,0], pb[:,1,0])
    wb = float(np.mean(pb[:,1,0] - pb[:,0,0]))
    print(f"  {a:>6.2f} {1-a:>9.2f} {cf:>10.4f} {wf:>9.4f} "
          f"{cb:>10.4f} {wb:>9.4f}")

_sec("CONDITIONAL COVERAGE  (Task B, physics-based subsets)")
print(f"\n  FORMATION ENERGY subsets:")
print(f"  {'Subset':<48} {'N':>5} {'Coverage':>9} {'MAE':>9} {'Width':>8}")
print(f"  {'-'*82}")
for lbl, msk in form_subsets:
    if not msk.any(): continue
    cov = regression_coverage_score(y_test_form[msk], res_form['lo'][msk], res_form['hi'][msk])
    mae = mean_absolute_error(y_test_form[msk], res_form['y_pred'][msk])
    w   = float(np.mean(res_form['hi'][msk] - res_form['lo'][msk]))
    print(f"  {lbl:<48} {msk.sum():>5} {cov:>9.4f} {mae:>9.4f} {w:>8.4f}")

print(f"\n  BAND GAP subsets:")
print(f"  {'Subset':<48} {'N':>5} {'Coverage':>9} {'MAE':>9} {'Width':>8}")
print(f"  {'-'*82}")
for lbl, msk in band_subsets:
    if not msk.any(): continue
    cov = regression_coverage_score(y_test_band[msk], res_band['lo'][msk], res_band['hi'][msk])
    mae = mean_absolute_error(y_test_band[msk], res_band['y_pred'][msk])
    w   = float(np.mean(res_band['hi'][msk] - res_band['lo'][msk]))
    print(f"  {lbl:<48} {msk.sum():>5} {cov:>9.4f} {mae:>9.4f} {w:>8.4f}")

_sec("SPACEGROUP-STRATIFIED COVERAGE  (Section 6.3 -- Novel)")
print(f"  {'Lattice':<10} {'SG':>4} {'N':>5}  "
      f"{'Cov_F':>7} {'W_F':>8} {'MAE_F':>8}  "
      f"{'Cov_B':>7} {'W_B':>8} {'MAE_B':>8}")
print(f"  {'-'*75}")
for row_f, row_b in zip(sg_rows_form, sg_rows_band):
    print(f"  {row_f['Lattice']:<10} {row_f['SG']:>4} {row_f['N']:>5}  "
          f"{row_f['Coverage']:>7.4f} {row_f['MeanWidth']:>8.4f} {row_f['MAE']:>8.4f}  "
          f"{row_b['Coverage']:>7.4f} {row_b['MeanWidth']:>8.4f} {row_b['MAE']:>8.4f}")

_sec("LEAKAGE DEFENCE  (Task C)")
print(f"  Most-frequent composition group: {focal['group']}")
print(f"    N={focal['n']}  MAE_form={focal['mae_form']:.5f}  "
      f"MAE_band={focal['mae_band']:.5f}")
print(f"  Least-frequent composition group: {rare['group']}")
print(f"    N={rare['n']}  MAE_form={rare['mae_form']:.5f}  "
      f"MAE_band={rare['mae_band']:.5f}")
print(f"  MAE ratio (frequent/rare, formation): {ratio:.3f}")
print(f"  Interpretation: ratio < 2.0 = no severe leakage artefact")
print(f"  Total unique composition groups in test set: {len(group_stats)}")

_sec("SHAP FEATURE IMPORTANCE  (top-10 features)")
print(f"\n  FORMATION ENERGY:")
mean_abs_form = np.abs(sv_form).mean(axis=0)
top10_form = np.argsort(mean_abs_form)[::-1][:10]
for rank, idx in enumerate(top10_form, 1):
    print(f"    {rank:>2}. {FEATURE_NAMES[idx]:<14}  mean|SHAP| = {mean_abs_form[idx]:.6f}")

print(f"\n  BAND GAP:")
mean_abs_band = np.abs(sv_band).mean(axis=0)
top10_band = np.argsort(mean_abs_band)[::-1][:10]
for rank, idx in enumerate(top10_band, 1):
    print(f"    {rank:>2}. {FEATURE_NAMES[idx]:<14}  mean|SHAP| = {mean_abs_band[idx]:.6f}")

_sec("SHAP STABILITY  (Task A.2 -- Kendall tau, 5 seeds, top-10 features)")
print(f"  Formation energy:  mean tau = {tau_mean_form:.5f}  "
      f"min tau = {tau_min_form:.5f}")
print(f"  Band gap:          mean tau = {tau_mean_band:.5f}  "
      f"min tau = {tau_min_band:.5f}")
print(f"  Threshold for 'stable': tau > 0.9")
print(f"  Formation stable: {'YES' if tau_mean_form > 0.9 else 'NO -- check'}")
print(f"  Band gap stable:  {'YES' if tau_mean_band > 0.9 else 'NO -- check'}")

_sec("INDIUM PROXY CHECK  (Task A -- Spearman rho)")
print(f"  Formation top feature:  {topnames_form[0]}")
print(f"    Spearman rho vs In%:  {rho_f:.5f}  p = {p_f:.3e}")
print(f"  Band gap top feature:   {topnames_band[0]}")
print(f"    Spearman rho vs In%:  {rho_b:.5f}  p = {p_b:.3e}")
print(f"  Threshold: rho < 0.90 = SOAP not a simple In proxy")
print(f"  Formation: {'PASS (rho < 0.90)' if abs(rho_f) < 0.90 else 'WARN (rho >= 0.90)'}")
print(f"  Band gap:  {'PASS (rho < 0.90)' if abs(rho_b) < 0.90 else 'WARN (rho >= 0.90)'}")

_sec("CROSS-LATTICE GENERALISATION  (Section 10 -- Novel)")
print(f"  cf. Sutton et al. (2019) Table 3")
print(f"\n  {'Lattice':<10} {'N':>5}  "
      f"{'MAE_F':>8} {'Cov_F':>7} {'W_F':>8}  "
      f"{'MAE_B':>8} {'Cov_B':>7} {'W_B':>8}")
print(f"  {'-'*75}")
for rf, rb in zip(cross_lat_form, cross_lat_band):
    print(f"  {rf['Lattice']:<10} {rf['N']:>5}  "
          f"{rf['MAE']:>8.4f} {rf['Coverage']:>7.3f} {rf['MeanWidth']:>8.4f}  "
          f"{rb['MAE']:>8.4f} {rb['Coverage']:>7.3f} {rb['MeanWidth']:>8.4f}")

r_cl_f = pearsonr(df_cl_form['MeanWidth'], df_cl_form['MAE'])[0]
r_cl_b = pearsonr(df_cl_band['MeanWidth'], df_cl_band['MAE'])[0]
print(f"\n  Pearson(mean_width, MAE) across lattices:")
print(f"    Formation: r = {r_cl_f:.5f}")
print(f"    Band Gap:  r = {r_cl_b:.5f}")
print(f"  Positive r = wider intervals correctly signal harder generalisations.")

_sec("OUTPUT FILES")
for fn in sorted(os.listdir(RESULTS_DIR)):
    fpath = os.path.join(RESULTS_DIR, fn)
    if os.path.isfile(fpath):
        size = os.path.getsize(fpath)
        print(f"  {fn:<50}  {size/1024:>8.1f} KB")

_hr("END OF RESULTS DUMP")
print("=== ANATOMY OF UNCERTAINTY v5 -- PIPELINE COMPLETE ===")
print(f"Outputs: {RESULTS_DIR}/  |  Checkpoints: {CHECKPOINT_DIR}/")
