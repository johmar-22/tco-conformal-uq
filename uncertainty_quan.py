# -*- coding: utf-8 -*-
"""
uncertainty_quan.py
===================

    "Conformal Uncertainty Quantification for Transparent Conductive Oxide
     Property Prediction: Coverage Diagnostics Across Crystal Symmetry
     Classes"

Usage
-----
    pip install -r requirements.txt
    python uncertainty_quan.py

Two reproduction modes:

1. CHECKPOINT MODE:
   On first run the script downloads the pre-computed checkpoints published
   as GitHub Release assets (v1.0.0): the SOAP feature cache, the prepared
   train/test arrays, the fitted scaler, and the trained MAPIE CV+ models.
   All tables and figures of the paper are then regenerated exactly.
   Set the environment variable TCO_UQ_NO_DOWNLOAD=1 to disable downloading.

2. FULL RE-COMPUTATION MODE (~4 h CPU / ~45 min GPU):
   Delete results/checkpoints/ and results/*.npz / *.pkl and provide the
   NOMAD 2018 geometry files (not redistributable; download from
   https://www.kaggle.com/c/nomad2018-predict-transparent-conductors/data)
   under data/train/<id>/geometry.xyz. train.csv is fetched automatically.

All outputs are written to results/. Randomness is controlled by fixed
seeds (SEED = 42; SHAP stability seeds 42/123/456/789/2024).
Pinned dependencies (requirements.txt): mapie==0.8.6, dscribe==2.1.2.
"""


# SECTION 0: SETUP, IMPORTS, GPU DETECTION

# Dependencies are installed via:  pip install -r requirements.txt
# (pinned: mapie==0.8.6, dscribe==2.1.2 -- API/feature-layout critical)

import os, json, gc, warnings, shutil, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
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

from ase.io import read as ase_read
from dscribe.descriptors import SOAP
from tqdm.auto import tqdm
import shap

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------- Local, repository-relative paths (no Google Drive needed) ----------
try:
    REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
except NameError:            # interactive session / notebook
    REPO_ROOT = os.getcwd()
os.chdir(REPO_ROOT)

RESULTS_DIR = os.environ.get('TCO_UQ_RESULTS', os.path.join(REPO_ROOT, 'results'))
os.makedirs(RESULTS_DIR, exist_ok=True)

SEED  = 42
ALPHA = 0.10   # target coverage = 1 - ALPHA = 90%
np.random.seed(SEED)

from scipy.stats import beta as _beta_dist
def clopper_pearson(k, n_trials, ci=0.95):
    """Exact binomial (Clopper-Pearson) CI for k successes in n_trials."""
    a = 1.0 - ci
    lo = _beta_dist.ppf(a / 2, k, n_trials - k + 1) if k > 0 else 0.0
    hi = _beta_dist.ppf(1 - a / 2, k + 1, n_trials - k) if k < n_trials else 1.0
    return float(lo), float(hi)

LATTICE_MAP = {
    12: 'C2/m', 33: 'Pna21', 167: 'R-3c',
    194: 'P63/mmc', 206: 'Ia-3', 227: 'Fd-3m',
}
# Older checkpoints/CSVs may carry the pre-v7 labels; normalise on load.
LABEL_FIX = {'C/2m': 'C2/m', 'R3c': 'R-3c'}

OUTPUT_DIR     = 'figures_working'                 # scratch dir, synced to results/
CHECKPOINT_DIR = os.path.join(RESULTS_DIR, 'checkpoints')
SOAP_CACHE     = os.path.join(CHECKPOINT_DIR, 'soap_features_corrected_v2.npz')
DATA_ROOT      = os.path.join(REPO_ROOT, 'data', 'train')
TRAIN_URL      = ('https://raw.githubusercontent.com/csutton7/'
                  'nomad_2018_kaggle_dataset/master/train.csv')

for d in [OUTPUT_DIR, CHECKPOINT_DIR]:
    os.makedirs(d, exist_ok=True)

# ---- Pre-computed checkpoints (GitHub Release assets, no Drive needed) ----
RELEASE_BASE = ('https://github.com/johmar-22/tco-conformal-uq/'
                'releases/download/v1.0.0/')
RELEASE_ASSETS = {                       # file name -> destination directory
    'soap_features_corrected_v2.npz': CHECKPOINT_DIR,   # SOAP cache (~340 MB)
    'data_prep_v5.npz':               CHECKPOINT_DIR,   # scaled arrays (~200 MB)
    'scaler_v5.pkl':                  CHECKPOINT_DIR,   # fitted StandardScaler
    'mapie_form_v5.pkl':              CHECKPOINT_DIR,   # trained CV+ model
    'mapie_band_v5.pkl':              CHECKPOINT_DIR,   # trained CV+ model
}

def fetch_release_assets():
    """Download missing checkpoint files from the GitHub Release.

    Skipped when TCO_UQ_NO_DOWNLOAD=1 is set, or when the files already
    exist. On download failure the pipeline falls back to full
    re-computation (which requires the Kaggle geometry files for SOAP).
    """
    if os.environ.get('TCO_UQ_NO_DOWNLOAD', '0') == '1':
        print('  [assets] TCO_UQ_NO_DOWNLOAD=1 -- skipping release downloads.')
        return
    import urllib.request
    for fname, dest_dir in RELEASE_ASSETS.items():
        dest = os.path.join(dest_dir, fname)
        if os.path.exists(dest):
            continue
        url = RELEASE_BASE + fname
        print(f'  [assets] Downloading {fname} ...')
        try:
            tmp = dest + '.part'
            urllib.request.urlretrieve(url, tmp)
            os.replace(tmp, dest)
            print(f'  [assets] OK: {dest} '
                  f'({os.path.getsize(dest)/1e6:.1f} MB)')
        except Exception as e:
            print(f'  [assets] WARNING: could not download {fname}: {e}')
            print('  [assets] The pipeline will recompute this artefact '
                  'from scratch if possible.')

fetch_release_assets()

# ---- Stage tracker (crash-safe resume) ----
STAGE_FILE = os.path.join(CHECKPOINT_DIR, 'pipeline_stage_v5.json')

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
        print(result.stdout.split('\n')[8])   # print GPU model line
    else:
        print("No GPU detected. Running on CPU.")
except Exception:
    print("nvidia-smi not available. Running on CPU.")

# n_jobs: use -1 on CPU (all cores), 1 on GPU (GPU manages parallelism)
LGBM_JOBS = 1 if USE_GPU else -1

# ---- Nature journal figure style ----
# Based on Nature guide: https://www.nature.com/nature/for-authors/formatting-guide
# Single column = 3.5 in (89 mm), double column = 7.2 in (183 mm)
# Font: Arial/Helvetica, 5-7pt for labels/ticks, 7-8pt for titles
# DPI: 600 for line art, 300 for halftone (we use 600 throughout)

NATURE_DPI    = 600
NATURE_FONT   = 'Arial'          # fallback: DejaVu Sans if Arial absent
NATURE_FS_SM  = 7                # small text (tick labels, legend)
NATURE_FS_MD  = 8                # medium (axis labels)
NATURE_FS_LG  = 9                # large (panel titles)
NATURE_LW     = 0.75             # default line width
NATURE_SINGLE = 3.5              # single-column width (inches)
NATURE_DOUBLE = 7.2              # double-column width (inches)
NATURE_HEIGHT = 2.8              # default panel height

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

def save_fig(fig, fname, subdir=OUTPUT_DIR):
    """Save figure to the scratch dir and mirror it into results/."""
    local_path = os.path.join(subdir, fname)
    fig.savefig(local_path, dpi=NATURE_DPI, bbox_inches='tight',
                facecolor='white')
    plt.close(fig)
    # Mirror into results/ immediately
    drive_path = os.path.join(RESULTS_DIR, fname)
    shutil.copy2(local_path, drive_path)
    print(f"  Figure saved: {fname}")

print(f"\nSetup complete  [v5 | GPU={USE_GPU} | DPI={NATURE_DPI}]")


# SECTION 1: DATA LOADING

print("\n--- Section 1: Loading Data ---")

DATA_CSV_CACHE = os.path.join(CHECKPOINT_DIR, 'full_df_v5.pkl')

if os.path.exists(DATA_CSV_CACHE):
    full_df = pd.read_pickle(DATA_CSV_CACHE)
    print(f"Loaded CSV from local cache: {len(full_df)} structures.")
else:
    full_df = pd.read_csv(TRAIN_URL)
    full_df.to_pickle(DATA_CSV_CACHE)
    print(f"Loaded {len(full_df)} structures; CSV cache saved.")

COMP_COLS = ['percent_atom_al', 'percent_atom_ga', 'percent_atom_in',
             'spacegroup', 'number_of_total_atoms']
missing = [c for c in COMP_COLS if c not in full_df.columns]
if missing:
    print(f"WARNING: Missing columns: {missing}")


# SECTION 2: SOAP DESCRIPTORS

print("\n--- Section 2: SOAP Features ---")

if os.path.exists(SOAP_CACHE):
    X_soap = np.load(SOAP_CACHE)['features']
    print(f"Loaded cached SOAP features: {X_soap.shape}")
else:
    if not os.path.isdir(DATA_ROOT):
        raise SystemExit(
            f"\nSOAP cache not found ({SOAP_CACHE}) and geometry files are "
            f"missing ({DATA_ROOT}).\n"
            "Either (a) allow the release-asset download (unset "
            "TCO_UQ_NO_DOWNLOAD), or\n(b) download the NOMAD 2018 data from "
            "https://www.kaggle.com/c/nomad2018-predict-transparent-conductors/data\n"
            "and unpack it so that data/train/<id>/geometry.xyz exists."
        )
    print("Computing SOAP from scratch (~20 min).")
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


# SECTION 3: DATA PREPARATION  (fully checkpointed)
# =============================================================================
print("\n--- Section 3: Data Preparation ---")

PREP_CACHE = os.path.join(CHECKPOINT_DIR, 'data_prep_v5.npz')

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
    _df_valid_path = os.path.join(CHECKPOINT_DIR, 'df_valid_v5.pkl')
    if os.path.exists(_df_valid_path):
        df_valid = pd.read_pickle(_df_valid_path)
    else:
        # Rebuild deterministically from train.csv + the SOAP cache
        # (df_valid_v5.pkl is not shipped as a release asset).
        _yf = full_df['formation_energy_ev_natom'].values.astype(np.float32)
        _yb = full_df['bandgap_energy_ev'].values.astype(np.float32)
        _mask = (np.isfinite(_yf) & np.isfinite(_yb) &
                 ~np.isnan(X_soap).any(axis=1))
        df_valid = full_df[_mask].reset_index(drop=True)
        df_valid.to_pickle(_df_valid_path)
        print(f"  Rebuilt df_valid ({len(df_valid)} rows) from train.csv + SOAP cache.")
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

    # Save everything to the local checkpoint
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


# SECTION 4: HPO  (Optuna TPE + GroupKFold + GPU)

print("\n--- Section 4: Hyperparameter Optimisation (GroupKFold + GPU) ---")

# HPO params are cached locally. If they already exist, skip the trials.
HPO_CACHE = os.path.join(CHECKPOINT_DIR, 'best_params_v5.json')

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
        SQLite storage = fully resumable if the run is interrupted mid-HPO.
        """
        storage = f"sqlite:///{CHECKPOINT_DIR}/{study_name}.db"
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
            base_params.update(GPU_PARAMS)   # add GPU keys if detected

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

    # Save HPO results immediately
    with open(HPO_CACHE, 'w') as f:
        json.dump({'formation': best_params_form, 'bandgap': best_params_band}, f, indent=2)
    save_stage('hpo_complete')
    print(f"  HPO params saved.")

print(f"  Formation: {best_params_form}")
print(f"  Band Gap:  {best_params_band}")
print(f"  GPU used for HPO: {USE_GPU}")


# SECTION 5: CONFORMAL PREDICTION TRAINING  (CV+, GroupKFold + GPU)

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
    """Train CV+ with GroupKFold(n=6). Checkpoint auto-saved."""
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
    # Mirror to results/ root for extra safety
    shutil.copy2(ckpt_path, os.path.join(RESULTS_DIR, ckpt_name))
    print(f"  Checkpoint saved: {ckpt_name}")
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


# SECTION 6: EVALUATION  

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

    # Points coloured by absolute error
    abs_err = np.abs(y_true - y_pred)
    sc = ax.scatter(y_true, y_pred, c=abs_err, cmap='plasma',
                    s=6, alpha=0.6, linewidths=0, rasterized=True)
    cb = fig.colorbar(sc, ax=ax, pad=0.02, shrink=0.85)
    cb.set_label('|Error|', fontsize=NATURE_FS_SM)
    cb.ax.tick_params(labelsize=NATURE_FS_SM)

    # 90% CP interval band (sort by x to avoid fill_between artefacts)
    order = np.argsort(y_true)
    ax.fill_between(y_true[order], lo[order], hi[order],
                    alpha=0.15, color=PALETTE['blue'],
                    label='90% CP interval', zorder=0)

    # y = x diagonal
    lim = [min(y_true.min(), y_pred.min()) - 0.05,
           max(y_true.max(), y_pred.max()) + 0.05]
    ax.plot(lim, lim, color=PALETTE['grey'], lw=NATURE_LW, ls='--', zorder=5)
    ax.set_xlim(lim); ax.set_ylim(lim)

    ax.set_xlabel(xlabel, fontsize=NATURE_FS_MD)
    ax.set_ylabel(ylabel, fontsize=NATURE_FS_MD)

    # --- TITLE REPLACED WITH PANEL LABEL ---
    ax.text(0.0, 1.04, panel_label, transform=ax.transAxes,
            fontsize=NATURE_FS_LG, fontweight='bold', va='bottom', ha='left')
    # ---------------------------------------

    ax.legend(fontsize=NATURE_FS_SM, loc='upper left')

    # Annotation: key metrics
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    cov = regression_coverage_score(y_true, lo, hi)
    ax.text(0.97, 0.06,
            f'MAE = {mae:.4f}\nR² = {r2:.4f}\nCov. = {cov:.3f}',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=NATURE_FS_SM,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8,
                      ec=PALETTE['grey'], lw=0.5))

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # --- ADDED JPG EXPORT AT 600 DPI ---
    fig.savefig(os.path.join(RESULTS_DIR, fname.replace('.pdf', '.jpg')), dpi=600, bbox_inches='tight')  # JPG companion
    save_fig(fig, fname)

# Passing (a) and (b) instead of string titles
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
    ('Stable (E_f ≤ 0.20 eV/atom)', y_test_form <= 0.20),
    ('Unstable (E_f > 0.20 eV/atom)', y_test_form > 0.20),
    ('Strongly stable (E_f < −1.0 eV/atom)', y_test_form < -1.0),
]
band_subsets = [
    ('Metal (E_g < 0.10 eV)', y_test_band < 0.10),
    ('Semiconductor (0.10 ≤ E_g < 3.0 eV)',
     (y_test_band >= 0.10) & (y_test_band < 3.0)),
    ('Insulator (E_g ≥ 3.0 eV)', y_test_band >= 3.0),
    ('TCO target window (2.0–4.0 eV)',
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
    n = int(msk.sum())
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

# ---- O1: Clopper-Pearson 95% CIs for per-spacegroup coverage (Table 4) ----
for _df in (df_sg_form, df_sg_band):
    _ks = np.rint(_df['Coverage'].values * _df['N'].values).astype(int)
    _ci = [clopper_pearson(k, nn) for k, nn in zip(_ks, _df['N'].values)]
    _df['Cov_CI95_lo'] = [round(c[0], 4) for c in _ci]
    _df['Cov_CI95_hi'] = [round(c[1], 4) for c in _ci]
print("\n  Coverage 95% Clopper-Pearson CIs (per spacegroup, Table 4):")
for _tag, _df in (('formation', df_sg_form), ('band gap', df_sg_band)):
    for _, _r in _df.iterrows():
        print(f"    [{_tag:<9}] {_r['Lattice']:<8} {_r['Coverage']:.3f} "
              f"[{_r['Cov_CI95_lo']:.3f}, {_r['Cov_CI95_hi']:.3f}]  (n={int(_r['N'])})")

df_sg_form.to_csv(os.path.join(RESULTS_DIR,'sg_coverage_formation_v5.csv'), index=False)
df_sg_band.to_csv(os.path.join(RESULTS_DIR,'sg_coverage_bandgap_v5.csv'), index=False)


# SECTION 6.3b (O2): PER-SPACEGROUP SOAP ENVIRONMENT DIVERSITY

# Converts the qualitative Fd-3m environment-diversity claim (manuscript
# Sections 3.3/3.4) into a measured result: per-spacegroup mean feature
# variance and mean pairwise SOAP descriptor distance on the test set.
print("\n--- Section 6.3b: Per-Spacegroup SOAP Environment Diversity (O2) ---")
from scipy.spatial.distance import pdist as _pdist
_div_rows = []
_rng = np.random.default_rng(SEED)
for _sg_int, _sg_name in LATTICE_MAP.items():
    _msk = sg_test == _sg_int
    if not _msk.any(): continue
    _Xg  = X_test_scaled[_msk]
    _fv  = float(np.mean(np.var(_Xg, axis=0)))
    _sel = _rng.choice(len(_Xg), size=min(len(_Xg), 60), replace=False)
    _mpd = float(np.mean(_pdist(_Xg[_sel])))
    _div_rows.append({'Lattice': _sg_name, 'SG': _sg_int, 'N': int(_msk.sum()),
                      'MeanFeatureVar': round(_fv, 4),
                      'MeanPairwiseDist': round(_mpd, 4)})
    print(f"  {_sg_name:<8} n={int(_msk.sum()):>3}  mean feature var={_fv:8.4f}  "
          f"mean pairwise dist={_mpd:9.4f}")
_df_div = pd.DataFrame(_div_rows).sort_values('MeanPairwiseDist', ascending=False)
_df_div.to_csv(os.path.join(RESULTS_DIR, 'soap_env_diversity_v7.csv'), index=False)
_top_div = _df_div.iloc[0]['Lattice']
print(f"  Most heterogeneous spacegroup by mean pairwise distance: {_top_div}")
print("  -> cite soap_env_diversity_v7.csv in Section 3.3 ONLY if Fd-3m ranks first.")

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
    ax2.set_ylabel('Mean interval width', fontsize=NATURE_FS_SM,
                   color=bold_color)
    ax2.tick_params(axis='y', labelcolor=bold_color,
                     labelsize=NATURE_FS_SM)
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
             '(GroupKFold calibration — each bar = one held-out lattice)',
             fontsize=NATURE_FS_LG, y=1.08)

plt.tight_layout()

# --- ADDED JPG EXPORT AT 600 DPI ---
fig.savefig(os.path.join(RESULTS_DIR, 'sg_coverage_v5.jpg'), dpi=600, bbox_inches='tight')  # JPG companion
save_fig(fig, 'sg_coverage_v5.pdf')

# ---- 6.4 Interval-width informativeness ----
print("\n--- Section 6.4: Interval Width Informativeness ---")

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

    # --- TITLE REPLACED WITH PANEL LABEL ---
    ax1.text(0.0, 1.04, panel_label_1, transform=ax1.transAxes,
             fontsize=NATURE_FS_LG, fontweight='bold', va='bottom', ha='left')
    # ---------------------------------------

    ax1.legend(fontsize=NATURE_FS_SM)
    ax1.grid(lw=0.4, alpha=0.4)

    # Bar: mean |error| per width quartile
    ax2.bar(range(4), q_stats.values, color=PALETTE['blue'],
            alpha=0.75, width=0.6)
    ax2.set_xticks(range(4))
    ax2.set_xticklabels(q_stats.index.tolist(), fontsize=NATURE_FS_SM,
                         rotation=20, ha='right')
    ax2.set_ylabel('Mean |error|', fontsize=NATURE_FS_MD)

    # --- TITLE REPLACED WITH PANEL LABEL ---
    ax2.text(0.0, 1.04, panel_label_2, transform=ax2.transAxes,
             fontsize=NATURE_FS_LG, fontweight='bold', va='bottom', ha='left')
    # ---------------------------------------

    ax2.grid(axis='y', lw=0.4, alpha=0.4)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # --- ADDED JPG EXPORT AT 600 DPI ---
    fig.savefig(os.path.join(RESULTS_DIR, f'width_informativeness_{fname_tag}_v5.jpg'), dpi=600, bbox_inches='tight')  # JPG companion
    save_fig(fig, f'width_informativeness_{fname_tag}_v5.pdf')

    print(f"  [{target_name}] Pearson r={r_p:.4f}  p={p_p:.2e}")

# Generating panels (a) and (b) for Formation Energy
width_informativeness_plot(
    y_test_form, res_form['y_pred'], res_form['widths'],
    'Formation energy', '(a)', '(b)', 'form')

# Generating panels (c) and (d) for Band Gap
width_informativeness_plot(
    y_test_band, res_band['y_pred'], res_band['widths'],
    'Band gap', '(c)', '(d)', 'band')

save_stage('evaluation_complete')

"""
Conditional Coverage Table — Task B (complete, paper-ready)
============================================================
Paste this block AFTER the existing Section 6.1 code in your notebook
(i.e., after df_form_cond and df_band_cond have been computed).

Produces:
  1. A combined LaTeX / CSV table ready for the manuscript.
  2. A single Nature-style figure (bar chart) showing coverage per subset
     for both targets side-by-side, with the 90% nominal line marked.

No new computations needed — uses df_form_cond and df_band_cond
that your Section 6.1 already produces.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

# ── 0.  Palette (keep consistent with your existing style) ──────────────────
PALETTE   = {'blue': '#2166AC', 'red': '#D6604D', 'grey': '#888888'}
NATURE_FS = 7
NATURE_FS_SM = 6

# ── 1.  Annotate the tables with target label ────────────────────────────────
df_form_cond['Target'] = 'Formation energy (eV/atom)'
df_band_cond['Target'] = 'Band gap (eV)'

# Re-order columns for the paper
col_order = ['Target', 'Subset', 'N', 'Coverage', 'MeanWidth', 'MAE']
df_form_display = df_form_cond[col_order].copy()
df_band_display = df_band_cond[col_order].copy()

combined = pd.concat([df_form_display, df_band_display], ignore_index=True)

# ── 2.  Flag subsets that violate nominal 90% coverage ───────────────────────
ALPHA = 0.10
combined['BelowNominal'] = combined['Coverage'] < (1 - ALPHA)

print("\n=== Combined Conditional Coverage Table ===")
print(combined.to_string(index=False))

# ── 3.  Save CSV ─────────────────────────────────────────────────────────────

out_csv = os.path.join(RESULTS_DIR, 'conditional_coverage_combined.csv')
combined.to_csv(out_csv, index=False)
print(f"\nCSV saved to: {out_csv}")

# ── 4.  LaTeX table snippet ───────────────────────────────────────────────────
# Prints a LaTeX table you can paste directly into the manuscript.
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
    # Print a midrule between the two targets
    if prev_target is not None and row['Target'] != prev_target:
        print(r"\midrule")
    prev_target = row['Target']

    target_str = row['Target'] if prev_target != row['Target'] else ''
    cov_str = f"{row['Coverage']:.3f}"
    if row['BelowNominal']:
        cov_str = r"\textbf{" + cov_str + r"}$^*$"
    print(
        f"{row['Target']} & {row['Subset']} & {row['N']} "
        f"& {cov_str} & {row['MeanWidth']:.3f} & {row['MAE']:.4f} \\\\"
    )

print(r"\bottomrule")
print(r"\end{tabular}")
print(r"\end{table}")

# ── 5.  Figure: bar chart of coverage per subset ─────────────────────────────
# Apply Nature-style font and line settings globally
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'axes.labelsize': NATURE_FS,
    'xtick.labelsize': NATURE_FS_SM,
    'ytick.labelsize': NATURE_FS_SM,
    'legend.fontsize': NATURE_FS_SM,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8
})

fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))

panel_labels = ['a', 'b']

for i, (ax, (df_sub, unit)) in enumerate(zip(
    axes,
    [
        (df_form_cond, 'eV/atom'),
        (df_band_cond, 'eV'),
    ]
)):
    # Splits the string at ' (' and keeps only the first part to shorten labels
    labels   = [str(label).split(' (')[0] for label in df_sub['Subset'].tolist()]
    coverage = df_sub['Coverage'].tolist()
    widths   = df_sub['MeanWidth'].tolist()
    ns       = df_sub['N'].tolist()

    x = np.arange(len(labels))
    bar_width = 0.55

    # Color bars: red if below 0.90, blue otherwise
    colors = [PALETTE['red'] if c < (1 - ALPHA) else PALETTE['blue']
              for c in coverage]

    bars = ax.bar(x, coverage, width=bar_width, color=colors,
                  edgecolor='white', linewidth=0.5, zorder=3)

    # Nominal line
    ax.axhline(1 - ALPHA, color='black', linestyle='--',
               linewidth=0.8, zorder=4)

    # N labels above bars
    for bar, n in zip(bars, ns):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.004,
                f'n={n}', ha='center', va='bottom',
                fontsize=NATURE_FS_SM, color='black')

    # Interval-width as overlay line (right axis)
    ax2 = ax.twinx()
    ax2.plot(x, widths, color=PALETTE['grey'], marker='o',
             markersize=3, linewidth=0.9, linestyle='-',
             zorder=5)
    ax2.set_ylabel(f'Mean width ({unit})', fontsize=NATURE_FS)
    ax2.tick_params(axis='y', labelsize=NATURE_FS_SM)
    ax2.set_ylim(0, max(widths) * 1.5)

    # Axis formatting
    ax.set_xticks(x)
    # Using horizontal (rotation=0) labels now that they are short
    ax.set_xticklabels(labels, rotation=0, ha='center',
                       fontsize=NATURE_FS_SM)
    ax.set_ylabel('Empirical coverage', fontsize=NATURE_FS)
    ax.set_ylim(0.80, 1.02)
    ax.tick_params(axis='both', labelsize=NATURE_FS_SM)
    ax.yaxis.grid(True, linestyle=':', linewidth=0.4, zorder=0)
    ax.set_axisbelow(True)

    # Add (a) and (b) labels to the upper left, outside the plot area
    ax.text(-0.15, 1.05, f"({panel_labels[i]})", transform=ax.transAxes,
            fontsize=8, fontweight='bold', va='bottom', ha='left')

# Create handles for the global legend
blue_patch  = mpatches.Patch(color=PALETTE['blue'], label='Coverage $\geq$ 0.90')
red_patch   = mpatches.Patch(color=PALETTE['red'], label='Coverage < 0.90')
nom_line    = plt.Line2D([0], [0], color='black', linestyle='--', linewidth=0.8, label='90% nominal')
width_line  = plt.Line2D([0], [0], color=PALETTE['grey'], marker='o', markersize=3, linewidth=0.9, label='Mean width (right axis)')

# Place a single, centralized legend below the entire figure
fig.legend(handles=[blue_patch, red_patch, nom_line, width_line],
           fontsize=NATURE_FS_SM, loc='lower center',
           bbox_to_anchor=(0.5, -0.08), ncol=4, frameon=False)

# Adjust layout to prevent cutting off the x-labels, leaving room for the legend
plt.tight_layout()

out_fig_pdf = os.path.join(RESULTS_DIR, 'fig_conditional_coverage_subset.pdf')
out_fig_jpg = os.path.join(RESULTS_DIR, 'fig_conditional_coverage_subset.jpg')

# bbox_inches='tight' ensures the external legend is captured in the saved files
fig.savefig(out_fig_pdf, bbox_inches='tight', dpi=600)
fig.savefig(out_fig_jpg, bbox_inches='tight', dpi=600)

print(f"\nFigure saved to: {out_fig_pdf}")
print(f"Figure saved to: {out_fig_jpg}")
plt.show()
set_nature_style()  # FIX (v7.1): restore global 600-DPI Nature rcParams after the local override above

# ── 6.  Interpretation helper ─────────────────────────────────────────────────
print("\n=== Interpretation notes for the paper ===")
for _, row in combined.iterrows():
    status = "BELOW NOMINAL" if row['BelowNominal'] else "OK"
    print(
        f"  [{status}] {row['Target']} | {row['Subset']} "
        f"(n={row['N']}): coverage={row['Coverage']:.3f}, "
        f"MAE={row['MAE']:.4f}, width={row['MeanWidth']:.3f}"
    )



import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from ase.io import read as ase_read, write as ase_write
from ase.visualize.plot import plot_atoms

# ── 0. Paths (must match your notebook) ──────────────────────────────────────
DATA_ROOT  = os.path.join(REPO_ROOT, 'data', 'train')   # folder with id/geometry.xyz
NATURE_FS  = 7
NATURE_FS_SM = 6

# ── 1. Identify the top outlier in Fd-3m (SG 227) ───────────────────────────
# We target band gap: it has the most dramatic coverage failure in SG 227.
mask_sg227 = (sg_test == 227)
widths_sg227 = res_band['widths'][mask_sg227]
errors_sg227 = np.abs(y_test_band[mask_sg227] - res_band['y_pred'][mask_sg227])

# Pick the structure with the WIDEST band gap CP interval in Fd-3m
# Instead of top_local_idx = np.argmax(widths_sg227)
top_local_idx = np.argsort(widths_sg227)[-2]  # 2nd widest

# Map back to full test-set index, then to df_valid index
sg227_test_positions = np.where(mask_sg227)[0]             # positions in test set
top_test_pos = sg227_test_positions[top_local_idx]         # position in test set
top_valid_idx = test_idx[top_test_pos]                     # row in df_valid

# Get structure metadata
row = df_valid.iloc[top_valid_idx]
structure_id  = int(row['id'])
true_bg       = float(y_test_band[top_test_pos])
pred_bg       = float(res_band['y_pred'][top_test_pos])
interval_lo   = float(res_band['lo'][top_test_pos])
interval_hi   = float(res_band['hi'][top_test_pos])
interval_w    = float(res_band['widths'][top_test_pos])
abs_error     = float(errors_sg227[top_local_idx])

# Also note top-3 for SI if needed
top3_local = np.argsort(widths_sg227)[-3:][::-1]
top3_ids   = [int(df_valid.iloc[test_idx[sg227_test_positions[i]]]['id'])
              for i in top3_local]

print("=" * 60)
print("HIGH-UNCERTAINTY OUTLIER (Fd-3m, band gap, SECOND-WIDEST CP interval by design)")
print("=" * 60)
print(f"  Structure ID   : {structure_id}")
print(f"  Spacegroup     : Fd-3m (227)")
print(f"  True band gap  : {true_bg:.3f} eV")
print(f"  Predicted      : {pred_bg:.3f} eV")
print(f"  CP interval    : [{interval_lo:.3f}, {interval_hi:.3f}] eV")
print(f"  Interval width : {interval_w:.3f} eV")
print(f"  Absolute error : {abs_error:.3f} eV")
if 'percent_atom_in' in row.index:
    print(f"  Indium fraction: {row['percent_atom_in']:.3f}")
if 'percent_atom_al' in row.index:
    print(f"  Al fraction    : {row['percent_atom_al']:.3f}")
if 'number_of_total_atoms' in row.index:
    print(f"  Atoms in cell  : {int(row['number_of_total_atoms'])}")
print(f"\n  Top-3 widest Fd-3m structure IDs: {top3_ids}")
# ---- M2/E33 verification: rank of the selected structure --------------------
_top3_w = [float(widths_sg227[i]) for i in np.argsort(widths_sg227)[-3:][::-1]]
for _rank, (_i, _w) in enumerate(zip(top3_ids, _top3_w), start=1):
    _mark = '  <-- SELECTED (Figure 9)' if _i == structure_id else ''
    print(f"    rank {_rank}: ID {_i}  width = {_w:.3f} eV{_mark}")
_sel_rank = top3_ids.index(structure_id) + 1 if structure_id in top3_ids else None
_rank_word = {1: 'widest', 2: 'second-widest', 3: 'third-widest'}.get(_sel_rank, 'UNKNOWN')
print(f"\n  CAPTION CHECK: the selected structure is the {_rank_word} Fd-3m interval.")
print(f"  Manuscript Figure 9 caption must read: '{_rank_word} among Fd-3m test structures'.")
print("=" * 60)

# ── 2. Load the geometry (skipped gracefully if Kaggle data absent) ───────────
geom_path = os.path.join(DATA_ROOT, str(structure_id), 'geometry.xyz')
if not os.path.exists(geom_path):
    print(f"\n  [skip] Geometry file not found: {geom_path}")
    print('  The NOMAD geometry files are not redistributed in this repository.')
    print('  To re-export the Figure 9 inputs (CIF + ASE views), download the')
    print('  Kaggle data into data/train/ (see README). All numerical results')
    print('  above are unaffected by this skip.')
else:
    atoms = ase_read(geom_path, format='aims')   # FHI-aims format

    print(f"\nLoaded structure: {len(atoms)} atoms")
    print(f"  Species: {set(atoms.get_chemical_symbols())}")
    print(f"  Cell (Å):\n{atoms.get_cell()}")

    # ── 3a. Export CIF for VESTA (recommended for publication) ───────────────────
    cif_path = os.path.join(RESULTS_DIR, f'outlier_sg227_{structure_id}.cif')
    ase_write(cif_path, atoms)
    print(f"\nCIF exported: {cif_path}")
    print("  -> Open this file in VESTA for publication-quality rendering.")
    print("  -> In VESTA: Objects > Polyhedra (add O coordination polyhedra)")
    print("     Style > Ball and Stick, export PNG at 300 DPI, white background.")

    # ── 3b. ASE + matplotlib figure (publishable Python alternative) ──────────────
    # Three views: a-axis, b-axis, c-axis. Use the most revealing one.
    # Rotation string format: 'Xdeg x, Ydeg y, Zdeg z'
    ROTATIONS = {
        'a-axis view': ('90x,0y,0z',   '(a)'),
        'b-axis view': ('90x,90y,0z',  '(b)'),
        'c-axis view': ('0x,0y,0z',    '(c)'),
    }

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.8))

    for ax, (view_label, (rot, panel_label)) in zip(axes, ROTATIONS.items()):
        plot_atoms(
            atoms,
            ax,
            radii       = 0.45,        # scale factor on covalent radii
            rotation    = rot,
            show_unit_cell = 2,        # 2 = show full unit cell box
            colors      = None,        # use ASE defaults: O=red, Al=grey, Ga=blue, In=purple
        )
        ax.set_title(f'{panel_label} {view_label}', fontsize=NATURE_FS,
                     fontweight='bold', pad=3)
        ax.axis('off')

    # Colour legend
    legend_elements = [
        mpatches.Patch(color='#FF6347', label='O'),    # tomato ~ ASE oxygen
        mpatches.Patch(color='#BFC2C7', label='Al'),
        mpatches.Patch(color='#6AAFB0', label='Ga'),
        mpatches.Patch(color='#A67CB5', label='In'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4,
               fontsize=NATURE_FS_SM, frameon=False,
               bbox_to_anchor=(0.5, -0.05))

    title_str = (
        f'Structure {structure_id} | Fd-3m (SG 227) | '
        f'E$_g$ = {true_bg:.2f} eV (pred: {pred_bg:.2f} eV) | '
        f'CP width = {interval_w:.2f} eV'
    )
    fig.suptitle(title_str, fontsize=NATURE_FS, y=1.02)
    plt.tight_layout()

    ase_fig_path = os.path.join(RESULTS_DIR, f'outlier_sg227_{structure_id}_ase.png')
    fig.savefig(ase_fig_path, dpi=300, bbox_inches='tight')
    print(f"\nASE figure saved: {ase_fig_path}")
    plt.show()

# ── 4. Caption text (fill in actual numbers after running) ───────────────────
print(f"""
=== SUGGESTED FIGURE CAPTION ===

Figure X | Crystal structure of a representative high-uncertainty outlier
in the Fd-3m (SG 227, spinel-type) subgroup. The structure (NOMAD ID
{structure_id}) has a DFT-PBE band gap of {true_bg:.2f} eV, with a
predicted value of {pred_bg:.2f} eV and a 90% CP interval of
[{interval_lo:.2f}, {interval_hi:.2f}] eV (width = {interval_w:.2f} eV),
the widest in the test set for this spacegroup. The cubic Fd-3m lattice
places cations on both tetrahedrally and octahedrally coordinated sites
(shown as O-centred polyhedra), producing a diversity of local Al/Ga/In
environments not represented in the other five spacegroups. This
structural heterogeneity reduces the similarity between Fd-3m test
residuals and the calibration distribution, driving both elevated
prediction error (|error| = {abs_error:.2f} eV) and the CP interval
widening observed for SG 227 in Table 4.
Colours: O (red), Al (grey), Ga (teal), In (purple).
Rendered with [VESTA / ASE v3.x].
""")



# SECTION 7: ALPHA-GRID + ENSEMBLE BASELINE

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
        ax1.annotate(f'α={a}',  (nc, ec),
                     xytext=(5, 4), textcoords='offset points',
                     fontsize=NATURE_FS_SM)

    # --- ADDED X-AXIS LABEL TO THE LEFT FIGURE ---
    ax1.set_xlabel('Nominal coverage (1 − α)', fontsize=NATURE_FS_MD)
    # ---------------------------------------------

    ax1.set_ylabel('Empirical coverage', fontsize=NATURE_FS_MD)

    # --- PANEL LABEL 1 REPLACES TITLE ---
    # Placed strictly outside the upper-left boundary
    ax1.text(0.0, 1.04, panel_label_1, transform=ax1.transAxes,
             fontsize=NATURE_FS_LG, fontweight='bold', va='bottom', ha='left')
    # ------------------------------------

    ax1.set_xlim(0.7, 1.02); ax1.set_ylim(0.7, 1.02)
    ax1.grid(lw=0.4, alpha=0.4)

    ax2.bar([f'α={a}' for a in alphas], mwidths,
            color=PALETTE['blue'], alpha=0.75, width=0.5)
    ax2.set_ylabel('Mean interval width', fontsize=NATURE_FS_MD)

    # --- PANEL LABEL 2 REPLACES TITLE ---
    # Placed strictly outside the upper-left boundary
    ax2.text(0.0, 1.04, panel_label_2, transform=ax2.transAxes,
             fontsize=NATURE_FS_LG, fontweight='bold', va='bottom', ha='left')
    # ------------------------------------

    ax2.tick_params(axis='x', labelsize=NATURE_FS_SM)
    ax2.grid(axis='y', lw=0.4, alpha=0.4)

    # Tight layout slightly restricted to protect outside labels
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # --- ADDED JPG EXPORT AT 600 DPI ---
    fig.savefig(os.path.join(RESULTS_DIR, fname.replace('.pdf', '.jpg')), dpi=600, bbox_inches='tight')  # JPG companion
    save_fig(fig, fname)

    for nc, ec, mw in zip(nom_covs, emp_covs, mwidths):
        print(f"  nom={nc:.2f}  emp={ec:.3f}  mean_width={mw:.4f}")

print("\n  [Formation Energy]")
# Passes (a) and (b) for Formation Energy
alpha_grid_fig(mapie_form, X_test_scaled, y_test_form,
               ALPHAS, '(a)', '(b)', 'alpha_grid_form_v5.pdf')

print("\n  [Band Gap]")
# Passes (c) and (d) for Band Gap
alpha_grid_fig(mapie_band, X_test_scaled, y_test_band,
               ALPHAS, '(c)', '(d)', 'alpha_grid_band_v5.pdf')

# ---- O5: exact alpha-grid values as an SI table ------------------------------
_ag_rows = []
for _a in ALPHAS:
    _, _pf = mapie_form.predict(X_test_scaled, alpha=_a)
    _, _pb = mapie_band.predict(X_test_scaled, alpha=_a)
    _ag_rows.append({
        'alpha': _a, 'nominal': 1 - _a,
        'cov_form':  round(regression_coverage_score(y_test_form, _pf[:,0,0], _pf[:,1,0]), 4),
        'width_form': round(float(np.mean(_pf[:,1,0] - _pf[:,0,0])), 4),
        'cov_band':  round(regression_coverage_score(y_test_band, _pb[:,0,0], _pb[:,1,0]), 4),
        'width_band': round(float(np.mean(_pb[:,1,0] - _pb[:,0,0])), 4),
    })
pd.DataFrame(_ag_rows).to_csv(os.path.join(RESULTS_DIR, 'alpha_grid_v7.csv'), index=False)
print("  Saved: alpha_grid_v7.csv (SI table for Figure 3)")


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

ENS_CKPT       = os.path.join(CHECKPOINT_DIR, 'ensemble_baseline_v5.pkl')
ENS_CKPT_DRIVE = os.path.join(RESULTS_DIR,     'ensemble_baseline_v5.pkl')

if os.path.exists(ENS_CKPT):
    print("  [checkpoint] Loading ensemble baseline from checkpoint...")
    ens_form, ens_band, ens_results = joblib.load(ENS_CKPT)
    for name, row in ens_results.items():
        print(f"  [{name}]  CV+: cov={row['cp_cov']:.4f} w={row['cp_width']:.4f} | "
              f"Ensemble: cov={row['ens_cov']:.4f} w={row['ens_width']:.4f} | "
              f"CV+ advantage: {row['delta']:+.4f}")
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
    shutil.copy2(ENS_CKPT, ENS_CKPT_DRIVE)
    print("  [checkpoint] ensemble_baseline_v5.pkl saved.")

# ---- O3: export ensemble baseline for the SI (works from checkpoint too) ----
pd.DataFrame(ens_results).T.reset_index().rename(columns={'index': 'Target'}).to_csv(
    os.path.join(RESULTS_DIR, 'ensemble_baseline_v7.csv'), index=False)
print("  Saved: ensemble_baseline_v7.csv (SI: answers 'why CP rather than an ensemble?')")

save_stage('ensemble_baseline_complete')


# SECTION 8: SHAP + KENDALL TAU 

print("\n--- Section 8: SHAP + Kendall Tau Stability ---")

base_form = mapie_form.estimator_.single_estimator_
base_band = mapie_band.estimator_.single_estimator_

# Checkpoint names updated to bypass the previous cache and force image regeneration
SHAP_VAL_CKPT       = os.path.join(CHECKPOINT_DIR, 'shap_values_v5_new2.npz')
SHAP_VAL_CKPT_DRIVE = os.path.join(RESULTS_DIR,     'shap_values_v5_new2.npz')
SHAP_META_CKPT      = os.path.join(CHECKPOINT_DIR, 'shap_meta_v5_new2.pkl')
SHAP_META_CKPT_DRIVE= os.path.join(RESULTS_DIR,     'shap_meta_v5_new2.pkl')

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

    # --- NATURE-STYLE X-AXIS LABEL ---
    main_ax.set_xlabel('SHAP value (impact on model output)', fontsize=NATURE_FS_MD)
    # ---------------------------------

  
    fig.text(0.02, 0.98, panel_label, fontsize=NATURE_FS_LG,
             fontweight='bold', va='top', ha='left')
    # -----------------------

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # --- ADDED JPG EXPORT AT 600 DPI ---
    fig.savefig(os.path.join(RESULTS_DIR, fname.replace('.pdf', '.jpg')), dpi=600, bbox_inches='tight')  # JPG companion
    save_fig(fig, fname)

    return sv, top_idx, top_names

# ---- Checkpoint: SHAP values + top-feature indices ----
if os.path.exists(SHAP_VAL_CKPT) and os.path.exists(SHAP_META_CKPT):
    print("  [checkpoint] Loading SHAP values from checkpoint...")
    d         = np.load(SHAP_VAL_CKPT)
    sv_form   = d['sv_form']
    sv_band   = d['sv_band']
    meta      = joblib.load(SHAP_META_CKPT)
    top_form  = meta['top_form']
    top_band  = meta['top_band']
    topnames_form = meta['topnames_form']
    topnames_band = meta['topnames_band']
    print(f"  top_form[0]={topnames_form[0]}  top_band[0]={topnames_band[0]}")
else:
    # Passes (a) for Formation Energy
    sv_form, top_form, topnames_form = shap_beeswarm_nature(
        base_form, X_test_scaled, FEATURE_NAMES,
        'Formation energy', '(a)', 'shap_beeswarm_form_v5.pdf')

    # Passes (b) for Band Gap
    sv_band, top_band, topnames_band = shap_beeswarm_nature(
        base_band, X_test_scaled, FEATURE_NAMES,
        'Band gap', '(b)', 'shap_beeswarm_band_v5.pdf')

    np.savez_compressed(SHAP_VAL_CKPT, sv_form=sv_form, sv_band=sv_band)
    joblib.dump({'top_form': top_form, 'top_band': top_band,
                 'topnames_form': topnames_form, 'topnames_band': topnames_band},
                SHAP_META_CKPT, compress=3)
    shutil.copy2(SHAP_VAL_CKPT,  SHAP_VAL_CKPT_DRIVE)
    shutil.copy2(SHAP_META_CKPT, SHAP_META_CKPT_DRIVE)
    print("  [checkpoint] SHAP values + metadata saved.")

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

# ---- Split tau checkpoints: one per target ----
TAU_FORM_CKPT       = os.path.join(CHECKPOINT_DIR, 'shap_tau_form_v5.pkl')
TAU_FORM_CKPT_DRIVE = os.path.join(RESULTS_DIR,     'shap_tau_form_v5.pkl')
TAU_BAND_CKPT       = os.path.join(CHECKPOINT_DIR, 'shap_tau_band_v5.pkl')
TAU_BAND_CKPT_DRIVE = os.path.join(RESULTS_DIR,     'shap_tau_band_v5.pkl')

# Formation tau
if os.path.exists(TAU_FORM_CKPT):
    print("  [checkpoint] Loading Formation tau from checkpoint...")
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
    shutil.copy2(TAU_FORM_CKPT, TAU_FORM_CKPT_DRIVE)
    print("  [checkpoint] shap_tau_form_v5.pkl saved.")

# Band gap tau
if os.path.exists(TAU_BAND_CKPT):
    print("  [checkpoint] Loading Band Gap tau from checkpoint...")
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
    shutil.copy2(TAU_BAND_CKPT, TAU_BAND_CKPT_DRIVE)
    print("  [checkpoint] shap_tau_band_v5.pkl saved.")

# Keep combined tau_results for backward compatibility with Section 11/13
tau_results = {
    'Formation': r_f,
    'Band Gap':  r_b,
}

save_stage('shap_complete')


# SECTION 8B (O4): STANDARD TOP-K AGREEMENT METRICS (JACCARD + RBO)

print("\n--- Section 8B: Top-k Agreement (Jaccard / RBO, 5 seeds) ---")

TOPK_CKPT       = os.path.join(CHECKPOINT_DIR, 'topk_agreement_v7.pkl')
TOPK_CKPT_DRIVE = os.path.join(RESULTS_DIR,     'topk_agreement_v7.pkl')

def _top10_lists(X_tr, y_tr, params, X_te, seeds, k=10):
    lists = []
    for s in seeds:
        p = dict(params); p.update(GPU_PARAMS)
        clean = {kk: v for kk, v in p.items() if kk not in ('metric', 'objective')}
        m = lgb.LGBMRegressor(**clean, n_estimators=700, n_jobs=LGBM_JOBS,
                              verbose=-1, max_bin=63, seed=s, force_col_wise=True)
        m.fit(X_tr, y_tr)
        sv = shap.TreeExplainer(m).shap_values(X_te)
        lists.append(list(np.argsort(np.abs(sv).mean(axis=0))[::-1][:k]))
        del m, sv; gc.collect()
    return lists

def _jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b)

def _rbo(list_a, list_b, p=0.9):
    """Extrapolated Rank-Biased Overlap for two truncated rankings."""
    k = min(len(list_a), len(list_b))
    a_seen, b_seen = set(), set()
    x = 0
    s = 0.0
    for d in range(1, k + 1):
        a_seen.add(list_a[d - 1]); b_seen.add(list_b[d - 1])
        x = len(a_seen & b_seen)
        s += (p ** (d - 1)) * x / d
    return (1 - p) * s + (p ** k) * x / k

if os.path.exists(TOPK_CKPT):
    topk_results = joblib.load(TOPK_CKPT)
    print("  Loaded top-k agreement from checkpoint.")
else:
    topk_results = {}
    for _name, _y in (('Formation', y_train_form), ('Band Gap', y_train_band)):
        _params = best_params_form if _name == 'Formation' else best_params_band
        print(f"  Refitting 5 seeds for {_name} (top-10 lists)...")
        _lists = _top10_lists(X_train_scaled, _y, _params,
                              X_test_scaled, STABILITY_SEEDS)
        _pairs = [(i, j) for i in range(len(_lists)) for j in range(i + 1, len(_lists))]
        _jac = [_jaccard(_lists[i], _lists[j]) for i, j in _pairs]
        _rbo_v = [_rbo(_lists[i], _lists[j]) for i, j in _pairs]
        topk_results[_name] = {
            'top10_lists':  _lists,
            'jaccard_mean': float(np.mean(_jac)),  'jaccard_min': float(np.min(_jac)),
            'rbo_mean':     float(np.mean(_rbo_v)), 'rbo_min':    float(np.min(_rbo_v)),
        }
    joblib.dump(topk_results, TOPK_CKPT, compress=3)
    shutil.copy2(TOPK_CKPT, TOPK_CKPT_DRIVE)
    print("  [checkpoint] topk_agreement_v7.pkl saved.")

_tk_rows = []
for _name, _r in topk_results.items():
    print(f"  [{_name}]  Jaccard(top-10): mean={_r['jaccard_mean']:.3f} "
          f"min={_r['jaccard_min']:.3f} | RBO(p=0.9): mean={_r['rbo_mean']:.3f} "
          f"min={_r['rbo_min']:.3f}")
    _tk_rows.append({'Target': _name,
                     'Jaccard_mean': round(_r['jaccard_mean'], 4),
                     'Jaccard_min':  round(_r['jaccard_min'], 4),
                     'RBO_mean':     round(_r['rbo_mean'], 4),
                     'RBO_min':      round(_r['rbo_min'], 4)})
pd.DataFrame(_tk_rows).to_csv(os.path.join(RESULTS_DIR, 'topk_agreement_v7.csv'), index=False)
print("  Saved: topk_agreement_v7.csv")


# SECTION 9: INDIUM PROXY CHECK

print("\n--- Section 9: Indium Proxy Check (Task A) ---")

feat_band_top = X_test_scaled[:, top_band[0]]
feat_form_top = X_test_scaled[:, top_form[0]]
rho_b, p_b    = spearmanr(feat_band_top, in_frac_test)
rho_f, p_f    = spearmanr(feat_form_top, in_frac_test)

print(f"  Band Gap   top ({topnames_band[0][:22]}) vs In%: rho={rho_b:.4f}  p={p_b:.2e}")
print(f"  Formation  top ({topnames_form[0][:22]}) vs In%: rho={rho_f:.4f}  p={p_f:.2e}")
if abs(rho_b) >= 0.90:
    print("  WARNING: rho >= 0.90 — top band-gap feature is a strong In proxy.")
else:
    print("  rho < 0.90 — SOAP captures geometry beyond In composition.")

fig, axes = plt.subplots(1, 2, figsize=(NATURE_DOUBLE, NATURE_HEIGHT))

# Added enumerate and removed the target names list
for i, (ax, feat, rho, p) in enumerate(zip(
        axes, [feat_form_top, feat_band_top],
        [rho_f, rho_b], [p_f, p_b])):

    ax.scatter(in_frac_test, feat, s=5, alpha=0.35, c=PALETTE['blue'],
               linewidths=0, rasterized=True)
    ax.set_xlabel('Indium atom fraction', fontsize=NATURE_FS_MD)
    ax.set_ylabel('Top-ranked feature value (standardized)', fontsize=NATURE_FS_MD)

    # --- PANEL LABEL (a) & (b) REPLACES TITLE ---
    # Placed strictly outside the upper-left boundary
    panel_label = '(a)' if i == 0 else '(b)'
    ax.text(0.0, 1.04, panel_label, transform=ax.transAxes,
            fontsize=NATURE_FS_LG, fontweight='bold', va='bottom', ha='left')
    # --------------------------------------------

    ax.text(0.03, 0.95, f'ρ = {rho:.3f}\np = {p:.1e}',
            transform=ax.transAxes, va='top', fontsize=NATURE_FS_SM,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8,
                      ec=PALETTE['grey'], lw=0.5))
    ax.grid(lw=0.4, alpha=0.4)

# Tight layout natively handles the outside text bounding boxes
plt.tight_layout()
save_fig(fig, 'indium_proxy_v5.pdf')


# SECTION 9B (M1): MECHANICAL SOAP FEATURE-INDEX DECODING

print("\n--- Section 9B: SOAP Feature Index Decoding (M1) ---")
try:
    import importlib.metadata as _ilm
    _soap_dec = SOAP(species=sorted({'Al', 'Ga', 'In', 'O'}), periodic=True,
                     r_cut=6.0, n_max=9, l_max=8, sparse=False)
    _PAIRS = [('O','O'),('Al','O'),('Ga','O'),('In','O'),('Al','Al'),
              ('Al','Ga'),('Al','In'),('Ga','Ga'),('Ga','In'),('In','In')]
    def decode_soap_index(gidx, n_max=9):
        for _p in _PAIRS:
            _loc = _soap_dec.get_location(_p)
            if _loc.start <= gidx < _loc.stop:
                _off   = gidx - _loc.start
                _same  = _p[0] == _p[1]
                _per_l = n_max * (n_max + 1) // 2 if _same else n_max * n_max
                _l, _r = divmod(_off, _per_l)
                if _same:
                    _n = 0
                    while _r >= n_max - _n:
                        _r -= n_max - _n
                        _n += 1
                    _npr = _n + _r
                else:
                    _n, _npr = divmod(_r, n_max)
                return {'pair': f'{_p[0]}-{_p[1]}', 'n': _n + 1,
                        'n_prime': _npr + 1, 'l': _l}
        raise ValueError(f'index {gidx} out of range')
    def decode_feature_name(fname):
        _agg, _idx = fname.split('_')
        return {'feature': fname, 'aggregation': _agg,
                **decode_soap_index(int(_idx))}
    print(f"  DScribe version: {_ilm.version('dscribe')}")
    _dec_rows = [decode_feature_name(f)
                 for f in list(topnames_form[:5]) + list(topnames_band[:5])]
    _df_dec = pd.DataFrame(_dec_rows)[['feature','aggregation','pair','n','n_prime','l']]
    print(_df_dec.to_string(index=False))
    _df_dec.to_csv(os.path.join(RESULTS_DIR, 'soap_feature_decoding_v7.csv'), index=False)
    print("  Saved: soap_feature_decoding_v7.csv "
          "(regenerate every decoded example in Section 3.4 from this table)")
except Exception as _e:
    print(f"  Decoding skipped ({_e})")


# SECTION 6.4 : INTERVAL WIDTH INFORMATIVENESS

print("\n--- Section 6.4: Interval Width Informativeness ---")

def width_informativeness(y_te, y_pred, widths, target_name):
    abs_err = np.abs(y_te - y_pred)
    r_p, p_p = pearsonr(widths, abs_err)

    # Rank structures into width quartiles and compute MAE per quartile
    quartile_labels = pd.qcut(widths, q=4,
                               labels=['Q1 (narrow)', 'Q2', 'Q3', 'Q4 (wide)'])
    quartile_df = pd.DataFrame({
        'quartile': quartile_labels,
        'abs_error': abs_err,
        'width': widths,
    })
    q_stats = quartile_df.groupby('quartile', observed=True).agg(
        n=('abs_error', 'count'),
        mean_abs_err=('abs_error', 'mean'),
        mean_width=('width', 'mean'),
    ).reset_index()

    print(f"\n  [{target_name}] Pearson r = {r_p:.4f}  p = {p_p:.2e}")
    print(f"  {'Quartile':<14} {'N':>5} {'MeanWidth':>10} {'MeanAbsErr':>11}")
    for _, row in q_stats.iterrows():
        print(f"  {row['quartile']:<14} {int(row['n']):>5} "
              f"{row['mean_width']:>10.4f} {row['mean_abs_err']:>11.4f}")

    # Scatter plot
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(widths, abs_err, s=8, alpha=0.3, c='steelblue')
    m, b = np.polyfit(widths, abs_err, 1)
    xfit = np.linspace(widths.min(), widths.max(), 200)
    ax.plot(xfit, m * xfit + b, 'r-', lw=2,
            label=f'Pearson r={r_p:.3f}  p={p_p:.1e}')
    ax.set_xlabel('CP interval width')
    ax.set_ylabel('|prediction error|')
    ax.set_title(f'{target_name}: Interval width vs. prediction error')
    ax.legend(fontsize=9)
    plt.tight_layout()
    import re
    fname = re.sub(r'[^A-Za-z0-9]+', '_', target_name).strip('_')
    plt.savefig(os.path.join(OUTPUT_DIR, f'width_informativeness_{fname}_v4.pdf'),
                dpi=200, bbox_inches='tight')
    plt.close()

    return {'pearson_r': r_p, 'pearson_p': p_p, 'quartile_stats': q_stats}

info_form = width_informativeness(
    y_test_form, res_form['y_pred'], res_form['widths'],
    'Formation Energy (eV/atom)')
info_band = width_informativeness(
    y_test_band, res_band['y_pred'], res_band['widths'],
    'Band Gap (eV)')

# =============================================================================
# SECTION 10: CROSS-LATTICE GENERALISATION EXPERIMENT  (fully checkpointed)
# =============================================================================
print("\n--- Section 10: Cross-Lattice Generalisation (Novel) ---")

CL_CACHE = os.path.join(RESULTS_DIR, 'cross_lattice_results_v5.json')

if os.path.exists(CL_CACHE):
    with open(CL_CACHE) as f:
        cl_cache = json.load(f)
    cross_lat_form = cl_cache['formation']
    cross_lat_band = cl_cache['bandgap']
    print("  Loaded cross-lattice results from cache.")
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

        # Per-lattice checkpoint
        cl_ckpt = os.path.join(CHECKPOINT_DIR, f'cl_{held_sg}_v5.pkl')
        if os.path.exists(cl_ckpt):
            row_f, row_b = joblib.load(cl_ckpt)
            print(f"  {held_name:<12} (loaded from checkpoint)")
        else:
            for target, params_cl, all_y, rows_out in [
                    ('form', best_params_form, all_form, cross_lat_form),
                    ('band', best_params_band, all_band, cross_lat_band)]:
                pass  # handled below

            # Train CP models for both targets simultaneously per lattice
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
            # Mirror to results/ root immediately
            shutil.copy2(cl_ckpt, os.path.join(RESULTS_DIR, f'cl_{held_sg}_v5.pkl'))
            del mapie_cl_form, mapie_cl_band; gc.collect()

        cross_lat_form.append(row_f)
        cross_lat_band.append(row_b)
        print(f"  {held_name:<12} {row_f['N']:>6}  "
              f"{row_f['MAE']:>9.4f} {row_f['Coverage']:>9.3f} {row_f['MeanWidth']:>8.4f}  "
              f"{row_b['MAE']:>9.4f} {row_b['Coverage']:>9.3f} {row_b['MeanWidth']:>8.4f}")

    # Save aggregated results
    with open(CL_CACHE, 'w') as f:
        json.dump({'formation': cross_lat_form, 'bandgap': cross_lat_band}, f, indent=2)
    save_stage('cross_lattice_complete')

df_cl_form = pd.DataFrame(cross_lat_form)
df_cl_band = pd.DataFrame(cross_lat_band)

# E37: normalise lattice labels stored in pre-v7 checkpoints
for _df in (df_cl_form, df_cl_band):
    _df['Lattice'] = _df['Lattice'].replace(LABEL_FIX)

# O1 (optional part): Clopper-Pearson 95% CIs for Table 5 coverages
for _df in (df_cl_form, df_cl_band):
    _ks = np.rint(_df['Coverage'].values * _df['N'].values).astype(int)
    _ci = [clopper_pearson(k, nn) for k, nn in zip(_ks, _df['N'].values)]
    _df['Cov_CI95_lo'] = [round(c[0], 4) for c in _ci]
    _df['Cov_CI95_hi'] = [round(c[1], 4) for c in _ci]
print("\n  Cross-lattice coverage 95% CIs (Table 5):")
for _tag, _df in (('formation', df_cl_form), ('band gap', df_cl_band)):
    for _, _r in _df.iterrows():
        print(f"    [{_tag:<9}] {_r['Lattice']:<8} {_r['Coverage']:.3f} "
              f"[{_r['Cov_CI95_lo']:.3f}, {_r['Cov_CI95_hi']:.3f}]  (n={int(_r['N'])})")

df_cl_form.to_csv(os.path.join(RESULTS_DIR,'cross_lattice_formation_v5.csv'), index=False)
df_cl_band.to_csv(os.path.join(RESULTS_DIR,'cross_lattice_bandgap_v5.csv'), index=False)

# ---- Cross-lattice figure (Nature double-column, 4-panel) ----
fig, axes = plt.subplots(2, 2, figsize=(NATURE_DOUBLE, NATURE_HEIGHT * 2.1))

for col, (df_cl, tit) in enumerate([(df_cl_form,'Formation energy'),
                                     (df_cl_band,'Band gap')]):
    x   = np.arange(len(df_cl))
    lbl = df_cl['Lattice'].tolist()

    # ==========================================
    # Panel Top Row: MAE (a) and (b)
    # ==========================================
    axes[0, col].bar(x, df_cl['MAE'], color=PALETTE['blue'],
                     alpha=0.75, width=0.6, zorder=3)
    axes[0, col].set_xticks(x)
    axes[0, col].set_xticklabels(lbl, rotation=45, ha='right', va='top',
                                 fontsize=NATURE_FS_SM)
    axes[0, col].tick_params(axis='x', pad=2)

    axes[0, col].set_ylabel('MAE', fontsize=NATURE_FS_MD)
    axes[0, col].grid(axis='y', lw=0.4, alpha=0.4, zorder=0)

    # Add Panel Label (a) / (b)
    lbl_top = '(a)' if col == 0 else '(b)'
    axes[0, col].text(0.0, 1.04, lbl_top, transform=axes[0, col].transAxes,
                      fontsize=NATURE_FS_LG, fontweight='bold', va='bottom', ha='left')

    # ==========================================
    # Panel Bottom Row: CP coverage (c) and (d)
    # ==========================================
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

    # --- LEGEND HEIGHT FIX APPLIED HERE ---
    axes[1, col].legend(fontsize=NATURE_FS_SM, loc='lower right',
                        bbox_to_anchor=(1.0, 1.01), frameon=False)
    # --------------------------------------

    axes[1, col].grid(axis='y', lw=0.4, alpha=0.4, zorder=0)

    # Add Panel Label (c) / (d)
    lbl_bot = '(c)' if col == 0 else '(d)'
    axes[1, col].text(0.0, 1.04, lbl_bot, transform=axes[1, col].transAxes,
                      fontsize=NATURE_FS_LG, fontweight='bold', va='bottom', ha='left')

    # Annotate N (with zorder and bbox to protect text from grid lines)
    for xi, (_, row) in zip(x, df_cl.iterrows()):
        axes[1, col].text(xi, row['Coverage'] + 0.02, f'n={row["N"]}',
                          ha='center', va='bottom', fontsize=5, zorder=10,
                          bbox=dict(facecolor='white', alpha=0.85, edgecolor='none', pad=1))

# Lifted the suptitle slightly to clear the panel labels
plt.suptitle('Cross-lattice generalisation experiment\n'
             '(one lattice held out per run, trained on remaining 5)',
             fontsize=NATURE_FS_LG, y=1.05)

plt.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, 'cross_lattice_v5.jpg'), dpi=600, bbox_inches='tight')  # JPG companion
save_fig(fig, 'cross_lattice_v5.pdf')


# FIGURE 9: Mean Width vs MAE Scatter Plot (Cross-Lattice)

print("\n  Generating Cross-Lattice Scatter Plot (Figure 9)...")

# 1. Calculate the correlations directly from the freshly built dataframes
r_form_cl = pearsonr(df_cl_form['MeanWidth'], df_cl_form['MAE'])[0]
r_band_cl = pearsonr(df_cl_band['MeanWidth'], df_cl_band['MAE'])[0]

print(f"  Pearson(mean_width, MAE) across lattices:")
print(f"    Formation: r = {r_form_cl:.4f}")
print(f"    Band Gap:  r = {r_band_cl:.4f}")

# 2. Draw the figure
fig, axes = plt.subplots(1, 2, figsize=(NATURE_DOUBLE, NATURE_HEIGHT))

for i, (ax, df, r_val, title) in enumerate(zip(
    axes,
    [df_cl_form, df_cl_band],
    [r_form_cl, r_band_cl],
    ['Formation energy', 'Band gap']
)):
    x_vals = df['MeanWidth'].values
    y_vals = df['MAE'].values

    # Scatter plot of the 6 lattices
    ax.scatter(x_vals, y_vals, s=30, color=PALETTE['blue'], alpha=0.8, zorder=3)

    # Calculate and plot the line of best fit
    m, b = np.polyfit(x_vals, y_vals, 1)
    x_fit = np.linspace(x_vals.min() - 0.01, x_vals.max() + 0.01, 100)
    ax.plot(x_fit, m * x_fit + b, color=PALETTE['red'], ls='--', lw=NATURE_LW,
            label=f'r = {r_val:.2f}', zorder=2)

    # Formatting
    ax.set_xlabel('Mean interval width', fontsize=NATURE_FS_MD)
    ax.set_ylabel('MAE (eV/atom)' if i == 0 else 'MAE (eV)', fontsize=NATURE_FS_MD)

    # Add Lattice Names next to the dots so readers know which dot is which
    for xi, yi, label in zip(x_vals, y_vals, df['Lattice']):
        ax.annotate(label, (xi, yi), textcoords="offset points", xytext=(5,0),
                    ha='left', fontsize=NATURE_FS_SM)

    # Panel Labels (a) and (b)
    panel_label = '(a)' if i == 0 else '(b)'
    ax.text(0.0, 1.04, panel_label, transform=ax.transAxes,
            fontsize=NATURE_FS_LG, fontweight='bold', va='bottom', ha='left')

    ax.legend(fontsize=NATURE_FS_SM, loc='best', frameon=False)
    ax.grid(lw=0.4, alpha=0.4, zorder=0)

plt.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(os.path.join(RESULTS_DIR, 'cross_lattice_scatter_v5.jpg'), dpi=600, bbox_inches='tight')  # JPG companion
save_fig(fig, 'cross_lattice_scatter_v5.pdf')
print("  Figure saved: cross_lattice_scatter_v5.pdf (and .jpg)")


# =============================================================================
# VERIFICATION: does Fd-3m have the widest spread of Max_0039 in the test set?
# (Supports the claim in Section 3.4: "produces the test set's widest spread
#  of Max_0039 values". Report the verdict line back before submission.)
# =============================================================================
import numpy as np
import pandas as pd

col = FEATURE_NAMES.index('Max_0039')
vals = X_test_scaled[:, col]

rows = []
for sg_int, sg_name in LATTICE_MAP.items():
    msk = (sg_test == sg_int)
    v = vals[msk]
    rows.append({
        'Lattice': sg_name,
        'N': int(msk.sum()),
        'Std': round(float(np.std(v)), 4),
        'Range': round(float(np.ptp(v)), 4),          # max - min
        'IQR': round(float(np.percentile(v, 75) - np.percentile(v, 25)), 4),
    })

df = pd.DataFrame(rows).sort_values('Std', ascending=False).reset_index(drop=True)
print(df.to_string(index=False))

top_std, top_rng = df.iloc[0]['Lattice'], df.sort_values('Range', ascending=False).iloc[0]['Lattice']
print(f"\nWidest spread by Std   : {top_std}")
print(f"Widest spread by Range : {top_rng}")

if top_std == 'Fd-3m':
    print("\nVERDICT: CLAIM SUPPORTED - keep the Section 3.4 sentence as written.")
else:
    print(f"\nVERDICT: CLAIM NOT SUPPORTED - {top_std} has the widest Max_0039 spread.")
    print("Report this back so the Section 3.4 sentence and the linked argument get reworked.")

df.to_csv(os.path.join(RESULTS_DIR, 'max0039_spread_by_spacegroup_v7.csv'), index=False)
print("\nSaved: max0039_spread_by_spacegroup_v7.csv")
