# =============================================================================
# Leakage check by TRAINING-set composition counts
# Reproduces the numbers in the last paragraph of Section 3.5.
# Inputs : leakage_defence_v5.csv (per test composition group: n, mae_form, mae_band)
#          train.csv (NOMAD 2018, downloaded from GitHub)
# Output : leakage_training_counts_v7.csv in the same folder (results/ inside the repo)
# =============================================================================
import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedShuffleSplit


if os.path.exists(os.path.join('results', 'leakage_defence_v5.csv')):
    GDRIVE_DIR = 'results'
else:
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        GDRIVE_DIR = '/content/drive/MyDrive/NOMAD_Uncer_Quan'
    except Exception:
        GDRIVE_DIR = os.environ.get('GDRIVE_DIR', './nomad_outputs')
print(f'Reading and writing in: {GDRIVE_DIR}')

TRAIN_URL = ('https://raw.githubusercontent.com/csutton7/'
             'nomad_2018_kaggle_dataset/master/train.csv')
SEED = 42

df = pd.read_csv(TRAIN_URL)
L = pd.read_csv(os.path.join(GDRIVE_DIR, 'leakage_defence_v5.csv'))

# Same split as the main pipeline (Section 3 of Uncertainty_Quan): stratified by spacegroup.
sg = df['spacegroup'].values.astype(int)
tr, te = next(StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
              .split(np.zeros((len(df), 1)), sg))

def make_group(row):   # identical to the pipeline's composition-group key
    return (f"SG{int(row['spacegroup'])}_"
            f"Al{round(row['percent_atom_al'], 1)}_"
            f"Ga{round(row['percent_atom_ga'], 1)}")

groups = df.apply(make_group, axis=1).values

# Sanity check: the reproduced TEST groups must match leakage_defence_v5.csv exactly.
test_counts = pd.Series(groups[te]).value_counts()
assert len(test_counts) == len(L), 'number of test groups differs'
assert (L.set_index('group')['n'] == test_counts.reindex(L['group']).values).all(), \
    'test group sizes differ: the split was not reproduced'
print(f'Split reproduced: {len(L)} test composition groups, {L["n"].sum()} test structures.')

train_counts = pd.Series(groups[tr]).value_counts()
L['n_train'] = L['group'].map(train_counts).fillna(0).astype(int)

rho_f, p_f = spearmanr(L['n_train'], L['mae_form'])
rho_b, p_b = spearmanr(L['n_train'], L['mae_band'])
print(f"Training counterparts per test group: median {L['n_train'].median():.0f}, "
      f"range {L['n_train'].min()}-{L['n_train'].max()}")
print(f'Spearman(n_train, MAE_form) = {rho_f:+.3f}  p = {p_f:.3f}')
print(f'Spearman(n_train, MAE_band) = {rho_b:+.3f}  p = {p_b:.3f}')
print(f"Groups with a single test structure: {(L['n'] == 1).sum()} of {len(L)}")

# Lattice confound control. Fd-3m groups have many training counterparts AND high errors, so the
# pooled correlation mixes a lattice effect with a composition effect. Two controls:
L['sg'] = L['group'].str.extract(r'SG(\d+)')[0].astype(int)
for col in ['n_train', 'mae_form', 'mae_band']:          # (1) rank within each spacegroup
    L['r_' + col] = L.groupby('sg')[col].rank(pct=True)
rw_f, pw_f = spearmanr(L['r_n_train'], L['r_mae_form'])
rw_b, pw_b = spearmanr(L['r_n_train'], L['r_mae_band'])
print(f'Within-spacegroup ranks: rho_form = {rw_f:+.3f} (p = {pw_f:.3f}), '
      f'rho_band = {rw_b:+.3f} (p = {pw_b:.3f})')
nf = L[L['sg'] != 227]                                     # (2) exclude Fd-3m
rx_f, px_f = spearmanr(nf['n_train'], nf['mae_form'])
rx_b, px_b = spearmanr(nf['n_train'], nf['mae_band'])
print(f'Excluding Fd-3m ({len(nf)} groups): rho_form = {rx_f:+.3f} (p = {px_f:.3f}), '
      f'rho_band = {rx_b:+.3f} (p = {px_b:.3f})')
top = L.loc[L['n'].idxmax()]
print(f"Most frequent test group: {top['group']} (N = {int(top['n'])}, "
      f"{int(top['n_train'])} training counterparts)")

L['bin'] = pd.cut(L['n_train'], [-1, 0, 2, 5, 10, 1000], labels=['0', '1-2', '3-5', '6-10', '>10'])
summary = L.groupby('bin', observed=True).apply(lambda d: pd.Series({
    'groups': len(d), 'structures': int(d['n'].sum()),
    'mae_form_meV_atom': 1000 * (d['n'] * d['mae_form']).sum() / d['n'].sum(),
    'mae_band_meV': 1000 * (d['n'] * d['mae_band']).sum() / d['n'].sum()}))
print('\nStructure-weighted test MAE by number of training counterparts:')
print(summary.round(1).to_string())

out = os.path.join(GDRIVE_DIR, 'leakage_training_counts_v7.csv')
L.drop(columns=['bin', 'r_n_train', 'r_mae_form', 'r_mae_band']).to_csv(out, index=False)
print(f'\nSaved: {out}')
