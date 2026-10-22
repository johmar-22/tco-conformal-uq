# =============================================================================
# Duplicate-entry check for Section 2.1 (data curation)
#
# 1) Finds entries in NOMAD 2018 train.csv that are identical in spacegroup, atom count,
#    Al/Ga/In fractions, all six lattice parameters, formation energy and band gap.
# 2) Reproduces the paper's train/test split (StratifiedShuffleSplit by spacegroup,
#    test_size = 0.2, random_state = 42) and counts test structures with an identical
#    entry in training.
# 3) If the geometry files are available,
#    also checks whether the entries of each group are the same atomic structure, allowing for
#    symmetry operations (see the geometry-check section below).
#
# Output: duplicate_entries_v7.csv (one row per duplicate structure)
# =============================================================================
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

if os.path.exists('results'):                         # inside the GitHub repository
    OUT_DIR, GEOM_DIR = 'results', os.path.join('data', 'train')
else:
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        OUT_DIR = '/content/drive/MyDrive/NOMAD_Uncer_Quan'
    except Exception:
        OUT_DIR = os.environ.get('GDRIVE_DIR', './nomad_outputs')
    GEOM_DIR = os.path.join(OUT_DIR, 'train')
os.makedirs(OUT_DIR, exist_ok=True)

TRAIN_URL = ('https://raw.githubusercontent.com/csutton7/'
             'nomad_2018_kaggle_dataset/master/train.csv')
df = pd.read_csv(TRAIN_URL)
assert len(df) == 2400

KEY = ['spacegroup', 'number_of_total_atoms', 'percent_atom_al', 'percent_atom_ga',
       'percent_atom_in', 'lattice_vector_1_ang', 'lattice_vector_2_ang', 'lattice_vector_3_ang',
       'lattice_angle_alpha_degree', 'lattice_angle_beta_degree', 'lattice_angle_gamma_degree',
       'formation_energy_ev_natom', 'bandgap_energy_ev']

tr, te = next(StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
              .split(np.zeros((len(df), 1)), df['spacegroup'].values))
df['partition'] = 'train'
df.loc[te, 'partition'] = 'test'

dup = df[df.duplicated(KEY, keep=False)].copy()
dup['group'] = dup.groupby(KEY).ngroup()
groups = dup.groupby('group')
cross = [g for g, d in groups if set(d['partition']) == {'train', 'test'}]
n_test_with_twin = int(sum((d['partition'] == 'test').sum() for g, d in groups if g in cross))

print(f'Duplicate groups: {groups.ngroups}   duplicate structures: {len(dup)}')
print('By spacegroup:', dup['spacegroup'].value_counts().sort_index().to_dict())
print(f'Groups spanning train and test: {len(cross)}   '
      f'test structures with an identical training entry: {n_test_with_twin} of {len(te)}')
for g in cross:
    d = groups.get_group(g)
    print(f"   test id(s) {list(d.loc[d.partition == 'test', 'id'])}  <->  "
          f"train id(s) {list(d.loc[d.partition == 'train', 'id'])}  (SG {d['spacegroup'].iloc[0]})")


# ---- geometry check (symmetry-aware) ------------------------------------------------------
# Two independent tests are used; a pair counts as the same structure only if BOTH agree:
#   (a) ASE SymmetryEquivalenceCheck (allows rotations, reflections and lattice re-choices);
#   (b) identical sorted interatomic-distance lists for every element pair within 6 A.
import itertools
from ase.io import read as ase_read
from ase.neighborlist import neighbor_list
from ase.utils.structure_comparator import SymmetryEquivalenceCheck

CHECK = SymmetryEquivalenceCheck(angle_tol=1.0, ltol=0.05, stol=0.05)


def distance_fingerprint(atoms, rc=6.0):
    i, j, d = neighbor_list('ijd', atoms, rc)
    s = np.array(atoms.get_chemical_symbols())
    fp = {}
    for x, y in itertools.combinations_with_replacement(sorted(set(s)), 2):
        m = ((s[i] == x) & (s[j] == y)) | ((s[i] == y) & (s[j] == x))
        fp[(x, y)] = np.sort(np.round(d[m], 3))
    return fp


def same_structure(a, b):
    fa, fb = distance_fingerprint(a), distance_fingerprint(b)
    fp_same = fa.keys() == fb.keys() and all(
        len(fa[k]) == len(fb[k]) and np.allclose(fa[k], fb[k], atol=2e-3) for k in fa)
    return bool(CHECK.compare(a, b)) and fp_same


dup['geometry_identical'] = pd.NA
if os.path.isdir(GEOM_DIR):
    group_ok = {}
    for g, d in groups:
        ids = list(d['id'])
        paths = [os.path.join(GEOM_DIR, str(i), 'geometry.xyz') for i in ids]
        if not all(os.path.exists(p) for p in paths):
            continue
        atoms = [ase_read(p, format='aims') for p in paths]
        verdicts = [True] + [same_structure(atoms[0], a) for a in atoms[1:]]
        for i, v in zip(ids, verdicts):
            dup.loc[dup['id'] == i, 'geometry_identical'] = v
        group_ok[g] = all(verdicts)
    n_ok = sum(group_ok.values())
    print(f'\nGeometry check (symmetry-aware): {len(group_ok)} of {groups.ngroups} groups compared; '
          f'groups whose entries are all the same structure: {n_ok}')
    for g, ok in group_ok.items():
        if not ok:
            print(f"   group {g}: NOT the same structure -> ids {list(groups.get_group(g)['id'])}")
else:
    print(f'\nGeometry files not found in {GEOM_DIR}; geometry check skipped.')

out = os.path.join(OUT_DIR, 'duplicate_entries_v7.csv')
dup.sort_values(['group', 'id'])[['group', 'id', 'spacegroup', 'percent_atom_al', 'percent_atom_ga',
                                  'percent_atom_in', 'formation_energy_ev_natom', 'bandgap_energy_ev',
                                  'partition', 'geometry_identical']].to_csv(out, index=False)
print(f'Saved: {out}')
