# Conformal Uncertainty Quantification for TCO Property Prediction

Code and data for:

> **Conformal Uncertainty Quantification for Transparent Conductive Oxide
> Property Prediction: Coverage Diagnostics Across Crystal Symmetry
> Classes**
> Johaimen M. Omar and Muhammed Tan

We apply cross-conformal prediction (CV+, MAPIE) to LightGBM models on
extended SOAP descriptors for the NOMAD 2018 benchmark of
(Al<sub>x</sub>Ga<sub>y</sub>In<sub>1−x−y</sub>)<sub>2</sub>O<sub>3</sub>
sesquioxides, and diagnose how crystal-symmetry heterogeneity causes marginal
coverage guarantees to conceal per-spacegroup coverage failures.

## Repository contents

| Path | Description |
|---|---|
| `uncertainty_quan.py` | Complete analysis pipeline (Sections 0–13; reproduces every table and figure in the paper) |
| `decode_soap_index.py` | Standalone utility mapping any SOAP feature index to its (species pair, n, n′, l) assignment via DScribe `get_location()` |
| `requirements.txt` | Pinned dependencies (`mapie==0.8.6`, `dscribe==2.1.2` are reproduction-critical) |
| `results/*.csv` | Pre-computed result tables cited in the paper (per-spacegroup and cross-lattice coverage with Clopper–Pearson CIs, conditional coverage, leakage defence, alpha grid, ensemble baseline, SOAP diversity, feature decodings, top-k attribution agreement) |
| `results/*.pdf` | Publication figures (600 DPI, Nature format) |
| `results/checkpoints/best_params_v5.json` | Optuna-selected LightGBM hyperparameters (Table 1) |
| `results/outlier_sg227_2202.cif` | Figure 9 structure (NOMAD ID 2202), exported for VESTA |

## Reproducing the results 

### 1. Checkpoint mode 

```bash
git clone https://github.com/johmar-22/tco-conformal-uq.git
cd tco-conformal-uq
pip install -r requirements.txt
python uncertainty_quan.py
```

On first run the script automatically downloads the pre-computed checkpoints
published as [Release v1.0.0 assets](https://github.com/johmar-22/tco-conformal-uq/releases/tag/v1.0.0)
(~550 MB total): the SOAP feature cache, prepared train/test arrays, fitted
scaler, and the two trained MAPIE CV+ models. `train.csv` is fetched from the
public [Sutton et al. repository](https://github.com/csutton7/nomad_2018_kaggle_dataset).
All evaluation sections then run deterministically and regenerate the paper's
tables and figures into `results/`.

Set `TCO_UQ_NO_DOWNLOAD=1` to forbid all checkpoint downloads.

### 2. Full re-computation mode

Delete `results/checkpoints/*.npz` and `*.pkl`, then provide the NOMAD 2018
geometry files (not redistributable here; download from
[Kaggle](https://www.kaggle.com/c/nomad2018-predict-transparent-conductors/data))
so that `data/train/<id>/geometry.xyz` exists, and run the same command.
SOAP featurization, Optuna HPO (25 trials, GroupKFold(6) grouped by
spacegroup), CV+ training, SHAP (5 seeds), and the six leave-one-spacegroup-out
retrainings are then computed from scratch. A CUDA GPU is auto-detected and
used by LightGBM when present.

### Determinism

`SEED = 42` throughout (StratifiedShuffleSplit, LightGBM, Optuna TPE);
SHAP stability uses seeds 42/123/456/789/2024. The pinned `mapie==0.8.6` and
`dscribe==2.1.2` are mandatory: MAPIE's API changed across 0.8/0.9/1.x, and
the SOAP feature-index layout cited in the paper is DScribe-version-dependent.

## System requirements

| Requirement | Minimum | Tested on |
|---|---|---|
| Python | 3.9 | 3.10 |
| RAM | 16 GB | 32 GB (Colab) |
| GPU | optional | NVIDIA T4 |
| Disk | 5 GB | 10 GB |

## Feature-index decoding

```bash
python decode_soap_index.py Max_0039 Std_3877 --blocks
# Max_0039: aggregation=Max, pair=O-O,  (n, n') = (7, 7), l = 0
# Std_3877: aggregation=Std, pair=Al-In, (n, n') = (8, 8), l = 1
```

## Data availability

The NOMAD 2018 dataset (3,000 DFT-PBE calculations) is available on
[Kaggle](https://www.kaggle.com/c/nomad2018-predict-transparent-conductors/data);
`train.csv` (2,400 public structures) is downloaded automatically. Raw
geometry files are required only for full re-computation and are not
redistributed in this repository.






