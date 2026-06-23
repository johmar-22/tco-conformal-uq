# Conformal Uncertainty Quantification for TCO Property Prediction
 

 
**Paper:**  
**Author:** Johaimen M Omar, Muhammed Tan  
**Journal:** 
 
## Overview
 
This repository contains the complete code, pre-computed checkpoints, and
result tables for the study. We apply conformal prediction (CV+) with
LightGBM and SOAP structural descriptors to the NOMAD 2018 benchmark of
(Al_x Ga_y In_{1-x-y})_2O_3 sesquioxides, diagnosing how crystal symmetry
heterogeneity causes marginal coverage guarantees to mask per-spacegroup
coverage failures.
 
## System Requirements
 
| Requirement | Minimum | Tested on |
|---|---|---|
| Python | 3.9 | 3.10 |
| RAM | 16 GB | 32 GB |
| GPU | Optional | NVIDIA A100 |
| Disk | 5 GB | 10 GB |
 
Estimated runtime: ~4 hours on CPU (8 cores), ~45 min with a CUDA GPU.
 
## Installation
 
```bash
git clone https://github.com/YOUR_USERNAME/tco-conformal-uq.git
cd tco-conformal-uq
pip install -r requirements.txt
```
 
## Dataset
 
`train.csv` is downloaded automatically on first run.
 
For SOAP computation and outlier visualisation, download the NOMAD 2018
XYZ geometry files from Kaggle:
https://www.kaggle.com/c/nomad2018-predict-transparent-conductors/data
 
Extract so that `data/train/<id>/geometry.xyz` exists.
 
## Download Pre-computed Checkpoints (Recommended)
 
To skip SOAP computation (~20 min CPU) and model training (~3 hrs CPU):
 
```bash
mkdir -p results/checkpoints
# Download from GitHub Releases:
wget https://github.com/johmar-22/tco-conformal-uq/releases/download/v1.0.0/soap_features_v5.npz -P results/checkpoints/
wget https://github.com/johmar-22/tco-conformal-uq/releases/download/v1.0.0/mapie_form_v5.pkl -P results/checkpoints/
wget https://github.com/johmar-22/tco-conformal-uq/releases/download/v1.0.0/mapie_band_v5.pkl -P results/checkpoints/
```
 
## Usage
 
```bash
python uncertainty_quan.py
```
 
The script is fully resumable. Re-run after any interruption.
 
## UNDER REVIEW
 

