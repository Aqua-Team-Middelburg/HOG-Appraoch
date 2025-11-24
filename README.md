# Nurdle Detection Pipeline

Simple, stage-based pipeline for nurdle count prediction using RGB HOG + LBP features and an SVR regressor.

## Quick Start
1) Create/activate the venv  
   `python -m venv .venv`  
   `.\.venv\Scripts\activate`
2) Install dependencies  
   `pip install -r requirements.txt`
3) Place your input images + JSON annotations in `input/`.
4) Run the pipeline (defaults to `config.yaml`)  
   `python pipeline.py`  
   Optional: limit stages, e.g. `python pipeline.py --steps normalization,features,tuning,training,evaluation`

## Stages (and outputs)
- normalization → `output/01_normalization` (sample visuals), checkpoint at `temp/normalized`
- features → `output/02_features` (feature visuals)
- tuning → `output/03_tuning` (Optuna results, plots, per-trial JSON)
- training → `output/04_training` (training metrics JSON), model in `output/models`
- evaluation → `output/05_evaluation` (eval metrics JSON, comparison plot, per-image preds)
- save (optional) → zipped `output/` to configured `save_dir`

## Config
- Default is `config.yaml` (no flag needed). Override by passing a positional path: `python pipeline.py myconfig.yaml`.
- Key sections used:  
  - `data`: input paths, train/test split, resolutions  
  - `features`: HOG cell size, image size  
  - `svr_optimization`: Optuna trial count  

## Notes
- Tuning plots are saved even if plotly/kaleido is blocked (matplotlib fallback).
- RGB HOG is used for training; HOG visualization is grayscale (gradient magnitude), which is expected.
- Logs are in `output/logs/`.
