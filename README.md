# Nurdle Detection Pipeline

Simple, stage-based pipeline for nurdle count prediction using RGB HOG + LBP features and an SVR regressor.

---

## Quick Start
1. **Create/activate the venv:**
   ```sh
   python -m venv .venv
   .\.venv\Scripts\activate
   ```
2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
3. **Prepare your data:**
   - Place your input images and JSON annotations in the `input/` directory.
4. **Run the pipeline:**
   - To run the full pipeline (using `config.yaml` by default):
     ```sh
     python pipeline.py
     ```
   - To run specific steps only (e.g., normalization, features, tuning, training, evaluation):
     ```sh
     python pipeline.py --steps normalization,features,tuning,training,evaluation
     ```
   - To use a custom config file:
     ```sh
     python pipeline.py myconfig.yaml
     ```

---

## Manual: How to Use the Pipeline

### 1. **Pipeline Stages**
The pipeline is modular and can be executed in individual steps or as a full sequence. Each stage produces outputs in a dedicated folder:

- **normalization** → `output/01_normalization` (sample visuals), checkpoint at `temp/normalized`
- **features** → `output/02_features` (feature visuals)
- **tuning** → `output/03_tuning` (Optuna results, plots, per-trial JSON)
- **training** → `output/04_training` (training metrics JSON), model in `output/models`
- **evaluation** → `output/05_evaluation` (eval metrics JSON, comparison plot, per-image preds)
- **save** (optional) → zipped `output/` to configured `save_dir`

### 2. **Running Individual Steps**
You can execute only certain steps of the pipeline by specifying them with the `--steps` argument. For example, to run normalization and feature extraction only:

```sh
python pipeline.py --steps normalization,features
```

You can resume from any stage, and intermediate outputs will be reused if available.

### 3. **Saving Outputs**
To save the current pipeline outputs (e.g., for sharing or backup), use the `save` step. This will zip the `output/` directory and place the archive in the configured `save_dir` (see config):

```sh
python pipeline.py --steps save
```

You can also append `save` to any step sequence to save after running those steps:

```sh
python pipeline.py --steps training,evaluation,save
```

### 4. **Configuration**
- The default config file is `config.yaml`. To use a different config, pass its path as a positional argument:
  ```sh
  python pipeline.py myconfig.yaml
  ```
- Key config sections:
  - `data`: input paths, train/test split, resolutions
  - `features`: HOG cell size, image size
  - `svr_optimization`: Optuna trial count

---

## Notes
- Tuning plots are saved even if plotly/kaleido is blocked (matplotlib fallback).
- RGB HOG is used for training; HOG visualization is grayscale (gradient magnitude), which is expected.
- Logs are in `output/logs/`.

---

## Troubleshooting
- Ensure your input images and annotation JSONs are correctly placed in the `input/` directory.
- Check the logs in `output/logs/` for error messages and progress.
- If you encounter issues with missing dependencies, re-run `pip install -r requirements.txt` in your active virtual environment.
