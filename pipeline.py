import argparse
import logging
import sys
import shutil
import json
import zipfile
from pathlib import Path
from typing import Dict, Any, List, Tuple
from datetime import datetime
import numpy as np
import os
import threading
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Import pipeline components
try:
    from src.utils.config import load_config
    from src.data import DataLoader
    from src.features import FeatureExtractor
    from src.models import ModelTrainer
except ImportError as e:
    print(f"Error importing pipeline components: {e}")
    print("Make sure all src modules are properly installed")
    sys.exit(1)


class NurdlePipeline:
    """
    Modular pipeline orchestrator with checkpoint-based stage execution.
    
    Each stage produces intermediate outputs and can be run independently.
    """
    
    # Define available stages
    AVAILABLE_STAGES = ['normalization', 'features', 'tuning', 'training', 'evaluation', 'save']
    
    def __init__(self, config_path: str):
        """Initialize pipeline with configuration."""
        self.config_path = config_path
        self.config = load_config(config_path)  # Load config on init
        self.logger = self._setup_logging()
        
        # Define checkpoint and output directories
        self.temp_dir = Path('temp')
        self.checkpoint_normalized = self.temp_dir / 'normalized'
        # Stage output directories
        self.output_dirs = {
            'normalization': Path('output/01_normalization'),
            'features': Path('output/02_features'),
            'tuning': Path('output/03_tuning'),
            'training': Path('output/04_training'),
            'evaluation': Path('output/05_evaluation')
        }
        
        # Models directory
        self.models_dir = Path('output/models')
        
        # Initialize components (will be created as needed per stage)
        self.data_loader = None
        self.feature_extractor = None
        self.model_trainer = None
        self.evaluator = None
        
        self.logger.info("Modular pipeline initialized successfully")
    
    # Remove _load_checkpoints (legacy, not used in single-stage pipeline)

    def _get_feature_image_size(self) -> tuple:
        """
        Resolve the feature extraction image size from config.

        Prefers explicit features.image_size; otherwise falls back to a square
        sized to data.target_resolution so we only need one resolution setting.
        """
        feature_size = self.config.features.get('image_size')
        if feature_size:
            return tuple(feature_size)
        target_res = self.config.data.get('target_resolution', 1080)
        return (target_res, target_res)

    def _get_count_bins(self) -> List[float]:
        """Count bin edges for stratification."""
        return self.config.data.get('count_bins', [0, 10, 30, 10**9])

    def _bin_labels(self, counts: np.ndarray) -> np.ndarray:
        edges = self._get_count_bins()
        return np.digitize(counts, edges, right=True)
    
    def _get_cv_splitter(self, strata: np.ndarray):
        """Return a CV splitter; prefer stratified if class sizes allow."""
        if len(strata) < 2:
            raise ValueError(f"Need at least 2 samples for CV splitting; got {len(strata)}")
        desired_folds = self.config.data.get('cv_folds', 5)
        bincount = np.bincount(strata) if len(strata) else np.array([0])
        min_class = np.min(bincount[bincount > 0]) if np.any(bincount > 0) else 0
        if min_class >= max(2, desired_folds):
            return StratifiedKFold(n_splits=desired_folds, shuffle=True, random_state=42)
        # Fallback: non-stratified KFold with safe n_splits
        n_splits = min(max(2, desired_folds), len(strata))
        return KFold(n_splits=n_splits, shuffle=True, random_state=42)

    def _svr_bounds(self) -> Dict[str, Any]:
        """Read SVR bounds from config (optimization.svr) with safe defaults."""
        opt_cfg = getattr(self.config, 'optimization', {}) or {}
        bounds = opt_cfg.get('svr', {})
        return {
            'c_min': bounds.get('c_min', 0.1),
            'c_max': bounds.get('c_max', 10.0),
            'eps_min': bounds.get('eps_min', 0.01),
            'eps_max': bounds.get('eps_max', 0.5),
            'kernel': bounds.get('kernel', 'rbf'),
            'gamma': bounds.get('gamma', 'scale'),
            'gamma_min': bounds.get('gamma_min'),
            'gamma_max': bounds.get('gamma_max'),
            'n_trials': bounds.get('n_trials', opt_cfg.get('n_trials', 5))
        }

    def _build_features_from_indices(self, annotations: List[Any], indices: np.ndarray, feature_extractor) -> Tuple[np.ndarray, np.ndarray]:
        """Build feature matrix and target vector for a subset of annotations."""
        feats = []
        targets = []
        for idx in indices:
            ann = annotations[idx]
            image = self.data_loader.load_image(ann.image_path)
            fvec = feature_extractor.extract_image_features(image)
            feats.append(fvec)
            targets.append(ann.nurdle_count)
        return np.array(feats), np.array(targets)
    
    def _setup_logging(self):
        """Setup pipeline-specific logging."""
        logs_dir = Path('output/logs')
        logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Clean up old log files, keeping only the last 3
        self._cleanup_old_logs(logs_dir, keep_last=3)
        
        logger = logging.getLogger('NurdlePipeline')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            # Console handler
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.INFO)
            
            # File handler
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            file_handler = logging.FileHandler(logs_dir / f'pipeline_{timestamp}.log')
            file_handler.setLevel(logging.DEBUG)
            
            # Formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(formatter)
            file_handler.setFormatter(formatter)
            
            logger.addHandler(console_handler)
            logger.addHandler(file_handler)
            logger.propagate = False
        
        return logger
    
    def _cleanup_old_logs(self, logs_dir: Path, keep_last: int = 3):
        """Remove old log files, keeping only the most recent ones."""
        log_files = sorted(logs_dir.glob('pipeline_*.log'), key=lambda f: f.stat().st_mtime, reverse=True)
        
        # Remove older log files beyond keep_last
        for old_log in log_files[keep_last:]:
            try:
                old_log.unlink()
            except Exception:
                pass  # Ignore errors during cleanup
    
    def _create_stage_output_dir(self, stage: str):
        """Create and clear stage output directory."""
        output_dir = self.output_dirs[stage]
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.logger.debug(f"Created output directory: {output_dir}")
        return output_dir
    
    def _check_checkpoint_exists(self, checkpoint_path: Path, stage_name: str, prerequisite: str) -> bool:
        """Check if required checkpoint exists."""
        if not checkpoint_path.exists():
            error_msg = f"ERROR: {stage_name} - Required checkpoint not found: {checkpoint_path}. Run --steps {prerequisite} first."
            self.logger.error(error_msg)
            return False
        return True
    
    def run(self, steps: List[str]) -> Dict[str, Any]:
        """
        Run specified pipeline stages with checkpoint-based execution.

        Available stages:
        - normalization: Load and normalize images, save checkpoints
        - features: Extract and visualize features
        - training: Train SVM count classifier and save metrics
        - evaluation: Evaluate count prediction accuracy
        - save: Zip output folder

        Args:
            steps: List of stage names to execute

        Returns:
            Dictionary with results from executed stages
        """
        self.logger.info("=" * 80)
        self.logger.info(f"STARTING MODULAR PIPELINE - STEPS: {', '.join(steps)}")
        self.logger.info("=" * 80)
        
        results = {}
        
        last_stage = None
        try:
            for stage in steps:
                if stage not in self.AVAILABLE_STAGES:
                    raise ValueError(f"Unknown stage: {stage}. Available: {self.AVAILABLE_STAGES}")
                self.logger.info(f"\n{'='*80}")
                self.logger.info(f"STAGE: {stage.upper()}")
                self.logger.info(f"{'='*80}")
                # Execute stage
                if stage == 'normalization':
                    results[stage] = self._run_normalization_stage()
                elif stage == 'features':
                    results[stage] = self._run_features_stage()
                elif stage == 'tuning':
                    results[stage] = self._run_tuning_stage()
                elif stage == 'training':
                    results[stage] = self._run_training_stage()
                elif stage == 'evaluation':
                    results[stage] = self._run_evaluation_stage()
                elif stage == 'save':
                    results[stage] = self._run_save_stage()
                self.logger.info(f"Stage '{stage}' completed successfully")
                last_stage = stage
        except Exception as e:
            stage_name = last_stage if last_stage is not None else 'unknown'
            self.logger.error(f"Pipeline failed at stage '{stage_name}': {e}", exc_info=True)
            raise
        
        self.logger.info("=" * 80)
        self.logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        self.logger.info("=" * 80)
        
        return results
    
    def _run_normalization_stage(self) -> Dict[str, Any]:
        """
        Stage 1: Load images, normalize, and save checkpoints.
        
        Outputs:
        - temp/normalized/ - Checkpoint with normalized images and metadata
        - output/01_normalization/ - Sample visualizations
        """
        output_dir = self._create_stage_output_dir('normalization')
        
        # Initialize data loader
        input_dir = self.config.data.get('input_dir', 'input')
        self.data_loader = DataLoader(self.config.data)
        
        # Load and process data
        self.logger.info(f"Loading data from {input_dir}...")
        self.data_loader.load_annotations(input_dir)
        
        if self.data_loader.num_annotations == 0:
            raise ValueError(f"No annotations loaded from {input_dir}")
        
        # Split train/test FIRST (before normalization)
        self.data_loader.split_train_test()
        self.logger.info(f"Loaded {self.data_loader.num_annotations} images: "
                        f"{self.data_loader.num_train} train, {self.data_loader.num_test} test")
        
        # THEN normalize and save (normalizes all images, saves split info)
        self.logger.info("Normalizing and saving all images...")
        self.data_loader.save_normalized_data(self.checkpoint_normalized)
        
        # Generate sample visualizations
        self.logger.info("Generating normalization visualizations...")
        self.data_loader.visualize_normalization_samples(self.checkpoint_normalized, output_dir, num_samples=3)
        
        return {
            'total_images': self.data_loader.num_annotations,
            'train_images': self.data_loader.num_train,
            'test_images': self.data_loader.num_test,
            'checkpoint': str(self.checkpoint_normalized)
        }
    
    def _run_features_stage(self) -> Dict[str, Any]:
        """
        Stage 2: Extract and visualize features.

        Prerequisites:
        - temp/normalized/ must exist

        Outputs:
        - output/03_features/ - Feature visualizations (HOG overlays, LBP patterns)

        Note: Features are NOT saved, training uses generator.
        """
        if not self._check_checkpoint_exists(self.checkpoint_normalized, 'Features', 'normalization'):
            raise FileNotFoundError(f"ERROR: Features - Normalized data not found. Run --steps normalization first.")
        output_dir = self._create_stage_output_dir('features')
        self.logger.info("Extracting features for all images...")
        self.data_loader = DataLoader(self.config.data)
        self.data_loader.load_normalized_data(self.checkpoint_normalized)
        image_size = self._get_feature_image_size()
        feature_config = {
            'hog_cell_size': self.config.features.get('hog_cell_size', [4, 4]),
            'image_size': image_size,
            'use_hog': self.config.features.get('use_hog', True),
            'use_lbp': self.config.features.get('use_lbp', True),
        }
        self.feature_extractor = FeatureExtractor(feature_config)
        # Visualize feature samples for up to 3 test images
        self.feature_extractor.visualize_feature_samples(
            self.data_loader.test_annotations,
            self.data_loader.load_image,
            str(output_dir),
            num_samples=3
        )
        return {'visualization_dir': str(output_dir)}
    
    def _run_tuning_stage(self) -> Dict[str, Any]:
        """
        Stage 3: Hyperparameter tuning for SVR regressor.

        Prerequisites:
        - temp/normalized/ must exist

        Outputs:
        - output/03_tuning/ - Tuning results and best parameters
        """
        from src.optuna import OptunaTuner
        if not self._check_checkpoint_exists(self.checkpoint_normalized, 'Tuning', 'normalization'):
            raise FileNotFoundError(f"ERROR: Tuning - Normalized data not found. Run --steps normalization first.")
        output_dir = self._create_stage_output_dir('tuning')
        self.logger.info("Loading normalized data for tuning...")
        self.data_loader = DataLoader(self.config.data)
        self.data_loader.load_normalized_data(self.checkpoint_normalized)
        image_size = self._get_feature_image_size()
        feature_config = {
            'hog_cell_size': self.config.features.get('hog_cell_size', [4, 4]),
            'image_size': image_size,
            'use_hog': self.config.features.get('use_hog', True),
            'use_lbp': self.config.features.get('use_lbp', True),
        }
        self.feature_extractor = FeatureExtractor(feature_config)
        train_annotations = self.data_loader.train_annotations
        y_all = np.array([ann.nurdle_count for ann in train_annotations])
        strata = self._bin_labels(y_all)

        config_dict = getattr(self.config, '_config', self.config if isinstance(self.config, dict) else {})
        tuner = OptunaTuner(config_dict)

        def cv_objective(params: Dict[str, Any]) -> Dict[str, Any]:
            splitter = self._get_cv_splitter(strata)
            fold_mae = []
            bnds = self._svr_bounds()
            for train_idx, val_idx in splitter.split(np.arange(len(train_annotations)),
                                                    strata if isinstance(splitter, StratifiedKFold) else None):
                X_tr, y_tr = self._build_features_from_indices(train_annotations, train_idx, self.feature_extractor)
                X_val, y_val = self._build_features_from_indices(train_annotations, val_idx, self.feature_extractor)
                y_tr_log = np.log1p(y_tr)
                C_val = max(bnds['c_min'], min(params.get('svr_c', 1.0), bnds['c_max']))
                eps_val = max(bnds['eps_min'], min(params.get('svr_epsilon', 0.1), bnds['eps_max']))
                kernel_val = params.get('svr_kernel', bnds['kernel'])
                gamma_val = params.get('svr_gamma', bnds.get('gamma', 'scale'))
                if isinstance(gamma_val, (int, float, np.floating)):
                    if bnds.get('gamma_min') is not None:
                        gamma_val = max(bnds['gamma_min'], gamma_val)
                    if bnds.get('gamma_max') is not None:
                        gamma_val = min(bnds['gamma_max'], gamma_val)
                kernel_val = bnds['kernel']  # enforce configured kernel
                reg = make_pipeline(
                    StandardScaler(),
                    SVR(
                        C=C_val,
                        kernel=kernel_val,
                        gamma=gamma_val,
                        epsilon=eps_val
                    )
                )
                reg.fit(X_tr, y_tr_log)
                preds_log = reg.predict(X_val)
                # cap to avoid overflow in expm1
                max_log = np.log1p(np.max(y_tr)) + 2.0
                preds_log = np.clip(preds_log, a_min=-5.0, a_max=max_log)
                preds = np.expm1(preds_log)
                preds = np.clip(preds, 0, np.max(y_tr) + 50)
                preds = np.nan_to_num(preds, nan=1e6, posinf=1e6, neginf=0)
                mae_val = mean_absolute_error(y_val, preds)
                if not np.isfinite(mae_val):
                    mae_val = 1e6
                fold_mae.append(mae_val)
            # Return full metrics dict for the tuner (Optuna expects .get access)
            return {
                'mae': float(np.mean(fold_mae)),
                'fold_mae': [float(m) for m in fold_mae],
                'n_folds': len(fold_mae)
            }

        tuning_results = tuner.optimize(cv_objective, str(output_dir))
        best_params = tuning_results.get('best_params', {})

        # Calculate best params CV mean/std for reporting
        if best_params:
            splitter_final = self._get_cv_splitter(strata)
            fold_mae = []
            fold_mape = []
            bnds = self._svr_bounds()
            for train_idx, val_idx in splitter_final.split(np.arange(len(train_annotations)),
                                                           strata if isinstance(splitter_final, StratifiedKFold) else None):
                X_tr, y_tr = self._build_features_from_indices(train_annotations, train_idx, self.feature_extractor)
                X_val, y_val = self._build_features_from_indices(train_annotations, val_idx, self.feature_extractor)
                y_tr_log = np.log1p(y_tr)
                C_val = max(bnds['c_min'], min(best_params.get('svr_c', 1.0), bnds['c_max']))
                eps_val = max(bnds['eps_min'], min(best_params.get('svr_epsilon', 0.1), bnds['eps_max']))
                kernel_val = bnds['kernel']
                gamma_val = best_params.get('svr_gamma', bnds.get('gamma', 'scale'))
                if isinstance(gamma_val, (int, float, np.floating)):
                    if bnds.get('gamma_min') is not None:
                        gamma_val = max(bnds['gamma_min'], gamma_val)
                    if bnds.get('gamma_max') is not None:
                        gamma_val = min(bnds['gamma_max'], gamma_val)
                reg = make_pipeline(
                    StandardScaler(),
                    SVR(
                        C=C_val,
                        kernel=kernel_val,
                        gamma=gamma_val,
                        epsilon=eps_val
                    )
                )
                reg.fit(X_tr, y_tr_log)
                max_log = np.log1p(np.max(y_tr)) + 2.0
                preds_log = reg.predict(X_val)
                preds_log = np.clip(preds_log, a_min=-5.0, a_max=max_log)
                preds = np.expm1(preds_log)
                preds = np.clip(preds, 0, np.max(y_tr) + 50)
                preds = np.nan_to_num(preds, nan=1e6, posinf=1e6, neginf=0)
                mae_val = mean_absolute_error(y_val, preds)
                if not np.isfinite(mae_val):
                    mae_val = 1e6
                fold_mae.append(mae_val)
                mape_val = np.mean(np.abs((y_val - preds) / np.maximum(y_val, 1))) * 100
                if not np.isfinite(mape_val):
                    mape_val = 1e6
                fold_mape.append(mape_val)
            cv_summary = {
                'cv_mae_mean': float(np.mean(fold_mae)),
                'cv_mae_std': float(np.std(fold_mae)),
                'cv_mape_mean': float(np.mean(fold_mape)),
                'cv_mape_std': float(np.std(fold_mape)),
            }
            with open(output_dir / 'cv_summary.json', 'w') as f:
                json.dump(cv_summary, f, indent=2)
            with open(output_dir / 'best_params.json', 'w') as f:
                json.dump(best_params, f, indent=2)
        return {
            'best_params': best_params,
            'best_mae': tuning_results.get('best_value'),
            'n_trials': tuning_results.get('n_trials'),
            'output_dir': str(output_dir),
            'tuning_results_path': str(output_dir / 'tuning_results.json'),
        }
    
    def _run_training_stage(self) -> Dict[str, Any]:
        """
        Stage 3: Train SVR regressor and collect metrics.

        Prerequisites:
        - temp/normalized/ must exist

        Outputs:
        - output/04_training/ - Training metrics and curves
        - output/models/ - Saved models
        """
        if not self._check_checkpoint_exists(self.checkpoint_normalized, 'Training', 'normalization'):
            raise FileNotFoundError(f"ERROR: Training - Normalized data not found. Run --steps normalization first.")
        output_dir = self._create_stage_output_dir('training')
        self.logger.info("Loading normalized data for training...")
        self.data_loader = DataLoader(self.config.data)
        self.data_loader.load_normalized_data(self.checkpoint_normalized)
        image_size = self._get_feature_image_size()
        feature_config = {
            'hog_cell_size': self.config.features.get('hog_cell_size', [4, 4]),
            'image_size': image_size,
            'use_hog': self.config.features.get('use_hog', True),
            'use_lbp': self.config.features.get('use_lbp', True),
        }
        self.feature_extractor = FeatureExtractor(feature_config)
        # Load best params from tuning if available
        tuning_best_path = self.output_dirs['tuning'] / 'best_params.json'
        config_dict = self.config if isinstance(self.config, dict) else vars(self.config)
        if tuning_best_path.exists():
            with open(tuning_best_path, 'r') as f:
                best_params = json.load(f)
            config_dict = {**config_dict, **best_params}
            self.logger.info(f"Using tuned params: {best_params}")

        # Compute CV metrics with selected params
        train_annotations = self.data_loader.train_annotations
        y_all = np.array([ann.nurdle_count for ann in train_annotations])
        strata = self._bin_labels(y_all)
        splitter = self._get_cv_splitter(strata)
        bnds = self._svr_bounds()

        def _resolve_gamma(val):
            """Normalize gamma value from config/best_params to a valid SVR input."""
            if isinstance(val, (list, tuple)):
                return val[0] if len(val) > 0 else 'scale'
            if isinstance(val, (int, float, np.floating)):
                if bnds.get('gamma_min') is not None:
                    val = max(bnds['gamma_min'], val)
                if bnds.get('gamma_max') is not None:
                    val = min(bnds['gamma_max'], val)
            return val if val is not None else 'scale'
        fold_mae = []
        fold_mape = []
        for train_idx, val_idx in splitter.split(np.arange(len(train_annotations)),
                                                 strata if isinstance(splitter, StratifiedKFold) else None):
            X_tr, y_tr = self._build_features_from_indices(train_annotations, train_idx, self.feature_extractor)
            X_val, y_val = self._build_features_from_indices(train_annotations, val_idx, self.feature_extractor)
            y_tr_log = np.log1p(y_tr)
            reg = make_pipeline(
                StandardScaler(),
                SVR(
                    C=max(bnds['c_min'], min(config_dict.get('svr_c', 1.0), bnds['c_max'])),
                    kernel=bnds['kernel'],
                    gamma=_resolve_gamma(config_dict.get('svr_gamma', bnds.get('gamma', 'scale'))),
                    epsilon=max(bnds['eps_min'], min(config_dict.get('svr_epsilon', 0.1), bnds['eps_max']))
                )
            )
            reg.fit(X_tr, y_tr_log)
            preds_log = reg.predict(X_val)
            max_log = np.log1p(np.max(y_tr)) + 2.0
            preds_log = np.clip(preds_log, a_min=-5.0, a_max=max_log)
            preds = np.expm1(preds_log)
            preds = np.clip(preds, 0, np.max(y_tr) + 50)
            preds = np.nan_to_num(preds, nan=1e6, posinf=1e6, neginf=0)
            mae_val = mean_absolute_error(y_val, preds)
            if not np.isfinite(mae_val):
                mae_val = 1e6
            fold_mae.append(mae_val)
            mape_val = np.mean(np.abs((y_val - preds) / np.maximum(y_val, 1))) * 100
            if not np.isfinite(mape_val):
                mape_val = 1e6
            fold_mape.append(mape_val)
        cv_metrics = {
            'cv_mae_mean': float(np.mean(fold_mae)),
            'cv_mae_std': float(np.std(fold_mae)),
            'cv_mape_mean': float(np.mean(fold_mape)),
            'cv_mape_std': float(np.std(fold_mape)),
        }

        # Fit final model on full training set (build once)
        full_features, full_counts = self._build_features_from_indices(train_annotations, np.arange(len(train_annotations)), self.feature_extractor)
        training_data = list(zip(full_features, full_counts))
        config_dict['svr_bounds'] = self._svr_bounds()
        self.model_trainer = ModelTrainer(config_dict)
        training_metrics = self.model_trainer.train_count_regressor(training_data)
        training_metrics.update(cv_metrics)

        # Bias/variance visual: CV dispersion
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(5, 4))
            plt.boxplot(fold_mae, vert=True, patch_artist=True, labels=['CV MAE'])
            plt.ylabel('MAE')
            plt.title('CV MAE Dispersion')
            plt.tight_layout()
            plt.savefig(output_dir / 'cv_dispersion.png', dpi=150, bbox_inches='tight')
            plt.close()
        except Exception as e:
            self.logger.warning(f"Could not generate CV dispersion plot: {e}")

        # Ablation: toggle feature flags (HOG/LBP) for comparison
        ablation_results = {}
        try:
            ablation_cfgs = []
            if self.config.features.get('use_lbp', True):
                ablation_cfgs.append(('no_lbp', {**feature_config, 'use_lbp': False}))
            if self.config.features.get('use_hog', True):
                ablation_cfgs.append(('no_hog', {**feature_config, 'use_hog': False}))

            def _cv_on_arrays(features_arr, targets_arr, splitter_obj):
                fold_mae_local = []
                fold_mape_local = []
                for tr_idx, va_idx in splitter_obj.split(np.arange(len(targets_arr)),
                                                        self._bin_labels(targets_arr) if isinstance(splitter_obj, StratifiedKFold) else None):
                    X_tr, y_tr = features_arr[tr_idx], targets_arr[tr_idx]
                    X_val, y_val = features_arr[va_idx], targets_arr[va_idx]
                    y_tr_log_local = np.log1p(y_tr)
                    reg_local = make_pipeline(
                        StandardScaler(),
                        SVR(
                            C=max(bnds['c_min'], min(config_dict.get('svr_c', 1.0), bnds['c_max'])),
                            kernel=bnds['kernel'],
                            gamma=_resolve_gamma(config_dict.get('svr_gamma', bnds.get('gamma', 'scale'))),
                            epsilon=max(bnds['eps_min'], min(config_dict.get('svr_epsilon', 0.1), bnds['eps_max']))
                        )
                    )
                    reg_local.fit(X_tr, y_tr_log_local)
                    preds_log_local = reg_local.predict(X_val)
                    max_log_local = np.log1p(np.max(y_tr)) + 2.0
                    preds_log_local = np.clip(preds_log_local, a_min=-5.0, a_max=max_log_local)
                    preds_local = np.expm1(preds_log_local)
                    preds_local = np.clip(preds_local, 0, np.max(y_tr) + 50)
                    preds_local = np.nan_to_num(preds_local, nan=1e6, posinf=1e6, neginf=0)
                    mae_local = mean_absolute_error(y_val, preds_local)
                    if not np.isfinite(mae_local):
                        mae_local = 1e6
                    fold_mae_local.append(mae_local)
                    mape_local = np.mean(np.abs((y_val - preds_local) / np.maximum(y_val, 1))) * 100
                    if not np.isfinite(mape_local):
                        mape_local = 1e6
                    fold_mape_local.append(mape_local)
                return {
                    'cv_mae_mean': float(np.mean(fold_mae_local)),
                    'cv_mae_std': float(np.std(fold_mae_local)),
                    'cv_mape_mean': float(np.mean(fold_mape_local)),
                    'cv_mape_std': float(np.std(fold_mape_local)),
                    'fold_mae': [float(x) for x in fold_mae_local],
                }

            for label, cfg in ablation_cfgs:
                ab_extractor = FeatureExtractor(cfg)
                feats = []
                counts = []
                for ann in train_annotations:
                    img = self.data_loader.load_image(ann.image_path)
                    feats.append(ab_extractor.extract_image_features(img))
                    counts.append(ann.nurdle_count)
                feats = np.array(feats)
                counts = np.array(counts)
                splitter_ab = self._get_cv_splitter(self._bin_labels(counts))
                ablation_results[label] = _cv_on_arrays(feats, counts, splitter_ab)

            if ablation_results:
                with open(output_dir / 'ablation_results.json', 'w') as f:
                    json.dump(ablation_results, f, indent=2)
        except Exception as e:
            self.logger.warning(f"Ablation run failed: {e}")

        # Learning curve on fractions of training data (using best params)
        learning_curve = []
        try:
            rng = np.random.default_rng(42)
            fractions = [0.2, 0.4, 0.6, 0.8, 1.0]
            base_indices = np.arange(len(full_counts))
            rng.shuffle(base_indices)

            def _cv_mae_on_subset(idx_subset):
                subset_feats = full_features[idx_subset]
                subset_counts = full_counts[idx_subset]
                splitter_subset = self._get_cv_splitter(self._bin_labels(subset_counts))
                stats = _cv_on_arrays(subset_feats, subset_counts, splitter_subset)
                return stats

            for frac in fractions:
                n = max(2, int(len(base_indices) * frac))
                idx_subset = base_indices[:n]
                stats = _cv_mae_on_subset(idx_subset)
                learning_curve.append({
                    'fraction': frac,
                    'n_samples': int(n),
                    **stats
                })

            if learning_curve:
                try:
                    import matplotlib.pyplot as plt
                    plt.figure(figsize=(6, 4))
                    plt.errorbar(
                        [p['fraction'] for p in learning_curve],
                        [p['cv_mae_mean'] for p in learning_curve],
                        yerr=[p['cv_mae_std'] for p in learning_curve],
                        fmt='-o',
                        capsize=4
                    )
                    plt.xlabel('Training Fraction')
                    plt.ylabel('CV MAE')
                    plt.title('Learning Curve (MAE)')
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(output_dir / 'learning_curve.png', dpi=150, bbox_inches='tight')
                    plt.close()
                except Exception as e:
                    self.logger.warning(f"Could not plot learning curve: {e}")
                with open(output_dir / 'learning_curve.json', 'w') as f:
                    json.dump(learning_curve, f, indent=2)
        except Exception as e:
            self.logger.warning(f"Learning curve computation failed: {e}")

        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.model_trainer.save_model(str(self.models_dir))
        metrics_path = output_dir / 'training_metrics.json'
        # Always overwrite to ensure metrics are present
        with open(metrics_path, 'w') as f:
            json.dump(training_metrics, f, indent=2)
        return {
            'training_metrics': training_metrics,
            'models_dir': str(self.models_dir),
            'output_dir': str(output_dir)
        }
    
    def _run_evaluation_stage(self) -> Dict[str, Any]:
        """
        Stage 4: Evaluate trained SVR regressor and generate visualizations.

        Prerequisites:
        - temp/normalized/ must exist
        - output/models/ must have trained models

        Outputs:
        - output/05_evaluation/ - Evaluation metrics, report, and visualizations
        """
        if not self._check_checkpoint_exists(self.checkpoint_normalized, 'Evaluation', 'normalization'):
            raise FileNotFoundError(f"ERROR: Evaluation - Normalized data not found. Run --steps normalization first.")
        if not self.models_dir.exists() or not any(self.models_dir.glob('*.pkl')):
            raise FileNotFoundError(f"ERROR: Evaluation - No trained models found in {self.models_dir}. Run --steps training first.")
        output_dir = self._create_stage_output_dir('evaluation')
        self.logger.info("Loading normalized data for evaluation...")
        self.data_loader = DataLoader(self.config.data)
        self.data_loader.load_normalized_data(self.checkpoint_normalized)
        image_size = self._get_feature_image_size()
        feature_config = {
            'hog_cell_size': self.config.features.get('hog_cell_size', [4, 4]),
            'image_size': image_size,
            'use_hog': self.config.features.get('use_hog', True),
            'use_lbp': self.config.features.get('use_lbp', True),
        }
        feature_extractor = FeatureExtractor(feature_config)
        model_trainer = ModelTrainer(self.config if isinstance(self.config, dict) else vars(self.config))
        model_trainer.load_model(str(self.models_dir))
        test_annots = self.data_loader.test_annotations
        y_true = []
        y_pred = []
        for batch in self.data_loader.get_image_batches(test_annots):
            self.logger.debug(f"Evaluating batch of {len(batch)} images (batch_size={self.data_loader.batch_size})")
            for annot in batch:
                image = self.data_loader.load_image(annot.image_path)
                features = feature_extractor.extract_image_features(image)
                pred_count = model_trainer.predict_count(features)
                y_true.append(annot.nurdle_count)
                y_pred.append(pred_count)
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((np.array(y_true) - np.array(y_pred)) / np.maximum(np.array(y_true), 1))) * 100
        eval_metrics = {
            'mae': float(mae),
            'rmse': float(rmse),
            'mape': float(mape),
            'n_test': len(y_true)
        }
        with open(output_dir / 'evaluation_metrics.json', 'w') as f:
            json.dump(eval_metrics, f, indent=2)
        import matplotlib.pyplot as plt
        training_output_dir = self.output_dirs['training']
        training_metrics_path = training_output_dir / 'training_metrics.json'
        if training_metrics_path.exists():
            with open(training_metrics_path, 'r') as f:
                training_metrics = json.load(f)
        else:
            training_metrics = None
            self.logger.warning(f"Training metrics not found at {training_metrics_path}; skipping train/test comparison plot")
        if training_metrics:
            metrics_names = ['MAE', 'RMSE', 'MAPE']
            train_vals = [
                training_metrics.get('mae', 0),
                training_metrics.get('rmse', 0),
                training_metrics.get('mape', 0)
            ]
            test_vals = [eval_metrics['mae'], eval_metrics['rmse'], eval_metrics['mape']]
            x = np.arange(len(metrics_names))
            plt.figure(figsize=(7, 5))
            plt.bar(x - 0.2, train_vals, width=0.4, label='Train')
            plt.bar(x + 0.2, test_vals, width=0.4, label='Test')
            plt.xticks(x, metrics_names)
            plt.ylabel('Metric Value')
            plt.title('Training vs Testing Metrics')
            plt.legend()
            plt.tight_layout()
            plt.savefig(output_dir / 'train_vs_test_metrics.png')
            plt.close()
            # Baseline vs model
            train_counts = [ann.nurdle_count for ann in self.data_loader.train_annotations]
            baseline_pred = float(np.mean(train_counts)) if train_counts else 0.0
            baseline_mae = mean_absolute_error(y_true, [baseline_pred] * len(y_true))
            baseline_mape = np.mean(np.abs((np.array(y_true) - baseline_pred) / np.maximum(np.array(y_true), 1))) * 100
            baseline_metrics = {
                'baseline_pred': baseline_pred,
                'baseline_mae': float(baseline_mae),
                'baseline_mape': float(baseline_mape)
            }
            with open(output_dir / 'baseline_metrics.json', 'w') as f:
                json.dump(baseline_metrics, f, indent=2)
            plt.figure(figsize=(6, 4))
            plt.bar(['Baseline MAE', 'SVR MAE'], [baseline_mae, eval_metrics['mae']], color=['gray', 'steelblue'])
            plt.title('Baseline vs SVR (MAE)')
            plt.tight_layout()
            plt.savefig(output_dir / 'baseline_vs_svr.png')
            plt.close()
            # CV summary if available
            if all(k in training_metrics for k in ['cv_mae_mean', 'cv_mae_std']):
                cv_vals = [training_metrics['cv_mae_mean'], training_metrics['cv_mape_mean']]
                cv_std = [training_metrics.get('cv_mae_std', 0), training_metrics.get('cv_mape_std', 0)]
                plt.figure(figsize=(6, 4))
                plt.bar(['CV MAE', 'CV MAPE'], cv_vals, yerr=cv_std, color='seagreen', alpha=0.8)
                plt.title('Cross-Validation Metrics (mean ± std)')
                plt.tight_layout()
                plt.savefig(output_dir / 'cv_metrics.png')
                plt.close()
        # Residuals plot
        residuals = np.array(y_true) - np.array(y_pred)
        plt.figure(figsize=(6, 4))
        plt.hist(residuals, bins=15, color='slateblue', alpha=0.8)
        plt.title('Residuals (Actual - Predicted)')
        plt.xlabel('Residual')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(output_dir / 'residuals_hist.png')
        plt.close()
        # Individual test image visualizations
        for ann, pred_count in zip(test_annots, y_pred):
            img = self.data_loader.load_image(ann.image_path)
            actual_count = ann.nurdle_count
            plt.figure(figsize=(6, 6))
            plt.imshow(img, cmap='gray')
            plt.title(f"Actual: {actual_count} | Predicted: {pred_count:.1f}")
            plt.axis('off')
            out_path = output_dir / f"{ann.image_id}_eval.png"
            plt.savefig(out_path, bbox_inches='tight')
            plt.close()
        return {
            'evaluation_metrics': eval_metrics,
            'visualization_dir': str(output_dir)
        }
    
    # Remove _generate_detailed_report (legacy, not used in single-stage pipeline)
    
    def clean_temp(self):
        """Clean temporary checkpoint files."""
        self.logger.info("Cleaning temporary files...")
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            self.logger.info(f"Removed {self.temp_dir}")
        self.temp_dir.mkdir(parents=True, exist_ok=True)
    
    def _run_save_stage(self) -> Dict[str, Any]:
        """Stage 7: Save a zipped copy of the output folder to configured save directory.
        
        Prerequisites:
        - output/ directory must exist with pipeline results
        
        Outputs:
        - Zipped copy of output folder in save_dir
        
        Returns:
            Dictionary with zip file path
        """
        save_dir = self.config.data.get('save_dir')
        
        if not save_dir:
            raise ValueError("ERROR: Save - No save_dir configured in config.yaml. Set data.save_dir to enable saving.")
        
        # Check if output directory exists
        output_dir = Path('output')
        if not output_dir.exists():
            raise FileNotFoundError(f"ERROR: Save - Output directory not found: {output_dir}. Run pipeline stages first.")
        
        # Create save directory if it doesn't exist
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Create zip filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        zip_filename = f"output_{timestamp}.zip"
        zip_path = save_path / zip_filename
        
        self.logger.info(f"Creating zip archive of output folder...")
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Walk through output directory and add all files
            file_count = 0
            for file_path in output_dir.rglob('*'):
                if file_path.is_file():
                    # Calculate relative path for archive
                    arcname = file_path.relative_to(output_dir.parent)
                    zipf.write(file_path, arcname)
                    self.logger.debug(f"Added to zip: {arcname}")
                    file_count += 1
        
        zip_size_mb = zip_path.stat().st_size / (1024*1024)
        self.logger.info(f"Output folder saved to: {zip_path}")
        self.logger.info(f"Archived {file_count} files, zip size: {zip_size_mb:.2f} MB")
        
        return {
            'zip_path': str(zip_path),
            'zip_size_mb': round(zip_size_mb, 2),
            'files_archived': file_count
        }


def setup_logging():
    """Setup basic logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def _shutdown_plot_backends():
    """Close any plotting backends (matplotlib/plotly-kaleido) to avoid hanging threads."""
    try:
        import matplotlib.pyplot as plt
        plt.close('all')
    except Exception:
        pass

    try:
        import plotly.io as pio
        # Kaleido keeps a worker process alive; try to shut it down if present
        scope = getattr(getattr(pio, "kaleido", None), "scope", None)
        if scope and hasattr(scope, "_shutdown_kaleido"):
            try:
                scope._shutdown_kaleido()
            except Exception:
                pass
    except Exception:
        pass


def _wait_for_background_threads(logger, timeout: float = 2.0) -> None:
    """
    Wait briefly for non-daemon threads to finish; log any lingerers.

    If stubborn threads remain, forcefully terminate the process so CLI prompts return.
    """
    alive = [
        t for t in threading.enumerate()
        if t is not threading.current_thread() and t is not threading.main_thread() and t.is_alive()
    ]
    if not alive:
        return

    logger.info("Waiting for background threads to finish...")
    for t in alive:
        t.join(timeout=timeout)

    remaining = [
        t for t in threading.enumerate()
        if t is not threading.current_thread() and t is not threading.main_thread() and t.is_alive()
    ]
    if remaining:
        names = ", ".join(f"{t.name}" for t in remaining)
        logger.warning(f"Forcing shutdown with lingering threads: {names}")
        logging.shutdown()
        os._exit(0)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Modular Nurdle Detection Pipeline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        'config',
        nargs='?',
        default='config.yaml',
        help='Path to configuration YAML file (default: config.yaml)'
    )
    
    parser.add_argument(
        '--steps',
        type=str,
        default=None,
        help='Comma-separated list of stages to run (default: all stages except save). Options: normalization,features,tuning,training,evaluation,save'
    )
    
    parser.add_argument(
        '--clean-temp',
        action='store_true',
        help='Clean temporary checkpoint files'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()


def main():
    """Main entry point for the pipeline."""
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    setup_logging()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Initialize pipeline
        pipeline = NurdlePipeline(args.config)
        
        # Clean temp if requested
        if args.clean_temp:
            pipeline.clean_temp()
            print("Temporary files cleaned.")
            return 0
        
        # Parse and validate steps
        if not args.steps:
            # Default to all stages except save
            steps = [s for s in NurdlePipeline.AVAILABLE_STAGES if s not in ['save']]
            print(f"No steps specified, running default stages: {', '.join(steps)}")
        else:
            steps = [s.strip() for s in args.steps.split(',')]
        
        # Validate steps
        invalid_steps = [s for s in steps if s not in NurdlePipeline.AVAILABLE_STAGES]
        if invalid_steps:
            print(f"ERROR: Invalid steps: {invalid_steps}")
            print(f"Available steps: {NurdlePipeline.AVAILABLE_STAGES}")
            return 1
        
        # Run pipeline
        results = pipeline.run(steps)
        
        print("=" * 80)
        print("PIPELINE EXECUTION SUMMARY")
        print("=" * 80)
        for stage, result in results.items():
            print(f"\n{stage.upper()}:")
            for key, value in result.items():
                print(f"  {key}: {value}")
        print("=" * 80)
        
        return 0
        
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user")
        return 1
        
    except Exception as e:
        print(f"\nPipeline failed: {e}")
        logging.exception("Full error trace:")
        return 1
    finally:
        # Ensure plotting and background workers are shut down so the process can exit cleanly
        _shutdown_plot_backends()
        _wait_for_background_threads(logging.getLogger('NurdlePipeline'))
        logging.shutdown()


if __name__ == "__main__":
    sys.exit(main())
