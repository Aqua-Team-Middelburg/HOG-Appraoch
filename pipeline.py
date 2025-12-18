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
import cv2
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
    from src.features import FeatureExtractor, build_nurdle_mask
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
        """Build feature matrix and target vector for a subset of annotations.
        
        Extracts HOG+LBP features from raw images and concatenates with segmentation mask statistics.
        This approach passes segmentation information in parallel with features to the model.
        """
        feats = []
        targets = []
        seg_config = self.config.features.get('segmentation', {})
        for idx in indices:
            ann = annotations[idx]
            image = self.data_loader.load_image(ann.image_path)
            # Generate segmentation mask (used for statistics, not to mask features)
            mask = build_nurdle_mask(
                image,
                min_dist=seg_config.get('min_dist', 8),
                min_area=seg_config.get('min_area', 50),
                max_area=seg_config.get('max_area', 2000)
            )
            # Extract HOG+LBP from raw image with mask statistics concatenated
            fvec = feature_extractor.extract_features_with_mask_stats(image, mask)
            feats.append(fvec)
            targets.append(ann.nurdle_count)
        return np.array(feats), np.array(targets)
    
    def _build_ensemble_features_from_indices(
        self,
        annotations: List[Any],
        indices: np.ndarray,
        feature_extractor,
        feature_types: List[str] = None
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """
        Build ensemble features for specified types only.
        
        Args:
            annotations: List of ImageAnnotation objects
            indices: Array of indices to extract
            feature_extractor: FeatureExtractor instance
            feature_types: List of feature types to extract ('hog', 'lbp', 'mask_stats')
                          If None, extracts all types
        
        Returns:
            Tuple of (feature_dict, targets_array)
        """
        if feature_types is None:
            feature_types = ['hog', 'lbp', 'mask_stats']
        
        hog_features = [] if 'hog' in feature_types else None
        lbp_features = [] if 'lbp' in feature_types else None
        mask_stats_features = [] if 'mask_stats' in feature_types else None
        targets = []
        
        seg_config = self.config.features.get('segmentation', {})
        for idx in indices:
            ann = annotations[idx]
            image = self.data_loader.load_image(ann.image_path)
            
            # Only generate mask if needed
            if 'mask_stats' in feature_types:
                mask = build_nurdle_mask(
                    image,
                    min_dist=seg_config.get('min_dist', 8),
                    min_area=seg_config.get('min_area', 50),
                    max_area=seg_config.get('max_area', 2000)
                )
            
            # Extract only requested feature types
            if 'hog' in feature_types:
                hog_feat = feature_extractor.extract_hog_features(image)
                hog_features.append(hog_feat)
            
            if 'lbp' in feature_types:
                lbp_feat = feature_extractor.extract_lbp_features(image)
                lbp_features.append(lbp_feat)
            
            if 'mask_stats' in feature_types:
                mask_stat_feat = feature_extractor.extract_mask_stats_features(mask)
                mask_stats_features.append(mask_stat_feat)
            
            targets.append(ann.nurdle_count)
        
        feature_dict = {}
        if hog_features:
            feature_dict['hog'] = np.array(hog_features)
        if lbp_features:
            feature_dict['lbp'] = np.array(lbp_features)
        if mask_stats_features:
            feature_dict['mask_stats'] = np.array(mask_stats_features)
        
        return feature_dict, np.array(targets)
    
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

        # Count distribution overview
        self.data_loader.save_count_histogram(output_dir, filename='nurdle_count_hist.png')
        
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
        - output/02_features/ - Feature visualizations (HOG, LBP, mask stats)

        Generates three sets of visualizations per sample image:
        - HOG features vs normalized image
        - LBP features vs normalized image
        - Segmentation mask with statistics vs normalized image
        
        Note: Features are NOT saved, training uses generator.
        """
        if not self._check_checkpoint_exists(self.checkpoint_normalized, 'Features', 'normalization'):
            raise FileNotFoundError(f"ERROR: Features - Normalized data not found. Run --steps normalization first.")
        output_dir = self._create_stage_output_dir('features')
        self.logger.info("Extracting and visualizing features for sample images...")
        
        self.data_loader = DataLoader(self.config.data)
        self.data_loader.load_normalized_data(self.checkpoint_normalized)
        image_size = self._get_feature_image_size()
        feature_config = {
            'hog_cell_size': self.config.features.get('hog_cell_size', [4, 4]),
            'image_size': image_size,
        }
        self.feature_extractor = FeatureExtractor(feature_config)
        
        # Define segmentation function for mask visualization
        seg_params = {
            'min_dist': self.config.features.get('min_dist', 8),
            'min_area': self.config.features.get('min_area', 30),
            'max_area': self.config.features.get('max_area', 8000),
        }
        
        def segment_image(img):
            return build_nurdle_mask(
                img,
                min_dist=seg_params['min_dist'],
                min_area=seg_params['min_area'],
                max_area=seg_params['max_area']
            )
        
        # Visualize HOG features for sample test images
        self.logger.info("Generating HOG feature visualizations...")
        self.feature_extractor.visualize_hog_features(
            self.data_loader.test_annotations,
            self.data_loader.load_image,
            str(output_dir),
            num_samples=1
        )
        
        # Visualize LBP features for sample test images
        self.logger.info("Generating LBP feature visualizations...")
        self.feature_extractor.visualize_lbp_features(
            self.data_loader.test_annotations,
            self.data_loader.load_image,
            str(output_dir),
            num_samples=1
        )
        
        # Visualize mask statistics for sample test images
        self.logger.info("Generating mask statistics visualizations...")
        self.feature_extractor.visualize_mask_stats_features(
            self.data_loader.test_annotations,
            self.data_loader.load_image,
            segment_image,
            str(output_dir),
            num_samples=1
        )
        
        self.logger.info(f"Feature visualizations saved to {output_dir}")
        return {'visualization_dir': str(output_dir)}
    
    def _run_ensemble_tuning_stage(self, output_dir: Path) -> Dict[str, Any]:
        """
        Hyperparameter tuning for ensemble models (per-model tuning).
        
        Tunes each base model (HOG, LBP, Mask Stats) independently using CV.
        Saves separate best_params files for each enabled model.
        Only extracts features needed for each model to optimize tuning speed.
        """
        from src.optuna import OptunaTuner
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVR
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        
        self.logger.info("Loading normalized data for per-model tuning...")
        self.data_loader = DataLoader(self.config.data)
        self.data_loader.load_normalized_data(self.checkpoint_normalized)
        image_size = self._get_feature_image_size()
        feature_config = {
            'hog_cell_size': self.config.features.get('hog_cell_size', [4, 4]),
            'image_size': image_size,
        }
        self.feature_extractor = FeatureExtractor(feature_config)
        train_annotations = self.data_loader.train_annotations
        y_all = np.array([ann.nurdle_count for ann in train_annotations])
        strata = self._bin_labels(y_all)
        
        ensemble_cfg = self.config.get('ensemble', {})
        all_best_params = {}
        
        # Tune each enabled model separately
        for model_name, use_model in [('hog', ensemble_cfg.get('use_hog', False)), 
                                       ('lbp', ensemble_cfg.get('use_lbp', True)),
                                       ('mask_stats', ensemble_cfg.get('use_mask_stats', True))]:
            if not use_model:
                self.logger.info(f"Skipping tuning for {model_name.upper()} (disabled)")
                continue
            
            self.logger.info(f"Tuning {model_name.upper()} model...")
            
            config_dict = getattr(self.config, '_config', self.config if isinstance(self.config, dict) else {})
            tuner = OptunaTuner(config_dict)
            
            def cv_objective(params: Dict[str, Any]) -> Dict[str, Any]:
                splitter = self._get_cv_splitter(strata)
                fold_mae = []
                fold_mape = []
                bnds = self._svr_bounds()
                denom_floor = 1.0
                
                for train_idx, val_idx in splitter.split(y_all, strata):
                    # Only extract the feature type needed for this model
                    feature_dict_tr, y_tr = self._build_ensemble_features_from_indices(
                        train_annotations, train_idx, self.feature_extractor, 
                        feature_types=[model_name]
                    )
                    feature_dict_val, y_val = self._build_ensemble_features_from_indices(
                        train_annotations, val_idx, self.feature_extractor,
                        feature_types=[model_name]
                    )
                    
                    X_tr = feature_dict_tr.get(model_name, np.array([]))
                    X_val = feature_dict_val.get(model_name, np.array([]))
                    
                    if X_tr.shape[0] == 0 or X_val.shape[0] == 0:
                        continue
                    
                    y_tr_log = np.log1p(y_tr)
                    C_val = max(bnds['c_min'], min(params.get('svr_c', 1.0), bnds['c_max']))
                    eps_val = max(bnds['eps_min'], min(params.get('svr_epsilon', 0.1), bnds['eps_max']))
                    kernel_val = bnds['kernel']
                    gamma_val = params.get('svr_gamma', bnds.get('gamma', 'scale'))
                    
                    reg = make_pipeline(
                        StandardScaler(),
                        SVR(C=C_val, kernel=kernel_val, gamma=gamma_val, epsilon=eps_val)
                    )
                    reg.fit(X_tr, y_tr_log)
                    
                    preds_log = reg.predict(X_val)
                    max_log = np.log1p(np.max(y_tr)) + 2.0
                    preds_log = np.clip(preds_log, a_min=-5.0, a_max=max_log)
                    preds = np.expm1(preds_log)
                    preds = np.clip(preds, 0, np.max(y_tr) + 50)
                    preds = np.nan_to_num(preds, nan=1e6, posinf=1e6, neginf=0)
                    
                    mae_val = mean_absolute_error(y_val, preds)
                    mape_val = np.mean(np.abs((y_val - preds) / np.maximum(y_val, denom_floor))) * 100
                    
                    if not np.isfinite(mae_val):
                        mae_val = 1e6
                    if not np.isfinite(mape_val):
                        mape_val = 1e6
                    
                    fold_mae.append(mae_val)
                    fold_mape.append(mape_val)
                
                return {
                    'mape': float(np.mean(fold_mape)) if fold_mape else 1e6,
                    'mae': float(np.mean(fold_mae)) if fold_mae else 1e6,
                    'n_folds': len(fold_mae)
                }
            
            model_tuning_dir = output_dir / f"{model_name}_tuning"
            model_tuning_dir.mkdir(parents=True, exist_ok=True)
            tuning_results = tuner.optimize(cv_objective, str(model_tuning_dir))
            best_params = tuning_results.get('best_params', {})
            all_best_params[model_name] = best_params
            
            if best_params:
                self.logger.info(f"{model_name.upper()} best params: {best_params}")
        
        # Save all best params to a single file with model keys
        combined_params_path = output_dir / 'best_params_ensemble.json'
        with open(combined_params_path, 'w') as f:
            json.dump(all_best_params, f, indent=2)
        self.logger.info(f"Saved per-model tuning results to {combined_params_path}")
        
        # Save all best params to a single file with model keys
        combined_params_path = output_dir / 'best_params_ensemble.json'
        with open(combined_params_path, 'w') as f:
            json.dump(all_best_params, f, indent=2)
        self.logger.info(f"Saved per-model tuning results to {combined_params_path}")
        
        # Also save a combined params file for backward compatibility (uses first enabled model)
        if all_best_params:
            first_model_params = next(iter(all_best_params.values()))
            compat_path = output_dir / 'best_params.json'
            with open(compat_path, 'w') as f:
                json.dump(first_model_params, f, indent=2)
        
        return {
            'best_params': all_best_params,
            'best_params_file': str(combined_params_path),
            'output_dir': str(output_dir)
        }
    
    def _run_tuning_stage(self) -> Dict[str, Any]:
        """
        Stage 3: Hyperparameter tuning for SVR regressor.

        Prerequisites:
        - temp/normalized/ must exist

        Outputs:
        - output/03_tuning/ - Tuning results and best parameters
        
        NOTE: For ensemble mode, this tunes a single combined feature set.
        For per-model tuning in ensemble, use _run_ensemble_tuning_stage() instead.
        """
        from src.optuna import OptunaTuner
        if not self._check_checkpoint_exists(self.checkpoint_normalized, 'Tuning', 'normalization'):
            raise FileNotFoundError(f"ERROR: Tuning - Normalized data not found. Run --steps normalization first.")
        output_dir = self._create_stage_output_dir('tuning')
        
        # Check if ensemble mode - if so, optionally use per-model tuning
        ensemble_cfg = self.config.get('ensemble', {})
        if ensemble_cfg.get('enabled', False):
            self.logger.info("Ensemble mode detected - using per-model hyperparameter tuning")
            return self._run_ensemble_tuning_stage(output_dir)
        
        self.logger.info("Loading normalized data for tuning...")
        self.data_loader = DataLoader(self.config.data)
        self.data_loader.load_normalized_data(self.checkpoint_normalized)
        image_size = self._get_feature_image_size()
        feature_config = {
            'hog_cell_size': self.config.features.get('hog_cell_size', [4, 4]),
            'image_size': image_size,
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
            fold_mape = []
            bnds = self._svr_bounds()
            denom_floor = 1.0
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
                mape_val = np.mean(np.abs((y_val - preds) / np.maximum(y_val, denom_floor))) * 100
                if not np.isfinite(mae_val):
                    mae_val = 1e6
                if not np.isfinite(mape_val):
                    mape_val = 1e6
                fold_mae.append(mae_val)
                fold_mape.append(mape_val)
            # Return full metrics dict for the tuner (Optuna expects .get access)
            return {
                'mape': float(np.mean(fold_mape)),
                'fold_mape': [float(m) for m in fold_mape],
                'mae': float(np.mean(fold_mae)),
                'fold_mae': [float(m) for m in fold_mae],
                'n_folds': len(fold_mae)
            }

        tuning_results = tuner.optimize(cv_objective, str(output_dir))
        best_params = tuning_results.get('best_params', {})
        best_params_path = output_dir / 'best_params.json'
        if best_params:
            with open(best_params_path, 'w') as f:
                json.dump(best_params, f, indent=2)
            self.logger.info(f"Optuna best params saved to {best_params_path}. Running CV summary on tuned params (this can take several minutes)...")
        else:
            self.logger.warning("Optuna returned no best_params; skipping CV summary.")

        # Calculate best params CV mean/std for reporting
        if best_params:
            splitter_final = self._get_cv_splitter(strata)
            fold_mae = []
            fold_mape = []
            bnds = self._svr_bounds()
            denom_floor = 1.0
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
                mape_val = np.mean(np.abs((y_val - preds) / np.maximum(y_val, denom_floor))) * 100
                if not np.isfinite(mape_val):
                    mape_val = 1e6
                fold_mape.append(mape_val)
            cv_summary = {
                'primary_metric': 'mape',
                'cv_mae_mean': float(np.mean(fold_mae)),
                'cv_mae_std': float(np.std(fold_mae)),
                'cv_mape_mean': float(np.mean(fold_mape)),
                'cv_mape_std': float(np.std(fold_mape)),
            }
            with open(output_dir / 'cv_summary.json', 'w') as f:
                json.dump(cv_summary, f, indent=2)
        return {
            'best_params': best_params,
            'best_mape': tuning_results.get('best_value'),
            'best_value': tuning_results.get('best_value'),
            'n_trials': tuning_results.get('n_trials'),
            'output_dir': str(output_dir),
            'tuning_results_path': str(output_dir / 'tuning_results.json'),
        }
    
    def _run_training_stage(self) -> Dict[str, Any]:
        """
        Stage 4: Train ensemble SVR models and meta-learner, collect metrics.

        Prerequisites:
        - temp/normalized/ must exist

        Outputs:
        - output/04_training/ - Training metrics and curves
        - output/models/ - Saved ensemble models
        
        Ensemble mode:
        - Trains separate SVR models for HOG, LBP, and mask statistics (if enabled)
        - Trains Ridge regression meta-learner to combine base model predictions
        - Saves all individual models and meta-learner
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
        
        # Setup ensemble configuration
        config_dict['svr_bounds'] = self._svr_bounds()
        ensemble_cfg = self.config.get('ensemble', {})
        ensemble_enabled = ensemble_cfg.get('enabled', True)
        
        self.logger.info("=" * 60)
        self.logger.info(f"ENSEMBLE MODE: {ensemble_enabled}")
        if ensemble_enabled:
            self.logger.info(f"  HOG model: {'ENABLED' if ensemble_cfg.get('use_hog', False) else 'DISABLED'}")
            self.logger.info(f"  LBP model: {'ENABLED' if ensemble_cfg.get('use_lbp', True) else 'DISABLED'}")
            self.logger.info(f"  Mask Stats model: {'ENABLED' if ensemble_cfg.get('use_mask_stats', True) else 'DISABLED'}")
        self.logger.info("=" * 60)

        # Compute CV metrics with selected params
        train_annotations = self.data_loader.train_annotations
        y_all = np.array([ann.nurdle_count for ann in train_annotations])
        strata = self._bin_labels(y_all)
        splitter = self._get_cv_splitter(strata)
        
        # ====== ENSEMBLE TRAINING WITH CROSS-VALIDATION ======
        # Use cross-validation to generate out-of-fold predictions for meta-learner
        # This eliminates data leakage
        self.logger.info("Building full ensemble features...")
        full_ensemble_features, full_counts = self._build_ensemble_features_from_indices(
            train_annotations, np.arange(len(train_annotations)), self.feature_extractor
        )
        
        # Initialize ModelTrainer with ensemble configuration
        config_dict['ensemble'] = ensemble_cfg
        self.model_trainer = ModelTrainer(config_dict)
        
        # Generate out-of-fold predictions for meta-learner training
        self.logger.info("Generating out-of-fold predictions for meta-learner (CV)...")
        oof_predictions = {name: np.zeros(len(y_all)) for name in ['hog', 'lbp', 'mask_stats']}
        fold_idx = 0
        base_model_metrics_list = []
        
        for train_idx, val_idx in splitter.split(y_all, strata):
            fold_idx += 1
            cv_folds = self.config.data.get('cv_folds', 5) if isinstance(self.config.data, dict) else self.config.data.cv_folds
            self.logger.info(f"  Fold {fold_idx}/{cv_folds}...")
            
            # Build features for this fold
            fold_train_features, fold_train_counts = self._build_ensemble_features_from_indices(
                train_annotations, train_idx, self.feature_extractor
            )
            fold_val_features, fold_val_counts = self._build_ensemble_features_from_indices(
                train_annotations, val_idx, self.feature_extractor
            )
            
            # Train base models on training fold
            fold_trainer = ModelTrainer(config_dict)
            fold_metrics = fold_trainer.train_ensemble_models(fold_train_features, fold_train_counts)
            base_model_metrics_list.append(fold_metrics)
            
            # Get out-of-fold predictions
            if fold_trainer.use_hog and 'hog' in fold_val_features:
                X_hog = fold_val_features['hog']
                preds_hog_log = fold_trainer.svr_hog.predict(X_hog)
                max_log = np.log1p(np.max(fold_train_counts)) + 2.0
                preds_hog_log = np.clip(preds_hog_log, a_min=-5.0, a_max=max_log)
                preds_hog = np.expm1(preds_hog_log)
                oof_predictions['hog'][val_idx] = np.clip(preds_hog, 0, None)
            
            if fold_trainer.use_lbp and 'lbp' in fold_val_features:
                X_lbp = fold_val_features['lbp']
                preds_lbp_log = fold_trainer.svr_lbp.predict(X_lbp)
                max_log = np.log1p(np.max(fold_train_counts)) + 2.0
                preds_lbp_log = np.clip(preds_lbp_log, a_min=-5.0, a_max=max_log)
                preds_lbp = np.expm1(preds_lbp_log)
                oof_predictions['lbp'][val_idx] = np.clip(preds_lbp, 0, None)
            
            if fold_trainer.use_mask_stats and 'mask_stats' in fold_val_features:
                X_mask = fold_val_features['mask_stats']
                preds_mask_log = fold_trainer.svr_mask_stats.predict(X_mask)
                max_log = np.log1p(np.max(fold_train_counts)) + 2.0
                preds_mask_log = np.clip(preds_mask_log, a_min=-5.0, a_max=max_log)
                preds_mask = np.expm1(preds_mask_log)
                oof_predictions['mask_stats'][val_idx] = np.clip(preds_mask, 0, None)
        
        # Aggregate metrics from CV folds
        base_model_metrics = {}
        for model_name in ['hog', 'lbp', 'mask_stats']:
            fold_metrics_list = [fold[model_name] for fold in base_model_metrics_list if model_name in fold]
            if fold_metrics_list:
                avg_mae = np.mean([m.get('mae', 0) for m in fold_metrics_list])
                avg_mape = np.mean([m.get('mape', 0) for m in fold_metrics_list])
                base_model_metrics[model_name] = {'mae': avg_mae, 'mape': avg_mape}
        
        self.logger.info("CV fold base model metrics (training folds only):")
        for model_name, metrics in base_model_metrics.items():
            self.logger.info(f"  {model_name.upper()} - MAE: {metrics.get('mae', 0):.4f}, MAPE: {metrics.get('mape', 0):.2f}%")
        
        # Build base_predictions dict for meta-learner training
        base_predictions = {}
        if self.model_trainer.use_hog:
            base_predictions['hog'] = oof_predictions['hog']
        if self.model_trainer.use_lbp:
            base_predictions['lbp'] = oof_predictions['lbp']
        if self.model_trainer.use_mask_stats:
            base_predictions['mask_stats'] = oof_predictions['mask_stats']
        
        # Train ensemble meta-learner on out-of-fold predictions
        self.logger.info("Training ensemble meta-learner on out-of-fold predictions...")
        meta_metrics = self.model_trainer.train_ensemble_meta_learner(base_predictions, full_counts)
        
        # Now train final base models on full training set for inference
        self.logger.info("Training final base models on full training set...")
        self.model_trainer.train_ensemble_models(full_ensemble_features, full_counts)
        
        # Collect all training metrics
        training_metrics = {
            'ensemble_enabled': ensemble_enabled,
            'base_models': base_model_metrics,
            'ensemble_metrics': meta_metrics,
        }
        
        # Save all models
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.model_trainer.save_ensemble_models(str(self.models_dir))
        
        # Save training metrics
        metrics_path = output_dir / 'training_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(training_metrics, f, indent=2)
        
        self.logger.info("Training stage completed successfully")
        return {
            'training_metrics': training_metrics,
            'models_dir': str(self.models_dir),
            'output_dir': str(output_dir)
        }
    
    def _run_evaluation_stage(self) -> Dict[str, Any]:
        """
        Stage 5: Evaluate trained ensemble SVR models and generate visualizations.

        Prerequisites:
        - temp/normalized/ must exist
        - output/models/ must have trained models

        Outputs:
        - output/05_evaluation/ - Evaluation metrics and visualizations
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
        }
        feature_extractor = FeatureExtractor(feature_config)
        config_dict = self.config if isinstance(self.config, dict) else vars(self.config)
        config_dict['ensemble'] = self.config.get('ensemble', {})
        model_trainer = ModelTrainer(config_dict)
        model_trainer.load_ensemble_models(str(self.models_dir))
        test_annots = self.data_loader.test_annotations
        y_true = []
        y_pred = []
        base_model_preds = {name: [] for name in ['hog', 'lbp', 'mask_stats']}
        
        # Get enabled models from config
        ensemble_cfg = self.config.get('ensemble', {})
        enabled_models = {
            'hog': ensemble_cfg.get('use_hog', False),
            'lbp': ensemble_cfg.get('use_lbp', True),
            'mask_stats': ensemble_cfg.get('use_mask_stats', True)
        }
        
        seg_config = self.config.features.get('segmentation', {})
        for batch in self.data_loader.get_image_batches(test_annots):
            self.logger.debug(f"Evaluating batch of {len(batch)} images (batch_size={self.data_loader.batch_size})")
            for annot in batch:
                image = self.data_loader.load_image(annot.image_path)
                # Generate segmentation mask for statistics
                mask = build_nurdle_mask(
                    image,
                    min_dist=seg_config.get('min_dist', 8),
                    min_area=seg_config.get('min_area', 50),
                    max_area=seg_config.get('max_area', 2000)
                )
                
                # Extract individual features
                hog_feat = feature_extractor.extract_hog_features(image)
                lbp_feat = feature_extractor.extract_lbp_features(image)
                mask_stat_feat = feature_extractor.extract_mask_stats_features(mask)
                
                feature_dict = {}
                if len(hog_feat) > 0:
                    feature_dict['hog'] = hog_feat
                if len(lbp_feat) > 0:
                    feature_dict['lbp'] = lbp_feat
                if len(mask_stat_feat) > 0:
                    feature_dict['mask_stats'] = mask_stat_feat
                
                # Get base model predictions (only for enabled models)
                base_preds = model_trainer.predict_ensemble(feature_dict)
                for name in ['hog', 'lbp', 'mask_stats']:
                    # Only append prediction if model is enabled and made a prediction
                    if enabled_models[name] and name in base_preds:
                        base_model_preds[name].append(base_preds[name])
                
                # Get ensemble prediction
                ensemble_pred = model_trainer.predict_ensemble_meta(base_preds)
                y_pred.append(ensemble_pred)
                y_true.append(annot.nurdle_count)
        
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
        
        # Add base model evaluation metrics only for enabled models
        for model_name in ['hog', 'lbp', 'mask_stats']:
            if enabled_models[model_name] and base_model_preds[model_name]:
                base_mae = mean_absolute_error(y_true, base_model_preds[model_name])
                base_rmse = np.sqrt(mean_squared_error(y_true, base_model_preds[model_name]))
                base_mape = np.mean(np.abs((np.array(y_true) - np.array(base_model_preds[model_name])) / np.maximum(np.array(y_true), 1))) * 100
                eval_metrics[f'{model_name}_mae'] = float(base_mae)
                eval_metrics[f'{model_name}_rmse'] = float(base_rmse)
                eval_metrics[f'{model_name}_mape'] = float(base_mape)
        
        with open(output_dir / 'evaluation_metrics.json', 'w') as f:
            json.dump(eval_metrics, f, indent=2)
        
        # Log evaluation results
        self.logger.info("Evaluation Results:")
        self.logger.info(f"  Ensemble - MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.2f}%")
        for model_name in ['hog', 'lbp', 'mask_stats']:
            if enabled_models[model_name] and f'{model_name}_mae' in eval_metrics:
                self.logger.info(f"  {model_name.upper()} - MAE: {eval_metrics[f'{model_name}_mae']:.4f}, MAPE: {eval_metrics[f'{model_name}_mape']:.2f}%")
        
        # Calculate baseline metrics (mean/median predictions)
        baseline_mean_pred = np.full_like(y_true, np.mean(y_true), dtype=float)
        baseline_median_pred = np.full_like(y_true, np.median(y_true), dtype=float)
        
        baseline_mean_mae = mean_absolute_error(y_true, baseline_mean_pred)
        baseline_mean_rmse = np.sqrt(mean_squared_error(y_true, baseline_mean_pred))
        baseline_mean_mape = np.mean(np.abs((np.array(y_true) - baseline_mean_pred) / np.maximum(np.array(y_true), 1))) * 100
        
        baseline_median_mae = mean_absolute_error(y_true, baseline_median_pred)
        baseline_median_rmse = np.sqrt(mean_squared_error(y_true, baseline_median_pred))
        baseline_median_mape = np.mean(np.abs((np.array(y_true) - baseline_median_pred) / np.maximum(np.array(y_true), 1))) * 100
        
        eval_metrics['baseline_mean_mae'] = float(baseline_mean_mae)
        eval_metrics['baseline_mean_rmse'] = float(baseline_mean_rmse)
        eval_metrics['baseline_mean_mape'] = float(baseline_mean_mape)
        eval_metrics['baseline_median_mae'] = float(baseline_median_mae)
        eval_metrics['baseline_median_rmse'] = float(baseline_median_rmse)
        eval_metrics['baseline_median_mape'] = float(baseline_median_mape)
        
        self.logger.info(f"  Baseline (Mean) - MAE: {baseline_mean_mae:.4f}, RMSE: {baseline_mean_rmse:.4f}, MAPE: {baseline_mean_mape:.2f}%")
        self.logger.info(f"  Baseline (Median) - MAE: {baseline_median_mae:.4f}, RMSE: {baseline_median_rmse:.4f}, MAPE: {baseline_median_mape:.2f}%")
        
        # Generate visualizations
        import matplotlib.pyplot as plt
        
        # True vs Predicted scatter with baseline
        plt.figure(figsize=(10, 8))
        plt.scatter(y_true, y_pred, alpha=0.7, color='steelblue', s=100, label='Ensemble predictions', edgecolors='black', linewidth=0.5)
        plt.scatter(y_true, baseline_mean_pred, alpha=0.5, color='red', s=80, marker='x', label='Baseline (mean)', linewidth=2)
        xy_min = min(min(y_true), min(y_pred), min(baseline_mean_pred))
        xy_max = max(max(y_true), max(y_pred), max(baseline_mean_pred))
        plt.plot([xy_min, xy_max], [xy_min, xy_max], 'k--', linewidth=2, label='Ideal (y = x)')
        plt.xlabel('True Count', fontsize=12)
        plt.ylabel('Predicted Count', fontsize=12)
        plt.title('True vs Predicted Counts (Ensemble vs Baseline)', fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'true_vs_pred_baseline.png', dpi=150)
        plt.close()
        
        # Metrics comparison bar chart
        metrics_names = ['MAE', 'RMSE', 'MAPE']
        ensemble_metrics_vals = [mae, rmse, mape]
        baseline_mean_vals = [baseline_mean_mae, baseline_mean_rmse, baseline_mean_mape]
        
        x = np.arange(len(metrics_names))
        width = 0.35
        
        plt.figure(figsize=(10, 6))
        plt.bar(x - width/2, ensemble_metrics_vals, width, label='Ensemble', color='red', edgecolor='black')
        plt.bar(x + width/2, baseline_mean_vals, width, label='Baseline (Mean)', color='steelblue', edgecolor='black')
        plt.xlabel('Metrics', fontsize=12)
        plt.ylabel('Error Value', fontsize=12)
        plt.title('Model Performance vs Baseline', fontsize=14)
        plt.xticks(x, metrics_names, fontsize=11)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(output_dir / 'metrics_comparison.png', dpi=150)
        plt.close()
        
        # Load training metrics for comparison
        training_metrics_path = self.output_dirs.get('training') / 'training_metrics.json' if 'training' in self.output_dirs else None
        if training_metrics_path and training_metrics_path.exists():
            with open(training_metrics_path, 'r') as f:
                train_metrics_data = json.load(f)
            
            # Get ensemble training metrics
            train_ensemble = train_metrics_data.get('ensemble_metrics', {})
            train_mae = train_ensemble.get('mae', 0)
            train_rmse = train_ensemble.get('rmse', 0)
            train_mape = train_ensemble.get('mape', 0)
            
            # Train vs Test comparison plot
            stage_names = ['Training', 'Test']
            mae_vals = [train_mae, mae]
            rmse_vals = [train_rmse, rmse]
            mape_vals = [train_mape, mape]
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            axes[0].bar(stage_names, mae_vals, color=['steelblue', 'coral'], edgecolor='black')
            axes[0].set_ylabel('MAE', fontsize=11)
            axes[0].set_title('Mean Absolute Error', fontsize=12)
            axes[0].grid(True, alpha=0.3, axis='y')
            
            axes[1].bar(stage_names, rmse_vals, color=['steelblue', 'coral'], edgecolor='black')
            axes[1].set_ylabel('RMSE', fontsize=11)
            axes[1].set_title('Root Mean Squared Error', fontsize=12)
            axes[1].grid(True, alpha=0.3, axis='y')
            
            axes[2].bar(stage_names, mape_vals, color=['steelblue', 'coral'], edgecolor='black')
            axes[2].set_ylabel('MAPE (%)', fontsize=11)
            axes[2].set_title('Mean Absolute Percentage Error', fontsize=12)
            axes[2].grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            plt.savefig(output_dir / 'train_vs_test.png', dpi=150)
            plt.close()
            
            self.logger.info(f"Train vs Test - Training MAPE: {train_mape:.2f}%, Test MAPE: {mape:.2f}%")
        
        # True vs Predicted scatter (original)
        plt.figure(figsize=(6, 6))
        plt.scatter(y_true, y_pred, alpha=0.7, color='steelblue', label='Ensemble predictions')
        xy_min = min(min(y_true), min(y_pred))
        xy_max = max(max(y_true), max(y_pred))
        plt.plot([xy_min, xy_max], [xy_min, xy_max], 'k--', label='Ideal (y = x)')
        plt.xlabel('True Count')
        plt.ylabel('Predicted Count')
        plt.title('True vs Predicted Counts (Ensemble)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'true_vs_pred.png')
        plt.close()
        
        # Individual test image visualizations
        for ann, pred_count in zip(test_annots, y_pred):
            img = self.data_loader.load_image(ann.image_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            actual_count = ann.nurdle_count
            plt.figure(figsize=(6, 6))
            plt.imshow(img_rgb)
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
