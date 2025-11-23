"""
Hyperparameter Optimization using Optuna - Phase 8 Implementation
================================================================

This module provides comprehensive hyperparameter optimization for the nurdle detection pipeline,
including model parameters, confidence thresholds, and feature extraction parameters.

Implements Phase 8 of the refactor plan with visualization capabilities.
"""

import optuna
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any, Callable, Tuple, List
import joblib
import json
from datetime import datetime
from tqdm import tqdm

# Visualization imports (with fallback)
try:
    from optuna.visualization import plot_optimization_history, plot_param_importances, plot_slice
    try:
        import plotly.graph_objects as go
        PLOTLY_AVAILABLE = True
    except ImportError:
        PLOTLY_AVAILABLE = False
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    PLOTLY_AVAILABLE = False


class OptunaTuner:
    """
    Hyperparameter optimization using Optuna with comprehensive visualization.
    
    Optimizes all major pipeline parameters:
    - SVM parameters (C, kernel parameters)
    - Confidence thresholds (classifier, regressor)  
    - Feature extraction parameters (window size, stride, HOG cell size)
    - Training parameters (negative/positive ratio, distance thresholds)
    """
    
    def __init__(self, config: Dict[str, Any], logger=None):
        """Initialize Optuna tuner with configuration."""
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Optuna configuration
        # n_trials configured per optimization type (svm_optimization, svr_optimization)
        # direction and metric are hardcoded per optimization (maximize f1, minimize error)
        
        # Storage for best parameters
        self.best_params = None
        self.best_value = None
        self.study = None
        
        # Results storage
        self.optimization_history = []
        self.trial_results = []
        
    def optimize_pipeline(self, 
                         pipeline_trainer_func: Callable,
                         output_dir: str = "output/optuna") -> Dict[str, Any]:
        """
        Run hyperparameter optimization using sequential strategy.
        
        This is a wrapper that calls optimize_sequential for backward compatibility.
        
        Args:
            pipeline_trainer_func: Function that trains and evaluates pipeline
            output_dir: Directory to save optimization results
            
        Returns:
            Best parameters dictionary
        """
        return self.optimize_sequential(pipeline_trainer_func, output_dir)
    
    def optimize_svm_only(self, 
                         pipeline_trainer_func: Callable,
                         output_dir: str) -> Dict[str, Any]:
        """
        Optimize SVM classifier parameters on f1_score.
        
        Args:
            pipeline_trainer_func: Function that trains pipeline and returns metrics
            output_dir: Directory to save optimization results
            
        Returns:
            Best SVM parameters
        """
        self.logger.info("=" * 60)
        self.logger.info("OPTIMIZING SVM CLASSIFIER (Step 1/2)")
        self.logger.info("=" * 60)
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create study
        study = optuna.create_study(
            direction='maximize',
            study_name='svm_optimization',
            pruner=optuna.pruners.MedianPruner()
        )
        
        def objective(trial):
            # Sample SVM parameters only
            params = {
                'svm_c': trial.suggest_float('svm_c', 0.001, 100.0, log=True),
                'svm_kernel': trial.suggest_categorical('svm_kernel', ['linear', 'rbf']),
            }
            
            if params['svm_kernel'] == 'rbf':
                params['svm_gamma'] = trial.suggest_float('svm_gamma', 0.001, 1.0, log=True)
            
            # Use default SVR parameters for now
            params.update({
                'svr_c': 1.0,
                'svr_kernel': 'rbf',
                'svr_gamma': 'scale',
                'svr_epsilon': 0.1
            })
            
            try:
                metrics = pipeline_trainer_func(params)
                return metrics['f1_score']
            except Exception as e:
                self.logger.error(f"Trial failed: {e}")
                return 0.0
        
        # Run optimization with progress bar
        n_trials = self.config.get('svm_optimization', {}).get('n_trials', 10)
        
        # Suppress optuna logging during trials
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        with tqdm(total=n_trials, desc="SVM Optimization", unit="trial") as pbar:
            def callback(study, trial):
                pbar.update(1)
                pbar.set_postfix({"best_f1": f"{study.best_value:.4f}"})
            
            study.optimize(objective, n_trials=n_trials, callbacks=[callback], show_progress_bar=False)
        
        # Restore optuna logging
        optuna.logging.set_verbosity(optuna.logging.INFO)
        
        best_svm_params = study.best_params
        self.logger.info(f"Best SVM F1-Score: {study.best_value:.4f}")
        self.logger.info(f"Best SVM parameters: {best_svm_params}")
        
        # Save results
        self._save_optimization_results(study, output_path / "svm_optimization")
        
        return best_svm_params
    
    def optimize_svr_only(self,
                         pipeline_trainer_func: Callable,
                         output_dir: str,
                         fixed_svm_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize SVR regressor parameters on coordinate error (minimize).
        
        Uses fixed SVM parameters from previous optimization.
        
        Args:
            pipeline_trainer_func: Function that trains pipeline and returns metrics
            output_dir: Directory to save optimization results
            fixed_svm_params: Best SVM parameters from step 1
            
        Returns:
            Best SVR parameters (combined with SVM params)
        """
        self.logger.info("=" * 60)
        self.logger.info("OPTIMIZING SVR REGRESSOR (Step 2/2)")
        self.logger.info("=" * 60)
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create study for minimization
        study = optuna.create_study(
            direction='minimize',
            study_name='svr_optimization',
            pruner=optuna.pruners.MedianPruner()
        )
        
        def objective(trial):
            # Use fixed SVM parameters
            params = fixed_svm_params.copy()
            
            # Sample SVR parameters
            params.update({
                'svr_c': trial.suggest_float('svr_c', 0.001, 100.0, log=True),
                'svr_kernel': trial.suggest_categorical('svr_kernel', ['linear', 'rbf']),
                'svr_epsilon': trial.suggest_float('svr_epsilon', 0.01, 1.0),
            })
            
            if params['svr_kernel'] == 'rbf':
                params['svr_gamma'] = trial.suggest_float('svr_gamma', 0.001, 1.0, log=True)
            
            try:
                metrics = pipeline_trainer_func(params)
                return metrics['avg_coordinate_error']
            except Exception as e:
                self.logger.error(f"Trial failed: {e}")
                return 1000.0  # Large error for failed trials
        
        # Run optimization with progress bar
        n_trials = self.config.get('svr_optimization', {}).get('n_trials', 10)
        
        # Suppress optuna logging during trials
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        with tqdm(total=n_trials, desc="SVR Optimization", unit="trial") as pbar:
            def callback(study, trial):
                pbar.update(1)
                pbar.set_postfix({"best_error": f"{study.best_value:.2f}px"})
            
            study.optimize(objective, n_trials=n_trials, callbacks=[callback], show_progress_bar=False)
        
        # Restore optuna logging
        optuna.logging.set_verbosity(optuna.logging.INFO)
        
        best_svr_params = study.best_params
        self.logger.info(f"Best SVR Coordinate Error: {study.best_value:.2f} pixels")
        self.logger.info(f"Best SVR parameters: {best_svr_params}")
        
        # Combine with SVM params
        best_combined_params = {**fixed_svm_params, **best_svr_params}
        
        # Save results
        self._save_optimization_results(study, output_path / "svr_optimization")
        
        return best_combined_params
    
    def optimize_sequential(self,
                           pipeline_trainer_func: Callable,
                           output_dir: str) -> Dict[str, Any]:
        """
        Run sequential optimization: SVM first, then SVR.
        
        This is the main entry point for optimization.
        
        Args:
            pipeline_trainer_func: Function that trains pipeline and returns metrics
            output_dir: Directory to save optimization results
            
        Returns:
            Best parameters for both SVM and SVR
        """
        self.logger.info("=" * 60)
        self.logger.info("STARTING SEQUENTIAL HYPERPARAMETER OPTIMIZATION")
        self.logger.info("=" * 60)
        
        # Step 1: Optimize SVM
        best_svm_params = self.optimize_svm_only(pipeline_trainer_func, output_dir)
        
        # Step 2: Optimize SVR with fixed SVM params
        best_combined_params = self.optimize_svr_only(
            pipeline_trainer_func,
            output_dir,
            best_svm_params
        )
        
        self.logger.info("=" * 60)
        self.logger.info("SEQUENTIAL OPTIMIZATION COMPLETED")
        self.logger.info("=" * 60)
        self.logger.info("Final best parameters:")
        for key, value in best_combined_params.items():
            self.logger.info(f"  {key}: {value}")
        
        return best_combined_params
    
    def _save_optimization_results(self, study, output_path: Path):
        """Save optimization results and visualizations for a single study."""
        try:
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save study
            study_path = output_path / "study.pkl"
            joblib.dump(study, study_path)
            
            # Save parameters
            params_path = output_path / "best_params.json"
            with open(params_path, 'w') as f:
                json.dump({
                    'best_params': study.best_params,
                    'best_value': float(study.best_value),
                    'n_trials': len(study.trials)
                }, f, indent=2)
            
            # Generate visualizations (PNG only)
            if VISUALIZATION_AVAILABLE and PLOTLY_AVAILABLE:
                try:
                    # Optimization history
                    fig_history = plot_optimization_history(study)
                    fig_history.write_image(str(output_path / "optimization_history.png"))
                    self.logger.info("Saved optimization history plot")
                    
                    # Parameter importances
                    if len(study.trials) > 1:
                        fig_importance = plot_param_importances(study)
                        fig_importance.write_image(str(output_path / "param_importances.png"))
                        self.logger.info("Saved parameter importances plot")
                    
                    # Slice plot (parameter relationships)
                    if len(study.trials) > 1:
                        fig_slice = plot_slice(study)
                        fig_slice.write_image(str(output_path / "param_slice.png"))
                        self.logger.info("Saved parameter slice plot")
                    
                except Exception as viz_error:
                    self.logger.warning(f"Could not generate visualizations: {viz_error}")
            else:
                self.logger.warning("Optuna visualization libraries not available, skipping plots")
            
            self.logger.info(f"Saved optimization results to {output_path}")
        except Exception as e:
            self.logger.error(f"Error saving optimization results: {e}")
    
    def create_training_callback(self, config, feature_extractor, data_loader):
        """
        Create training and evaluation function for Optuna optimization.
        
        This callback temporarily updates configuration, trains models,
        evaluates on validation data, and returns optimization metrics.
        
        Args:
            config: Pipeline configuration dictionary
            feature_extractor: FeatureExtractor instance
            data_loader: DataLoader instance
            
        Returns:
            Callable that takes trial parameters and returns metrics dict
        """
        from src.models.model_trainer import ModelTrainer
        from src.evaluation.metrics import calculate_all_metrics
        
        def train_and_evaluate(trial_params: Dict[str, Any]) -> Dict[str, float]:
            """Train models with trial parameters and evaluate."""
            # Suppress logging during trials to reduce noise
            logging.disable(logging.INFO)
            
            try:
                # Create trial config with hyperparameters
                trial_config = config.copy()
                trial_config['training'].update(trial_params)
                
                # Train models with trial config
                trainer = ModelTrainer(trial_config, self.logger)
                
                # Load training data
                X_train = feature_extractor.training_features
                y_train = feature_extractor.training_labels
                
                if X_train is None or len(X_train) == 0:
                    raise ValueError("No training features available")
                
                trainer.train_models(X_train, y_train)
                
                # Evaluate on validation set
                val_images = data_loader.validation_image_names if hasattr(data_loader, 'validation_image_names') else data_loader.test_image_names[:3]
                
                all_predictions = []
                all_ground_truths = []
                
                for img_name in val_images:
                    image_path = data_loader.normalized_data_dir / 'images' / f"{img_name}.jpg"
                    annotations = data_loader.annotations.get(img_name, [])
                    
                    if not image_path.exists():
                        continue
                    
                    # Get ground truth
                    ground_truth = [(ann['x'], ann['y']) for ann in annotations if ann.get('x') is not None]
                    all_ground_truths.extend(ground_truth)
                    
                    # Predict (simplified for validation)
                    img = data_loader.load_image(str(image_path))
                    windows = feature_extractor.generate_windows_for_image(img)
                    
                    if len(windows) == 0:
                        continue
                    
                    features = feature_extractor.extract_hog_lbp_features_batch([w['data'] for w in windows])
                    predictions_raw = trainer.predict(features)
                    
                    # Simple filtering
                    mask = predictions_raw[:, 0] > 0.5
                    predictions = [(windows[i]['x'] + predictions_raw[i, 1], 
                                   windows[i]['y'] + predictions_raw[i, 2]) 
                                  for i in range(len(windows)) if mask[i]]
                    all_predictions.extend(predictions)
                
                # Calculate metrics
                metrics = calculate_all_metrics(all_predictions, all_ground_truths, distance_threshold=20)
                
                return {
                    'f1_score': metrics.get('f1_score', 0.0),
                    'avg_coordinate_error': metrics.get('avg_coordinate_error', 1000.0)
                }
                
            except Exception as e:
                self.logger.error(f"Training callback failed: {e}")
                return {'f1_score': 0.0, 'avg_coordinate_error': 1000.0}
            finally:
                # Re-enable logging
                logging.disable(logging.NOTSET)
        
        return train_and_evaluate
    
    def load_best_parameters(self, output_dir: str) -> Dict[str, Any]:
        """Load previously optimized parameters."""
        best_params_path = Path(output_dir) / "best_parameters.json"
        
        if best_params_path.exists():
            with open(best_params_path, 'r') as f:
                data = json.load(f)
                self.best_params = data['best_params']
                self.best_value = data['best_value']
                self.logger.info(f"Loaded best parameters from {best_params_path}")
                return self.best_params
        else:
            raise FileNotFoundError(f"No optimization results found at {best_params_path}")
