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
        self.n_trials = config.get('n_trials', 20)
        self.tune_thresholds = config.get('tune_thresholds', True)
        self.save_plots = config.get('save_plots', True)
        self.optimization_direction = config.get('direction', 'maximize')
        self.metric_to_optimize = config.get('metric', 'f1_score')
        
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
        
        # Run optimization
        n_trials = 20
        study.optimize(objective, n_trials=n_trials)
        
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
        
        # Run optimization
        n_trials = 20
        study.optimize(objective, n_trials=n_trials)
        
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
        """Save optimization results for a single study."""
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
            
            self.logger.info(f"Saved optimization results to {output_path}")
        except Exception as e:
            self.logger.error(f"Error saving optimization results: {e}")
    
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
