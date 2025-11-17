"""
Hyperparameter optimization module for SVM training.

This module implements hyperparameter tuning using Optuna or GridSearchCV
to find optimal model parameters for maximum performance.
"""

import numpy as np
import logging
from typing import Dict, Any
from sklearn.model_selection import GridSearchCV, StratifiedKFold

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

logger = logging.getLogger(__name__)


class HyperparameterOptimizer:
    """
    Optimizes SVM hyperparameters using Optuna or GridSearchCV.
    
    Searches parameter space to find best C, kernel, gamma values
    for maximum cross-validation performance.
    """
    
    def __init__(self, config: dict, random_state: int = 42):
        """
        Initialize hyperparameter optimizer.
        
        Args:
            config: Hyperparameter tuning configuration
            random_state: Random seed for reproducibility
        """
        self.config = config
        self.random_state = random_state
        self.enabled = config.get('enabled', False)
        
        # Optuna configuration
        self.use_optuna = config.get('use_optuna', True) and OPTUNA_AVAILABLE
        self.n_trials = config.get('n_trials', 50)
        self.cv_folds = config.get('cv_folds', 3)
        
        # GridSearch configuration
        self.param_grid = config.get('param_grid', {
            'C': [0.1, 1.0, 10.0, 100.0],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto']
        })
        
        logger.info(f"Hyperparameter optimizer initialized: enabled={self.enabled}, "
                   f"method={'optuna' if self.use_optuna else 'gridsearch'}")
    
    def optimize(self, 
                features: np.ndarray,
                labels: np.ndarray,
                feature_type: str,
                model_factory,
                default_params: dict) -> Dict[str, Any]:
        """
        Optimize hyperparameters using configured method.
        
        Args:
            features: Training feature matrix
            labels: Training labels
            feature_type: Type of features being optimized
            model_factory: Function that creates a new SVM model
            default_params: Default SVM parameters
            
        Returns:
            Dictionary containing best parameters and optimization results
        """
        if not self.enabled:
            logger.info("Hyperparameter tuning disabled, using default parameters")
            return {'best_params': default_params, 'method': 'default'}
        
        if self.use_optuna:
            return self._optimize_with_optuna(features, labels, feature_type, model_factory)
        else:
            return self._optimize_with_gridsearch(features, labels, feature_type, model_factory)
    
    def _optimize_with_optuna(self,
                             features: np.ndarray,
                             labels: np.ndarray,
                             feature_type: str,
                             model_factory) -> Dict[str, Any]:
        """
        Optimize hyperparameters using Optuna.
        
        Args:
            features: Training feature matrix
            labels: Training labels
            feature_type: Type of features being optimized
            model_factory: Function that creates a new SVM model
            
        Returns:
            Dictionary containing best parameters and optimization results
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna not available")
        
        import optuna  # Import inside function to avoid issues
        
        logger.info(f"Starting Optuna optimization for {feature_type} features")
        
        def objective(trial):
            # Suggest hyperparameters
            C = trial.suggest_float('C', 0.01, 100.0, log=True)
            kernel = trial.suggest_categorical('kernel', ['linear', 'rbf'])
            
            params = {'C': C, 'kernel': kernel}
            
            if kernel != 'linear':
                gamma = trial.suggest_categorical('gamma', ['scale', 'auto'])
                params['gamma'] = gamma
            
            # Create and evaluate model
            model = model_factory(**params)
            
            # Cross-validation
            from sklearn.model_selection import cross_val_score
            scores = cross_val_score(
                model, features, labels,
                cv=StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state),
                scoring='f1',
                n_jobs=-1
            )
            
            return float(np.mean(scores))
        
        # Create and run study
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=self.random_state)
        )
        
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
        
        results = {
            'best_params': study.best_params,
            'best_score': study.best_value,
            'method': 'optuna',
            'n_trials': len(study.trials)
        }
        
        logger.info(f"Optuna optimization completed: {results['best_score']:.4f} F1 score")
        logger.info(f"Best parameters: {results['best_params']}")
        
        return results
    
    def _optimize_with_gridsearch(self,
                                 features: np.ndarray,
                                 labels: np.ndarray,
                                 feature_type: str,
                                 model_factory) -> Dict[str, Any]:
        """
        Optimize hyperparameters using GridSearchCV.
        
        Args:
            features: Training feature matrix
            labels: Training labels
            feature_type: Type of features being optimized
            model_factory: Function that creates a new SVM model
            
        Returns:
            Dictionary containing best parameters and optimization results
        """
        logger.info(f"Starting GridSearch optimization for {feature_type} features")
        
        # Create base model (will be cloned by GridSearchCV)
        model = model_factory()
        
        # Grid search
        grid_search = GridSearchCV(
            model,
            self.param_grid,
            cv=StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state),
            scoring='f1',
            n_jobs=-1
        )
        
        grid_search.fit(features, labels)
        
        results = {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'method': 'gridsearch',
            'n_combinations': len(grid_search.cv_results_['params'])
        }
        
        logger.info(f"GridSearch optimization completed: {results['best_score']:.4f} F1 score")
        logger.info(f"Best parameters: {results['best_params']}")
        
        return results
