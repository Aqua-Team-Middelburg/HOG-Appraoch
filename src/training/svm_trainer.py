"""
SVM trainer module for all feature types (HOG, LBP, Combined, Stacking).

This module implements comprehensive SVM training with hyperparameter tuning,
cross-validation, and model persistence for the nurdle detection pipeline.
"""

import numpy as np
import joblib
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional, Union
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

from ..utils.config import ConfigLoader

logger = logging.getLogger(__name__)


class SVMTrainer:
    """
    Comprehensive SVM trainer supporting multiple feature types and ensemble methods.
    
    Supports:
    - HOG-only SVM
    - LBP-only SVM  
    - Combined HOG+LBP SVM
    - Stacking ensemble of all models
    """
    
    def __init__(self, config: ConfigLoader):
        """
        Initialize SVM trainer.
        
        Args:
            config: Configuration loader instance
        """
        self.config = config
        
        # Training configuration
        training_config = config.get_section('training')
        self.svm_config = training_config.get('svm', {})
        self.cv_config = training_config.get('cross_validation', {})
        self.tuning_config = training_config.get('hyperparameter_tuning', {})
        self.ensemble_config = training_config.get('ensemble', {})
        
        # Paths
        self.models_dir = Path(config.get('paths.output_dir')) / config.get('paths.models_dir')
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Random state for reproducibility
        self.random_state = training_config.get('random_state', 42)
        
        # Scalers for each feature type
        self.scalers = {}
        
        # Trained models
        self.trained_models = {}
        
        logger.info("SVM trainer initialized")
    
    def create_svm_model(self, **kwargs) -> Union[LinearSVC, SVC]:
        """
        Create SVM model with configured parameters.
        
        Args:
            **kwargs: Override parameters
            
        Returns:
            Configured SVM model
        """
        # Merge default config with overrides
        params = {**self.svm_config, **kwargs}
        
        # Use LinearSVC for linear kernel (more efficient)
        if params.get('kernel', 'linear') == 'linear':
            model = LinearSVC(
                C=params.get('C', 1.0),
                max_iter=params.get('max_iter', 1000),
                random_state=self.random_state,
                class_weight=params.get('class_weight', None)
            )
        else:
            model = SVC(
                kernel=params.get('kernel', 'rbf'),
                C=params.get('C', 1.0),
                gamma=params.get('gamma', 'scale'),
                max_iter=params.get('max_iter', 1000),
                random_state=self.random_state,
                class_weight=params.get('class_weight', None),
                probability=True  # Enable probability estimates for stacking
            )
        
        return model
    
    def optimize_hyperparameters(self, 
                                features: np.ndarray, 
                                labels: np.ndarray,
                                feature_type: str) -> Dict[str, Any]:
        """
        Optimize SVM hyperparameters using Optuna or GridSearch.
        
        Args:
            features: Training feature matrix
            labels: Training labels
            feature_type: Type of features being optimized
            
        Returns:
            Dictionary containing best parameters and optimization results
        """
        if not self.tuning_config.get('enabled', False):
            logger.info("Hyperparameter tuning disabled, using default parameters")
            return {'best_params': self.svm_config, 'method': 'default'}
        
        if OPTUNA_AVAILABLE and self.tuning_config.get('use_optuna', True):
            return self._optimize_with_optuna(features, labels, feature_type)
        else:
            return self._optimize_with_gridsearch(features, labels, feature_type)
    
    def _optimize_with_optuna(self, 
                            features: np.ndarray, 
                            labels: np.ndarray,
                            feature_type: str) -> Dict[str, Any]:
        """Optimize hyperparameters using Optuna."""
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
            model = self.create_svm_model(**params)
            
            # Cross-validation
            cv_folds = self.tuning_config.get('cv_folds', 3)
            scores = cross_val_score(
                model, features, labels,
                cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state),
                scoring='f1',
                n_jobs=-1
            )
            
            return float(np.mean(scores))
        
        # Create and run study
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=self.random_state)
        )
        
        n_trials = self.tuning_config.get('n_trials', 50)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        
        results = {
            'best_params': study.best_params,
            'best_score': study.best_value,
            'method': 'optuna',
            'n_trials': len(study.trials)
        }
        
        logger.info(f"Optuna optimization completed: {results['best_score']:.4f} F1 score")
        return results
    
    def _optimize_with_gridsearch(self,
                                features: np.ndarray,
                                labels: np.ndarray, 
                                feature_type: str) -> Dict[str, Any]:
        """Optimize hyperparameters using GridSearchCV."""
        logger.info(f"Starting GridSearch optimization for {feature_type} features")
        
        # Get parameter grid from config
        param_grid = self.tuning_config.get('param_grid', {
            'C': [0.1, 1.0, 10.0],
            'kernel': ['linear', 'rbf']
        })
        
        # Create base model
        model = self.create_svm_model()
        
        # Grid search
        cv_folds = self.tuning_config.get('cv_folds', 3)
        grid_search = GridSearchCV(
            model,
            param_grid,
            cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state),
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
        return results
    
    def train_single_model(self,
                          features: np.ndarray,
                          labels: np.ndarray,
                          feature_type: str) -> Dict[str, Any]:
        """
        Train a single SVM model for specified feature type.
        
        Args:
            features: Training feature matrix
            labels: Training labels  
            feature_type: Type of features ('hog', 'lbp', 'combined')
            
        Returns:
            Dictionary containing trained model and training results
        """
        logger.info(f"Training {feature_type} SVM model")
        
        # Scale features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        self.scalers[feature_type] = scaler
        
        # Optimize hyperparameters
        optimization_results = self.optimize_hyperparameters(scaled_features, labels, feature_type)
        
        # Train final model with best parameters
        best_params = optimization_results['best_params']
        model = self.create_svm_model(**best_params)
        model.fit(scaled_features, labels)
        
        # Cross-validation evaluation
        cv_folds = self.cv_config.get('folds', 5)
        cv_scores = cross_val_score(
            model, scaled_features, labels,
            cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state),
            scoring='f1'
        )
        
        # Additional metrics
        predictions = model.predict(scaled_features)
        
        try:
            # For probability-capable models (only SVC with probability=True has predict_proba)
            if hasattr(model, 'decision_function'):
                decision_scores = model.decision_function(scaled_features)
                auc_score = roc_auc_score(labels, decision_scores)
            else:
                auc_score = None
        except Exception:
            auc_score = None
        
        # Create model package
        model_package = {
            'model': model,
            'scaler': scaler,
            'feature_type': feature_type,
            'best_params': best_params,
            'optimization_results': optimization_results,
            'cv_scores': cv_scores,
            'cv_mean': np.mean(cv_scores),
            'cv_std': np.std(cv_scores),
            'auc_score': auc_score,
            'training_samples': len(features),
            'feature_dimension': features.shape[1],
            'class_distribution': np.bincount(labels),
            'timestamp': datetime.now().isoformat()
        }
        
        # Save model
        self._save_model(model_package, feature_type)
        
        # Store in trained models
        self.trained_models[feature_type] = model_package
        
        logger.info(f"{feature_type} model trained - CV F1: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
        
        return model_package
    
    def train_stacking_ensemble(self,
                               feature_data: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, Any]:
        """
        Train stacking ensemble using base models.
        
        Args:
            feature_data: Dictionary containing features for each type
            
        Returns:
            Dictionary containing stacking model and results
        """
        if not self.ensemble_config.get('stacking', {}).get('enabled', False):
            logger.info("Stacking ensemble disabled")
            return {}
        
        logger.info("Training stacking ensemble")
        
        # Get base model types
        base_model_types = self.ensemble_config.get('stacking', {}).get('base_models', ['hog', 'lbp'])
        
        # Prepare base estimators
        base_estimators = []
        stacking_features_list = []
        
        # Get common samples (intersection of all feature types)
        common_labels = None
        min_samples = float('inf')
        
        for model_type in base_model_types:
            if model_type in feature_data and model_type in self.trained_models:
                n_samples = len(feature_data[model_type]['labels'])
                if n_samples < min_samples:
                    min_samples = n_samples
                    common_labels = feature_data[model_type]['labels']
        
        if common_labels is None:
            raise ValueError("No common samples found for stacking")
        
        # Create base estimators and prepare features
        for model_type in base_model_types:
            if model_type in self.trained_models:
                model_package = self.trained_models[model_type]
                base_estimators.append((
                    f'{model_type}_svm',
                    model_package['model']
                ))
                
                # Get scaled features
                features = feature_data[model_type]['features'][:min_samples]
                scaled_features = self.scalers[model_type].transform(features)
                stacking_features_list.append(scaled_features)
        
        if len(base_estimators) < 2:
            raise ValueError("Need at least 2 base models for stacking")
        
        # Meta-classifier
        meta_classifier_type = self.ensemble_config.get('stacking', {}).get('meta_classifier', 'logistic_regression')
        
        if meta_classifier_type == 'logistic_regression':
            meta_classifier = LogisticRegression(random_state=self.random_state)
        else:
            # Use SVM as meta-classifier
            meta_classifier = self.create_svm_model(C=1.0)
        
        # Create stacking classifier
        stacking_cv = self.ensemble_config.get('stacking', {}).get('cv_folds', 3)
        stacking_model = StackingClassifier(
            estimators=base_estimators,
            final_estimator=meta_classifier,
            cv=stacking_cv,
            n_jobs=-1
        )
        
        # We need to create a feature matrix that combines all base features
        # For stacking, we'll use the combined features if available
        if 'combined' in feature_data:
            stacking_features = feature_data['combined']['features'][:min_samples]
            stacking_scaler = StandardScaler()
            scaled_stacking_features = stacking_scaler.fit_transform(stacking_features)
        else:
            # Fallback: concatenate all available features
            stacking_features = np.hstack(stacking_features_list)
            stacking_scaler = StandardScaler()
            scaled_stacking_features = stacking_scaler.fit_transform(stacking_features)
        
        # Train stacking model
        stacking_model.fit(scaled_stacking_features, common_labels[:min_samples])
        
        # Cross-validation evaluation
        cv_folds = self.cv_config.get('folds', 5)
        cv_scores = cross_val_score(
            stacking_model, scaled_stacking_features, common_labels[:min_samples],
            cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state),
            scoring='f1'
        )
        
        # Create stacking package
        stacking_package = {
            'model': stacking_model,
            'scaler': stacking_scaler,
            'feature_type': 'stacking',
            'base_models': base_model_types,
            'meta_classifier': meta_classifier_type,
            'cv_scores': cv_scores,
            'cv_mean': np.mean(cv_scores),
            'cv_std': np.std(cv_scores),
            'training_samples': min_samples,
            'feature_dimension': scaled_stacking_features.shape[1],
            'timestamp': datetime.now().isoformat()
        }
        
        # Save stacking model
        self._save_model(stacking_package, 'stacking')
        
        # Store in trained models
        self.trained_models['stacking'] = stacking_package
        
        logger.info(f"Stacking model trained - CV F1: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
        
        return stacking_package
    
    def train_all_models(self, feature_data: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, Dict[str, Any]]:
        """
        Train all model types (HOG, LBP, Combined, Stacking).
        
        Args:
            feature_data: Dictionary containing features and labels for each type
            
        Returns:
            Dictionary containing all trained models
        """
        logger.info("Starting training for all model types")
        
        results = {}
        
        # Train individual models
        for feature_type in ['hog', 'lbp', 'combined']:
            if feature_type in feature_data:
                try:
                    model_result = self.train_single_model(
                        feature_data[feature_type]['features'],
                        feature_data[feature_type]['labels'],
                        feature_type
                    )
                    results[feature_type] = model_result
                    logger.info(f"✅ {feature_type} model training completed")
                except Exception as e:
                    logger.error(f"❌ {feature_type} model training failed: {e}")
                    results[feature_type] = {'error': str(e)}
        
        # Train stacking ensemble
        if len(results) >= 2:  # Need at least 2 base models
            try:
                stacking_result = self.train_stacking_ensemble(feature_data)
                if stacking_result:
                    results['stacking'] = stacking_result
                    logger.info("✅ Stacking ensemble training completed")
            except Exception as e:
                logger.error(f"❌ Stacking ensemble training failed: {e}")
                results['stacking'] = {'error': str(e)}
        
        logger.info(f"Training completed: {len(results)} models trained")
        return results
    
    def _save_model(self, model_package: Dict[str, Any], feature_type: str) -> None:
        """
        Save trained model package to disk.
        
        Args:
            model_package: Complete model package
            feature_type: Type of model being saved
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Model filename
        model_filename = f"svm_{feature_type}_model_{timestamp}.pkl"
        model_path = self.models_dir / model_filename
        
        # Save model package
        joblib.dump(model_package, model_path)
        
        # Save metadata separately
        metadata = {k: v for k, v in model_package.items() 
                   if k not in ['model', 'scaler']}  # Exclude non-serializable objects
        
        metadata_filename = f"svm_{feature_type}_metadata_{timestamp}.json"
        metadata_path = self.models_dir / metadata_filename
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Model saved: {model_filename}")
    
    def load_model(self, model_path: str) -> Dict[str, Any]:
        """
        Load previously trained model.
        
        Args:
            model_path: Path to saved model file
            
        Returns:
            Loaded model package
        """
        model_package = joblib.load(model_path)
        feature_type = model_package['feature_type']
        
        # Store scaler for future use
        self.scalers[feature_type] = model_package['scaler']
        self.trained_models[feature_type] = model_package
        
        logger.info(f"Model loaded: {feature_type}")
        return model_package
    
    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get summary of all trained models.
        
        Returns:
            Summary dictionary
        """
        summary = {
            'total_models': len(self.trained_models),
            'models': {}
        }
        
        for feature_type, model_package in self.trained_models.items():
            if 'error' not in model_package:
                summary['models'][feature_type] = {
                    'cv_f1_mean': model_package.get('cv_mean', 0),
                    'cv_f1_std': model_package.get('cv_std', 0),
                    'training_samples': model_package.get('training_samples', 0),
                    'feature_dimension': model_package.get('feature_dimension', 0),
                    'timestamp': model_package.get('timestamp', 'unknown')
                }
            else:
                summary['models'][feature_type] = {
                    'status': 'failed',
                    'error': model_package['error']
                }
        
        return summary