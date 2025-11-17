"""
Ensemble training module for stacking classifiers.

This module implements stacking ensemble training that combines
multiple base models (HOG, LBP) with a meta-classifier.
"""

import numpy as np
import logging
from typing import Dict, Any, List
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class EnsembleTrainer:
    """
    Trains stacking ensemble models from base classifiers.
    
    Combines predictions from multiple feature-specific models
    using a meta-classifier for improved overall performance.
    """
    
    def __init__(self, config: dict, random_state: int = 42):
        """
        Initialize ensemble trainer.
        
        Args:
            config: Ensemble configuration
            random_state: Random seed for reproducibility
        """
        self.config = config
        self.random_state = random_state
        
        # Stacking configuration
        stacking_config = config.get('stacking', {})
        self.enabled = stacking_config.get('enabled', False)
        self.base_models = stacking_config.get('base_models', ['hog', 'lbp'])
        self.meta_classifier_type = stacking_config.get('meta_classifier', 'logistic_regression')
        self.cv_folds = stacking_config.get('cv_folds', 3)
        
        logger.info(f"Ensemble trainer initialized: enabled={self.enabled}, "
                   f"base_models={self.base_models}")
    
    def train_stacking_ensemble(self,
                                feature_data: Dict[str, Dict[str, np.ndarray]],
                                trained_models: Dict[str, Dict[str, Any]],
                                scalers: Dict[str, StandardScaler],
                                cv_folds: int = 5) -> Dict[str, Any]:
        """
        Train stacking ensemble using base models.
        
        Args:
            feature_data: Dictionary containing features for each type
            trained_models: Dictionary of trained base models
            scalers: Feature scalers for each model type
            cv_folds: Number of CV folds for final evaluation
            
        Returns:
            Dictionary containing stacking model and results
        """
        if not self.enabled:
            logger.info("Stacking ensemble disabled")
            return {}
        
        logger.info("Training stacking ensemble")
        
        # Prepare base estimators
        base_estimators = []
        stacking_features_list = []
        
        # Get common samples (intersection of all feature types)
        common_labels = None
        min_samples = float('inf')
        
        for model_type in self.base_models:
            if model_type in feature_data and model_type in trained_models:
                n_samples = len(feature_data[model_type]['labels'])
                if n_samples < min_samples:
                    min_samples = n_samples
                    common_labels = feature_data[model_type]['labels']
        
        if common_labels is None:
            raise ValueError("No common samples found for stacking")
        
        # Create base estimators and prepare features
        for model_type in self.base_models:
            if model_type in trained_models:
                model_package = trained_models[model_type]
                base_estimators.append((
                    f'{model_type}_svm',
                    model_package['model']
                ))
                
                # Get scaled features
                features = feature_data[model_type]['features'][:min_samples]
                scaled_features = scalers[model_type].transform(features)
                stacking_features_list.append(scaled_features)
        
        if len(base_estimators) < 2:
            raise ValueError("Need at least 2 base models for stacking")
        
        # Create meta-classifier
        if self.meta_classifier_type == 'logistic_regression':
            meta_classifier = LogisticRegression(random_state=self.random_state)
        else:
            # Use SVM as meta-classifier (would need model_factory parameter)
            from sklearn.svm import LinearSVC
            meta_classifier = LinearSVC(C=1.0, max_iter=1000, random_state=self.random_state)
        
        # Create stacking classifier
        stacking_model = StackingClassifier(
            estimators=base_estimators,
            final_estimator=meta_classifier,
            cv=self.cv_folds,
            n_jobs=-1
        )
        
        # Prepare stacking features
        # For stacking, use combined features if available
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
            'base_models': self.base_models,
            'meta_classifier': self.meta_classifier_type,
            'cv_scores': cv_scores,
            'cv_mean': np.mean(cv_scores),
            'cv_std': np.std(cv_scores),
            'training_samples': min_samples,
            'feature_dimension': scaled_stacking_features.shape[1]
        }
        
        logger.info(f"Stacking model trained - CV F1: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
        
        return stacking_package
