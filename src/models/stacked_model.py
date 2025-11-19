"""
Stacked Model Implementation for Two-Stage Detection
===================================================

Meta-learner that combines SVM and SVR predictions using their confidence scores
to make final detection decisions.

Meta-features: [svm_pred, svm_conf, offset_x, offset_y, svr_conf]
Meta-output: Final classification (keep/reject window)
Coordinates: Uses SVR offsets directly
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from typing import Tuple, List, Dict, Any
import logging


class StackedNurdleDetector:
    """
    Stacked model combining SVM classifier and SVR regressor predictions.
    
    Uses confidence scores from both models as additional features for meta-learning.
    """
    
    def __init__(self, model_trainer, config: Dict[str, Any] = None, logger=None):
        """
        Initialize stacked model.
        
        Args:
            model_trainer: Trained ModelTrainer instance with SVM and SVR
            config: Configuration dictionary
            logger: Optional logger instance
        """
        self.model_trainer = model_trainer
        self.config = config or {}
        self.logger = logger or logging.getLogger(__name__)
        
        # Meta-learner for final classification decision
        meta_learner_type = self.config.get('stacking', {}).get('meta_learner', 'logistic_regression')
        
        if meta_learner_type == 'random_forest':
            self.meta_classifier = RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=42
            )
        else:  # Default: logistic_regression
            self.meta_classifier = LogisticRegression(
                random_state=42,
                max_iter=1000
            )
        
        self.is_trained = False
    
    def train_stacking_model(self, training_windows: List) -> None:
        """
        Train meta-learner using base model predictions on training data.
        
        Args:
            training_windows: List of TrainingWindow objects used for training base models
        """
        self.logger.info("Training stacking meta-learner...")
        
        meta_features = []
        meta_labels = []
        
        for window in training_windows:
            # Get base model predictions
            is_nurdle, svm_conf, offset_x, offset_y, svr_conf = \
                self.model_trainer.predict_with_confidence(window.features)
            
            # Create meta-features: [svm_pred, svm_conf, offset_x, offset_y, svr_conf]
            meta_feat = [
                float(is_nurdle),   # SVM binary prediction
                svm_conf,           # SVM confidence
                offset_x,           # SVR offset x
                offset_y,           # SVR offset y
                svr_conf if is_nurdle else 0.0  # SVR confidence (0 if SVM said negative)
            ]
            meta_features.append(meta_feat)
            meta_labels.append(window.is_nurdle)  # Ground truth label
        
        X_meta = np.array(meta_features)
        y_meta = np.array(meta_labels)
        
        # Train meta-classifier
        self.meta_classifier.fit(X_meta, y_meta)
        self.is_trained = True
        
        # Log meta-learner information
        train_accuracy = self.meta_classifier.score(X_meta, y_meta)
        self.logger.info(f"Stacking model trained on {len(X_meta)} samples")
        self.logger.info(f"Meta-learner training accuracy: {train_accuracy:.3f}")
        
        if hasattr(self.meta_classifier, 'coef_'):
            self.logger.info(f"Meta-learner feature weights: {self.meta_classifier.coef_[0]}")
        elif hasattr(self.meta_classifier, 'feature_importances_'):
            self.logger.info(f"Meta-learner feature importances: {self.meta_classifier.feature_importances_}")
    
    def predict_with_confidence(self, features: np.ndarray) -> Tuple[bool, float, float, float]:
        """
        Stacked prediction using meta-learner.
        
        Args:
            features: Window feature vector
            
        Returns:
            Tuple of (is_nurdle, confidence, offset_x, offset_y)
        """
        if not self.is_trained:
            raise ValueError("Stacking model must be trained before prediction")
        
        # Get base model predictions
        is_nurdle_base, svm_conf, offset_x, offset_y, svr_conf = \
            self.model_trainer.predict_with_confidence(features)
        
        # Create meta-features
        meta_feat = np.array([[
            float(is_nurdle_base),
            svm_conf,
            offset_x,
            offset_y,
            svr_conf if is_nurdle_base else 0.0
        ]])
        
        # Meta-learner final decision
        final_pred = self.meta_classifier.predict(meta_feat)[0]
        final_conf_proba = self.meta_classifier.predict_proba(meta_feat)[0]
        final_conf = float(final_conf_proba[1])  # Probability of positive class
        
        # Use base SVR offsets for coordinates
        return bool(final_pred), final_conf, float(offset_x), float(offset_y)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the stacked model components."""
        return {
            'meta_learner_type': type(self.meta_classifier).__name__,
            'is_trained': self.is_trained,
            'meta_features': ['svm_pred', 'svm_conf', 'offset_x', 'offset_y', 'svr_conf'],
            'base_svm': type(self.model_trainer.svm_classifier).__name__ if self.model_trainer.svm_classifier else None,
            'base_svr': type(self.model_trainer.svr_regressor).__name__ if self.model_trainer.svr_regressor else None,
        }
