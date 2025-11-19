"""
Model training for two-stage nurdle detection pipeline.

This module implements:
- SVM Classifier: Binary classification (is window positive?)
- SVR Regressor: Coordinate offset prediction (where is nurdle in window?)
"""

import numpy as np
import pickle
from pathlib import Path
from typing import List, Tuple, Dict, Any
import logging
from sklearn.svm import SVC, SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import KFold

from ..features import TrainingWindow


class ModelTrainer:
    """
    Trains SVM classifier and SVR regressor for two-stage detection.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize model trainer with configuration."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Models
        self.svm_classifier = None      # Binary classifier (all windows)
        self.svr_regressor = None       # Coordinate regressor (positive windows only)
        self.svr_ensemble = []          # Ensemble for confidence estimation
        
        # Configuration
        self.confidence_method = config.get('confidence', {}).get('svr_method', 'ensemble')
        self.ensemble_size = config.get('confidence', {}).get('svr_ensemble_size', 5)
        
    def train_models(self, window_batches) -> None:
        """
        Train both SVM classifier and SVR regressor.
        
        Args:
            window_batches: Iterator of training window batches
        """
        self.logger.info("Training models...")
        
        # Collect all windows
        all_windows = []
        for batch in window_batches:
            all_windows.extend(batch)
        
        if not all_windows:
            raise ValueError("No training windows generated")
        
        self.logger.info(f"Total training windows: {len(all_windows)}")
        
        # Separate positive and negative windows
        positive_windows = [w for w in all_windows if w.is_nurdle]
        negative_windows = [w for w in all_windows if not w.is_nurdle]
        
        self.logger.info(f"Positive windows: {len(positive_windows)}")
        self.logger.info(f"Negative windows: {len(negative_windows)}")
        
        # Train SVM classifier on all windows
        self.train_svm_classifier(all_windows)
        
        # Train SVR regressor on positive windows only
        self.train_svr_regressor(positive_windows)
        
        self.logger.info("Model training completed")
    
    def train_svm_classifier(self, all_windows: List[TrainingWindow]) -> None:
        """
        Train SVM classifier on ALL windows (positive + negative).
        
        Args:
            all_windows: List of all training windows
        """
        X = np.array([w.features for w in all_windows])
        y = np.array([w.is_nurdle for w in all_windows])
        
        self.svm_classifier = SVC(
            C=self.config.get('svm_c', 1.0),
            kernel=self.config.get('svm_kernel', 'rbf'),
            gamma=self.config.get('svm_gamma', 'scale'),
            probability=True,  # Enable confidence scores via predict_proba
            random_state=42
        )
        
        self.svm_classifier.fit(X, y)
        
        # Calculate training accuracy for logging
        train_accuracy = self.svm_classifier.score(X, y)
        self.logger.info(f"SVM Classifier trained on {len(X)} windows")
        self.logger.info(f"SVM training accuracy: {train_accuracy:.3f}")
    
    def train_svr_regressor(self, positive_windows: List[TrainingWindow]) -> None:
        """
        Train SVR regressor on POSITIVE windows only.
        
        Predicts coordinate offsets (offset_x, offset_y) from window center.
        
        Args:
            positive_windows: List of positive training windows with offset labels
        """
        if not positive_windows:
            raise ValueError("No positive windows for SVR training")
        
        X = np.array([w.features for w in positive_windows])
        y = np.array([[w.offset_x, w.offset_y] for w in positive_windows])
        
        # Create base SVR
        base_svr = SVR(
            C=self.config.get('svr_c', 1.0),
            kernel=self.config.get('svr_kernel', 'rbf'),
            gamma=self.config.get('svr_gamma', 'scale'),
            epsilon=self.config.get('svr_epsilon', 0.1)
        )
        
        # Use MultiOutputRegressor to predict both x and y offsets
        self.svr_regressor = MultiOutputRegressor(base_svr)
        self.svr_regressor.fit(X, y)
        
        self.logger.info(f"SVR Regressor trained on {len(X)} positive windows")
        
        # Train ensemble for confidence estimation
        if self.confidence_method == 'ensemble':
            self._train_svr_ensemble(X, y)
    
    def _train_svr_ensemble(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train ensemble of SVRs for confidence estimation via prediction variance.
        
        Args:
            X: Feature matrix
            y: Offset labels (N x 2)
        """
        self.logger.info(f"Training SVR ensemble ({self.ensemble_size} models) for confidence estimation...")
        
        kfold = KFold(n_splits=self.ensemble_size, shuffle=True, random_state=42)
        self.svr_ensemble = []
        
        for fold_idx, (train_idx, _) in enumerate(kfold.split(X)):
            base_svr = SVR(
                C=self.config.get('svr_c', 1.0),
                kernel=self.config.get('svr_kernel', 'rbf'),
                gamma=self.config.get('svr_gamma', 'scale'),
                epsilon=self.config.get('svr_epsilon', 0.1)
            )
            svr = MultiOutputRegressor(base_svr)
            svr.fit(X[train_idx], y[train_idx])
            self.svr_ensemble.append(svr)
        
        self.logger.info(f"SVR ensemble training completed")
    
    def predict_with_confidence(self, features: np.ndarray) -> Tuple[bool, float, float, float, float]:
        """
        Two-stage prediction with confidence scores.
        
        Args:
            features: Feature vector for a single window
            
        Returns:
            Tuple of (is_nurdle, svm_confidence, offset_x, offset_y, svr_confidence)
        """
        if self.svm_classifier is None or self.svr_regressor is None:
            raise ValueError("Models not trained yet")
        
        features_2d = features.reshape(1, -1)
        
        # Stage 1: SVM Classification
        svm_pred = self.svm_classifier.predict(features_2d)[0]
        svm_proba = self.svm_classifier.predict_proba(features_2d)[0]
        svm_confidence = float(svm_proba[1])  # Probability of positive class
        
        # Early return if classified as negative or low confidence
        svm_threshold = self.config.get('confidence', {}).get('svm_threshold', 0.5)
        if not svm_pred or svm_confidence < svm_threshold:
            return False, svm_confidence, 0.0, 0.0, 0.0
        
        # Stage 2: SVR Coordinate Refinement
        offsets = self.svr_regressor.predict(features_2d)[0]
        offset_x = float(offsets[0])
        offset_y = float(offsets[1])
        
        # Calculate SVR confidence
        svr_confidence = self._calculate_svr_confidence(features_2d, offsets)
        
        return True, svm_confidence, offset_x, offset_y, svr_confidence
    
    def _calculate_svr_confidence(self, features: np.ndarray, prediction: np.ndarray) -> float:
        """
        Calculate SVR confidence using ensemble variance.
        
        Lower variance = higher confidence in prediction.
        
        Args:
            features: Feature vector (1 x D)
            prediction: SVR prediction (not used, but kept for future methods)
            
        Returns:
            Confidence score between 0 and 1
        """
        if self.confidence_method == 'ensemble' and self.svr_ensemble:
            # Get predictions from all ensemble members
            predictions = []
            for svr in self.svr_ensemble:
                pred = svr.predict(features)[0]
                predictions.append(pred)
            
            # Calculate variance across ensemble predictions
            variance = np.var(predictions, axis=0)
            avg_variance = np.mean(variance)
            
            # Map variance to confidence (0 to 1)
            # High variance = low confidence, low variance = high confidence
            confidence = 1.0 / (1.0 + avg_variance)
            
            return float(confidence)
        
        elif self.confidence_method == 'simple':
            # Simple heuristic: confidence based on offset magnitude
            # Small offsets = high confidence (nurdle near window center)
            offset_magnitude = np.sqrt(prediction[0]**2 + prediction[1]**2)
            max_offset = self.config.get('windows', {}).get('size', [20, 20])[0] / 2.0
            confidence = 1.0 - min(offset_magnitude / max_offset, 1.0)
            return float(confidence)
        
        else:
            # Default: assume high confidence
            return 1.0
    
    def save_models(self, output_dir: str) -> None:
        """Save trained models to disk."""
        output_path = Path(output_dir)
        models_dir = output_path / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        
        if self.svm_classifier:
            svm_path = models_dir / "svm_classifier.pkl"
            with open(svm_path, 'wb') as f:
                pickle.dump(self.svm_classifier, f)
            self.logger.info(f"Saved SVM classifier to {svm_path}")
        
        if self.svr_regressor:
            svr_path = models_dir / "svr_regressor.pkl"
            with open(svr_path, 'wb') as f:
                pickle.dump(self.svr_regressor, f)
            self.logger.info(f"Saved SVR regressor to {svr_path}")
        
        if self.svr_ensemble:
            ensemble_path = models_dir / "svr_ensemble.pkl"
            with open(ensemble_path, 'wb') as f:
                pickle.dump(self.svr_ensemble, f)
            self.logger.info(f"Saved SVR ensemble to {ensemble_path}")
    
    def load_models(self, model_dir: str) -> None:
        """Load trained models from disk."""
        model_path = Path(model_dir)
        
        svm_path = model_path / "svm_classifier.pkl"
        if svm_path.exists():
            with open(svm_path, 'rb') as f:
                self.svm_classifier = pickle.load(f)
            self.logger.info(f"Loaded SVM classifier from {svm_path}")
        
        svr_path = model_path / "svr_regressor.pkl"
        if svr_path.exists():
            with open(svr_path, 'rb') as f:
                self.svr_regressor = pickle.load(f)
            self.logger.info(f"Loaded SVR regressor from {svr_path}")
        
        ensemble_path = model_path / "svr_ensemble.pkl"
        if ensemble_path.exists():
            with open(ensemble_path, 'rb') as f:
                self.svr_ensemble = pickle.load(f)
            self.logger.info(f"Loaded SVR ensemble from {ensemble_path}")
    
    @property
    def is_trained(self) -> bool:
        """Check if both models are trained."""
        return self.svm_classifier is not None and self.svr_regressor is not None