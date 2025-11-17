"""
Bounding box regressor for precise object localization.

This module implements bounding box regression to refine window-based detections
by predicting offsets and scale factors for more accurate localization.
"""

import numpy as np
import logging
from typing import Dict, Any, List, Tuple, Optional
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

from ..utils.config import ConfigLoader

logger = logging.getLogger(__name__)


class BoundingBoxRegressor:
    """
    Predicts bounding box refinements for detected windows.
    
    Trains regression models to predict:
    - Δx: Offset from window center to object center (X)
    - Δy: Offset from window center to object center (Y)
    - Δw: Scale factor for width (object_width / window_width)
    - Δh: Scale factor for height (object_height / window_height)
    """
    
    def __init__(self, config: ConfigLoader):
        """
        Initialize bounding box regressor.
        
        Args:
            config: Configuration loader instance
        """
        self.config = config
        
        # Load configuration
        bbox_config = config.get_section('training').get('bounding_box_regression', {})
        self.enabled = bbox_config.get('enabled', False)
        self.normalize_targets = bbox_config.get('normalize_targets', True)
        self.clip_targets = bbox_config.get('clip_targets', True)
        self.max_offset_ratio = bbox_config.get('max_offset_ratio', 0.5)
        
        # Regression model parameters
        self.regressor_type = bbox_config.get('regressor_type', 'svr')
        self.kernel = bbox_config.get('kernel', 'linear')
        self.C = bbox_config.get('C', 1.0)
        self.epsilon = bbox_config.get('epsilon', 0.1)
        self.independent_outputs = bbox_config.get('independent_outputs', True)
        
        # Trained components
        self.regressor = None
        self.scaler = None
        
        if self.enabled:
            logger.info(f"BBox regressor initialized: {self.regressor_type}, "
                       f"kernel={self.kernel}, normalize={self.normalize_targets}")
    
    def _calculate_regression_targets(self, window_bboxes: np.ndarray, 
                                     gt_bboxes: np.ndarray) -> np.ndarray:
        """
        Calculate regression targets (Δx, Δy, Δw, Δh) for training.
        
        Args:
            window_bboxes: Array of window bounding boxes (N, 4) as [x, y, w, h]
            gt_bboxes: Array of ground truth bounding boxes (N, 4) as [x, y, w, h]
            
        Returns:
            Regression targets (N, 4) as [Δx, Δy, Δw, Δh]
        """
        # Calculate window centers
        window_cx = window_bboxes[:, 0] + window_bboxes[:, 2] / 2
        window_cy = window_bboxes[:, 1] + window_bboxes[:, 3] / 2
        window_w = window_bboxes[:, 2]
        window_h = window_bboxes[:, 3]
        
        # Calculate ground truth centers
        gt_cx = gt_bboxes[:, 0] + gt_bboxes[:, 2] / 2
        gt_cy = gt_bboxes[:, 1] + gt_bboxes[:, 3] / 2
        gt_w = gt_bboxes[:, 2]
        gt_h = gt_bboxes[:, 3]
        
        # Calculate deltas
        if self.normalize_targets:
            # Normalize offsets by window size
            delta_x = (gt_cx - window_cx) / window_w
            delta_y = (gt_cy - window_cy) / window_h
        else:
            delta_x = gt_cx - window_cx
            delta_y = gt_cy - window_cy
        
        # Calculate scale factors
        delta_w = gt_w / window_w
        delta_h = gt_h / window_h
        
        # Clip extreme values if requested
        if self.clip_targets:
            delta_x = np.clip(delta_x, -self.max_offset_ratio, self.max_offset_ratio)
            delta_y = np.clip(delta_y, -self.max_offset_ratio, self.max_offset_ratio)
            delta_w = np.clip(delta_w, 0.1, 10.0)  # Reasonable scale range
            delta_h = np.clip(delta_h, 0.1, 10.0)
        
        # Stack targets
        targets = np.column_stack([delta_x, delta_y, delta_w, delta_h])
        
        return targets
    
    def _create_regressor(self) -> Any:
        """
        Create the regression model based on configuration.
        
        Returns:
            Configured regression model
        """
        if self.regressor_type == 'svr':
            # Support Vector Regression
            base_regressor = SVR(
                kernel=self.kernel,
                C=self.C,
                epsilon=self.epsilon
            )
        elif self.regressor_type == 'ridge':
            # Ridge Regression
            base_regressor = Ridge(alpha=1.0 / self.C)
        else:
            raise ValueError(f"Unknown regressor type: {self.regressor_type}")
        
        # Use multi-output wrapper for 4-dimensional output
        if self.independent_outputs:
            # Train separate regressor for each output dimension
            regressor = MultiOutputRegressor(base_regressor, n_jobs=-1)
        else:
            # Single multi-output regressor
            regressor = base_regressor
        
        return regressor
    
    def train(self, positive_features: np.ndarray, window_bboxes: np.ndarray,
             gt_bboxes: np.ndarray) -> Dict[str, Any]:
        """
        Train bounding box regression model.
        
        Args:
            positive_features: Feature vectors for positive windows (N, D)
            window_bboxes: Window bounding boxes (N, 4) as [x, y, w, h]
            gt_bboxes: Ground truth bounding boxes (N, 4) as [x, y, w, h]
            
        Returns:
            Dictionary with training results
        """
        if not self.enabled:
            logger.info("BBox regression disabled, skipping training")
            return {'enabled': False}
        
        logger.info(f"Training bbox regressor on {len(positive_features)} positive samples")
        
        # Calculate regression targets
        targets = self._calculate_regression_targets(window_bboxes, gt_bboxes)
        
        # Scale features
        self.scaler = StandardScaler()
        scaled_features = self.scaler.fit_transform(positive_features)
        
        # Create and train regressor
        self.regressor = self._create_regressor()
        self.regressor.fit(scaled_features, targets)
        
        # Calculate training statistics
        predictions = self.regressor.predict(scaled_features)
        mse = np.mean((predictions - targets) ** 2, axis=0)
        mae = np.mean(np.abs(predictions - targets), axis=0)
        
        results = {
            'enabled': True,
            'n_samples': len(positive_features),
            'feature_dim': positive_features.shape[1],
            'regressor_type': self.regressor_type,
            'mse_per_dim': mse.tolist(),
            'mae_per_dim': mae.tolist(),
            'mean_mse': float(np.mean(mse)),
            'mean_mae': float(np.mean(mae))
        }
        
        logger.info(f"BBox regressor trained - MSE: {results['mean_mse']:.4f}, "
                   f"MAE: {results['mean_mae']:.4f}")
        
        return results
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Predict bounding box refinements for features.
        
        Args:
            features: Feature vectors (N, D)
            
        Returns:
            Predictions (N, 4) as [Δx, Δy, Δw, Δh]
        """
        if not self.enabled or self.regressor is None:
            # Return zero refinements (no change)
            return np.zeros((len(features), 4))
        
        # Scale features
        scaled_features = self.scaler.transform(features)
        
        # Predict refinements
        predictions = self.regressor.predict(scaled_features)
        
        return predictions
    
    def refine_boxes(self, window_bboxes: np.ndarray, 
                    predictions: np.ndarray) -> np.ndarray:
        """
        Apply predicted refinements to window bounding boxes.
        
        Args:
            window_bboxes: Window bounding boxes (N, 4) as [x, y, w, h]
            predictions: Predicted refinements (N, 4) as [Δx, Δy, Δw, Δh]
            
        Returns:
            Refined bounding boxes (N, 4) as [x, y, w, h]
        """
        # Extract window properties
        window_cx = window_bboxes[:, 0] + window_bboxes[:, 2] / 2
        window_cy = window_bboxes[:, 1] + window_bboxes[:, 3] / 2
        window_w = window_bboxes[:, 2]
        window_h = window_bboxes[:, 3]
        
        # Extract predictions
        delta_x = predictions[:, 0]
        delta_y = predictions[:, 1]
        delta_w = predictions[:, 2]
        delta_h = predictions[:, 3]
        
        # Apply refinements
        if self.normalize_targets:
            refined_cx = window_cx + delta_x * window_w
            refined_cy = window_cy + delta_y * window_h
        else:
            refined_cx = window_cx + delta_x
            refined_cy = window_cy + delta_y
        
        refined_w = window_w * delta_w
        refined_h = window_h * delta_h
        
        # Convert back to [x, y, w, h] format
        refined_x = refined_cx - refined_w / 2
        refined_y = refined_cy - refined_h / 2
        
        # Ensure positive dimensions
        refined_w = np.maximum(refined_w, 1.0)
        refined_h = np.maximum(refined_h, 1.0)
        
        # Stack refined boxes
        refined_bboxes = np.column_stack([refined_x, refined_y, refined_w, refined_h])
        
        return refined_bboxes
    
    def refine_detections(self, detections: List[Dict[str, Any]], 
                         features: np.ndarray,
                         min_confidence: float = 0.5) -> List[Dict[str, Any]]:
        """
        Refine detected bounding boxes with predictions.
        
        Args:
            detections: List of detection dictionaries with 'bbox' and 'confidence'
            features: Feature vectors for detections (N, D)
            min_confidence: Minimum confidence to apply refinement
            
        Returns:
            List of detections with refined bounding boxes
        """
        if not self.enabled or len(detections) == 0:
            return detections
        
        # Extract window bboxes
        window_bboxes = np.array([d['bbox'] for d in detections])
        confidences = np.array([d.get('confidence', 1.0) for d in detections])
        
        # Predict refinements
        predictions = self.predict(features)
        
        # Apply refinements only to high-confidence detections
        refined_bboxes = window_bboxes.copy()
        high_conf_mask = confidences >= min_confidence
        
        if np.any(high_conf_mask):
            refined_bboxes[high_conf_mask] = self.refine_boxes(
                window_bboxes[high_conf_mask],
                predictions[high_conf_mask]
            )
        
        # Update detections with refined boxes
        refined_detections = []
        for i, detection in enumerate(detections):
            refined_det = detection.copy()
            refined_det['bbox'] = refined_bboxes[i].tolist()
            refined_det['original_bbox'] = window_bboxes[i].tolist()
            refined_det['bbox_refined'] = high_conf_mask[i]
            refined_detections.append(refined_det)
        
        return refined_detections
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get regressor configuration for saving.
        
        Returns:
            Configuration dictionary
        """
        return {
            'enabled': self.enabled,
            'regressor_type': self.regressor_type,
            'kernel': self.kernel,
            'C': self.C,
            'epsilon': self.epsilon,
            'normalize_targets': self.normalize_targets,
            'clip_targets': self.clip_targets,
            'max_offset_ratio': self.max_offset_ratio,
            'independent_outputs': self.independent_outputs
        }
    
    def save(self, filepath: str) -> None:
        """
        Save trained regressor to file.
        
        Args:
            filepath: Path to save regressor
        """
        import joblib
        
        if not self.enabled or self.regressor is None:
            logger.warning("No regressor to save (disabled or not trained)")
            return
        
        package = {
            'regressor': self.regressor,
            'scaler': self.scaler,
            'config': self.get_config()
        }
        
        joblib.dump(package, filepath)
        logger.info(f"BBox regressor saved to {filepath}")
    
    def load(self, filepath: str) -> None:
        """
        Load trained regressor from file.
        
        Args:
            filepath: Path to load regressor from
        """
        import joblib
        
        package = joblib.load(filepath)
        
        self.regressor = package['regressor']
        self.scaler = package['scaler']
        
        # Update config from saved values
        saved_config = package['config']
        self.enabled = saved_config['enabled']
        self.regressor_type = saved_config['regressor_type']
        
        logger.info(f"BBox regressor loaded from {filepath}")
