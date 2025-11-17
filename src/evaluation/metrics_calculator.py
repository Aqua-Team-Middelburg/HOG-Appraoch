"""
Metrics calculation module for model evaluation.

This module provides pure metric computation functions separated from
evaluation orchestration logic, following the Single Responsibility Principle.
"""

import numpy as np
import logging
from typing import Dict, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score
)

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """
    Calculates classification and regression metrics for model evaluation.
    
    Provides pure computation functions with no side effects, making it
    easy to test and reuse across different evaluation contexts.
    """
    
    @staticmethod
    def calculate_classification_metrics(y_true: np.ndarray,
                                        y_pred: np.ndarray,
                                        y_prob: Optional[np.ndarray] = None,
                                        calculate_mape: bool = False,
                                        calculate_mae: bool = False) -> Dict[str, float]:
        """
        Calculate comprehensive classification metrics.
        
        Args:
            y_true: True binary labels
            y_pred: Predicted binary labels
            y_prob: Prediction probabilities (optional, for ROC/PR metrics)
            calculate_mape: Whether to calculate MAPE
            calculate_mae: Whether to calculate MAE
            
        Returns:
            Dictionary of metric names to values
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
        }
        
        # Add probability-based metrics if available
        if y_prob is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
                metrics['average_precision'] = average_precision_score(y_true, y_prob)
            except ValueError:
                # Handle cases where only one class is present
                logger.warning("Could not calculate ROC AUC - only one class present in y_true")
                metrics['roc_auc'] = None
                metrics['average_precision'] = None
        else:
            metrics['roc_auc'] = None
            metrics['average_precision'] = None
        
        # Custom metrics
        if calculate_mape:
            metrics['mape'] = MetricsCalculator.calculate_mape(y_true, y_pred)
        
        if calculate_mae:
            metrics['mae'] = MetricsCalculator.calculate_mae(y_true, y_pred)
        
        return metrics
    
    @staticmethod
    def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Mean Absolute Percentage Error.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            MAPE value as percentage
        """
        # For binary classification, treat as 0/1 and avoid division by zero
        mask = y_true != 0
        if np.sum(mask) == 0:
            return 0.0
        
        return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
    
    @staticmethod
    def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Mean Absolute Error.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            MAE value
        """
        return float(np.mean(np.abs(y_true - y_pred)))
    
    @staticmethod
    def calculate_detection_metrics(true_positives: int,
                                   false_positives: int,
                                   false_negatives: int) -> Dict[str, float]:
        """
        Calculate detection-specific metrics from counts.
        
        Args:
            true_positives: Number of correctly detected objects
            false_positives: Number of false detections
            false_negatives: Number of missed objects
            
        Returns:
            Dictionary with precision, recall, f1
        """
        # Avoid division by zero
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives
        }
    
    @staticmethod
    def calculate_image_level_metrics(image_predictions: Dict[str, Dict]) -> Dict[str, float]:
        """
        Calculate aggregated image-level metrics from per-image predictions.
        
        Args:
            image_predictions: Dictionary mapping image IDs to prediction dicts
                              containing 'predicted', 'actual', 'confidence'
            
        Returns:
            Dictionary with aggregated metrics
        """
        if not image_predictions:
            return {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0
            }
        
        # Extract predictions and ground truth
        y_pred = np.array([pred['predicted'] for pred in image_predictions.values()])
        y_true = np.array([pred['actual'] for pred in image_predictions.values()])
        
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0)
        }
