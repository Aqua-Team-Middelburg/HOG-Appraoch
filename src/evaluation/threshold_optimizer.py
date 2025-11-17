"""
Threshold optimization module for finding optimal decision thresholds.

This module implements threshold optimization using validation curves
to maximize various metrics (F1, precision, recall, accuracy).
"""

import numpy as np
import logging
from typing import Dict, Any

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

from ..utils.config import ConfigLoader

logger = logging.getLogger(__name__)


class ThresholdOptimizer:
    """
    Optimizes detection thresholds for each model using validation curves.
    
    Supports optimization for different metrics:
    - F1-score (balanced precision and recall)
    - Precision (minimize false positives)
    - Recall (minimize false negatives)
    - Accuracy (overall correctness)
    """
    
    def __init__(self, config: ConfigLoader):
        """
        Initialize threshold optimizer.
        
        Args:
            config: Configuration loader instance
        """
        self.config = config
    
    def optimize_threshold(self, 
                          y_true: np.ndarray, 
                          y_scores: np.ndarray,
                          metric: str = 'f1') -> Dict[str, Any]:
        """
        Find optimal decision threshold using validation curve.
        
        Args:
            y_true: True binary labels
            y_scores: Model decision scores or probabilities
            metric: Optimization metric ('f1', 'precision', 'recall', 'accuracy')
            
        Returns:
            Dictionary with optimal threshold and metrics
        """
        if y_scores is None:
            return {
                'optimal_threshold': 0.0,
                'optimal_score': 0.0,
                'thresholds': [0.0],
                'scores': [0.0],
                'method': 'default'
            }
        
        # Generate threshold range
        min_score = np.min(y_scores)
        max_score = np.max(y_scores)
        thresholds = np.linspace(min_score, max_score, 100)
        
        scores = []
        
        for threshold in thresholds:
            y_pred = (y_scores >= threshold).astype(int)
            
            if metric == 'f1':
                score = f1_score(y_true, y_pred, zero_division=0)
            elif metric == 'precision':
                score = precision_score(y_true, y_pred, zero_division=0)
            elif metric == 'recall':
                score = recall_score(y_true, y_pred, zero_division=0)
            elif metric == 'accuracy':
                score = accuracy_score(y_true, y_pred)
            else:
                score = f1_score(y_true, y_pred, zero_division=0)
            
            scores.append(score)
        
        # Find optimal threshold
        best_idx = np.argmax(scores)
        optimal_threshold = thresholds[best_idx]
        optimal_score = scores[best_idx]
        
        return {
            'optimal_threshold': float(optimal_threshold),
            'optimal_score': float(optimal_score),
            'thresholds': thresholds.tolist(),
            'scores': scores,
            'method': 'validation_curve'
        }
