"""
Evaluation module for nurdle detection pipeline.

This module provides comprehensive evaluation capabilities including:
- Window-level and image-level performance metrics
- Model comparison and ranking
- ROC and Precision-Recall curve analysis
- Confusion matrix visualization
- Statistical analysis and reporting
"""

from .evaluator import ModelEvaluator

__all__ = ['ModelEvaluator']