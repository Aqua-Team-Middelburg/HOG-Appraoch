"""
Evaluation Module for Nurdle Detection Pipeline
==============================================

This module provides evaluation functionality for the nurdle detection pipeline,
including metrics calculation, coordinate matching, and results reporting.
"""

from .evaluator import ModelEvaluator
from .metrics import EvaluationMetrics

__all__ = [
    'ModelEvaluator',
    'EvaluationMetrics'
]