"""
Optuna Integration Module
========================

This module provides Optuna-based hyperparameter optimization for the nurdle detection pipeline.
"""

try:
    from .tuner import OptunaTuner
    __all__ = ['OptunaTuner']
except ImportError:
    # Handle case where optuna is not installed
    __all__ = []