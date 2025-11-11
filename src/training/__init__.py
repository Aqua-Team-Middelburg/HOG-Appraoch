"""
Training module for nurdle detection pipeline.

This module provides comprehensive SVM training capabilities including:
- Single model training (HOG, LBP, Combined features)
- Hyperparameter optimization (Optuna or GridSearch)
- Cross-validation evaluation
- Stacking ensemble training
- Model persistence and loading
"""

from .svm_trainer import SVMTrainer

__all__ = ['SVMTrainer']