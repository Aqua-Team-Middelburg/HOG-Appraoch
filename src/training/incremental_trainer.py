"""
Incremental training module for memory-efficient SVM training.

This module implements incremental (online) learning using SGDClassifier
for training on large datasets that don't fit in memory.
"""

import numpy as np
import logging
from typing import Dict, Any, Generator, Tuple, Optional
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold

logger = logging.getLogger(__name__)


class IncrementalTrainer:
    """
    Trains SVM models incrementally using mini-batch gradient descent.
    
    Uses SGDClassifier with partial_fit for memory-efficient training
    on large datasets via generators/batches.
    """
    
    def __init__(self, config: dict, random_state: int = 42):
        """
        Initialize incremental trainer.
        
        Args:
            config: Incremental training configuration
            random_state: Random seed for reproducibility
        """
        self.config = config
        self.random_state = random_state
        self.enabled = config.get('enabled', False)
        
        # Configuration parameters
        self.partial_fit_batch_size = config.get('partial_fit_batch_size', 5000)
        self.n_epochs = config.get('n_epochs', 5)
        self.learning_rate = config.get('learning_rate', 'optimal')
        
        logger.info(f"Incremental trainer initialized: enabled={self.enabled}, "
                   f"batch_size={self.partial_fit_batch_size}, epochs={self.n_epochs}")
    
    def train_incremental(self,
                         features: np.ndarray,
                         labels: np.ndarray,
                         feature_type: str) -> Dict[str, Any]:
        """
        Train model incrementally using SGDClassifier.
        
        Args:
            features: Training feature matrix (can be very large)
            labels: Training labels
            feature_type: Type of features being trained
            
        Returns:
            Dictionary containing trained model and training results
        """
        if not self.enabled:
            logger.info("Incremental training disabled")
            return {}
        
        logger.info(f"Training {feature_type} model incrementally on {len(features)} samples")
        
        # Scale features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        
        # Create SGDClassifier
        model = SGDClassifier(
            loss='hinge',  # SVM loss
            penalty='l2',
            alpha=0.0001,
            max_iter=1000,
            tol=1e-3,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        # Train incrementally over multiple epochs
        n_samples = len(features)
        batch_size = self.partial_fit_batch_size
        classes = np.unique(labels)
        
        logger.info(f"Training over {self.n_epochs} epochs with batch size {batch_size}")
        
        for epoch in range(self.n_epochs):
            # Shuffle data each epoch
            indices = np.random.permutation(n_samples)
            shuffled_features = scaled_features[indices]
            shuffled_labels = labels[indices]
            
            # Process in batches
            n_batches = int(np.ceil(n_samples / batch_size))
            
            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, n_samples)
                
                batch_features = shuffled_features[start_idx:end_idx]
                batch_labels = shuffled_labels[start_idx:end_idx]
                
                # Partial fit
                model.partial_fit(batch_features, batch_labels, classes=classes)
            
            logger.debug(f"Epoch {epoch + 1}/{self.n_epochs} complete")
        
        # Evaluate with cross-validation on a subset (for performance)
        eval_size = min(10000, n_samples)
        eval_indices = np.random.choice(n_samples, eval_size, replace=False)
        eval_features = scaled_features[eval_indices]
        eval_labels = labels[eval_indices]
        
        cv_scores = cross_val_score(
            model, eval_features, eval_labels,
            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state),
            scoring='f1'
        )
        
        # Create model package
        model_package = {
            'model': model,
            'scaler': scaler,
            'feature_type': feature_type,
            'training_method': 'incremental',
            'cv_scores': cv_scores,
            'cv_mean': np.mean(cv_scores),
            'cv_std': np.std(cv_scores),
            'training_samples': n_samples,
            'feature_dimension': features.shape[1],
            'n_epochs': self.n_epochs,
            'batch_size': batch_size
        }
        
        logger.info(f"{feature_type} incremental model trained - CV F1: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
        
        return model_package
    
    def train_from_generator(self,
                            feature_generator: Generator[Tuple[np.ndarray, np.ndarray], None, None],
                            feature_type: str,
                            total_samples: Optional[int] = None) -> Dict[str, Any]:
        """
        Train model from a feature generator for maximum memory efficiency.
        
        Args:
            feature_generator: Generator yielding (features, labels) batches
            feature_type: Type of features being trained
            total_samples: Optional total sample count for logging
            
        Returns:
            Dictionary containing trained model and training results
        """
        if not self.enabled:
            logger.info("Incremental training disabled")
            return {}
        
        logger.info(f"Training {feature_type} model from generator")
        
        # Create SGDClassifier
        model = SGDClassifier(
            loss='hinge',
            penalty='l2',
            alpha=0.0001,
            max_iter=1000,
            tol=1e-3,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        # We need to fit a scaler on first batch
        scaler = None
        classes = None
        batch_count = 0
        
        # Train over epochs
        for epoch in range(self.n_epochs):
            logger.info(f"Epoch {epoch + 1}/{self.n_epochs}")
            
            for batch_features, batch_labels in feature_generator:
                # Initialize scaler on first batch
                if scaler is None:
                    scaler = StandardScaler()
                    scaler.fit(batch_features)
                    classes = np.unique(batch_labels)
                
                # Scale features
                scaled_features = scaler.transform(batch_features)
                
                # Partial fit
                model.partial_fit(scaled_features, batch_labels, classes=classes)
                batch_count += 1
        
        # Create model package (without CV since we used generator)
        model_package = {
            'model': model,
            'scaler': scaler,
            'feature_type': feature_type,
            'training_method': 'incremental_generator',
            'training_samples': total_samples if total_samples else 'unknown',
            'n_epochs': self.n_epochs,
            'batches_processed': batch_count
        }
        
        logger.info(f"{feature_type} incremental model trained from generator - {batch_count} batches processed")
        
        return model_package
