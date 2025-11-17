"""
Hard negative mining module for SVM training.

This module implements hard negative mining to reduce training set size
and improve model focus on difficult examples.
"""

import numpy as np
import logging
from typing import Tuple, Any
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class HardNegativeMiner:
    """
    Implements hard negative mining for SVM training.
    
    Uses a two-stage approach:
    1. Train initial model on balanced subset
    2. Collect false positives (hard negatives)
    3. Retrain on positives + hard negatives only
    """
    
    def __init__(self, config: dict, random_state: int = 42):
        """
        Initialize hard negative miner.
        
        Args:
            config: Hard negative mining configuration
            random_state: Random seed for reproducibility
        """
        self.config = config
        self.random_state = random_state
        self.enabled = config.get('enabled', False)
        
        # Configuration parameters
        self.initial_negative_ratio = config.get('initial_negative_ratio', 3.0)
        self.initial_sample_method = config.get('initial_sample_method', 'random')
        self.false_positive_threshold = config.get('false_positive_threshold', 0.0)
        self.max_hard_negatives = config.get('max_hard_negatives', 10000)
        self.hard_negative_ratio = config.get('hard_negative_ratio', 2.0)
        self.mining_iterations = config.get('mining_iterations', 1)
        
        logger.info(f"Hard negative miner initialized: enabled={self.enabled}")
    
    def mine_hard_negatives(self, 
                           positive_features: np.ndarray,
                           negative_features: np.ndarray,
                           positive_labels: np.ndarray,
                           negative_labels: np.ndarray,
                           model_factory) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform hard negative mining.
        
        Args:
            positive_features: All positive training features
            negative_features: All negative training features
            positive_labels: Labels for positive samples
            negative_labels: Labels for negative samples
            model_factory: Function that creates a new SVM model
            
        Returns:
            Tuple of (final_features, final_labels) with hard negatives
        """
        if not self.enabled:
            logger.info("Hard negative mining disabled. Using all samples.")
            final_features = np.vstack([positive_features, negative_features])
            final_labels = np.concatenate([positive_labels, negative_labels])
            return final_features, final_labels
        
        logger.info("Starting hard negative mining...")
        
        # Stage 1: Train initial model on balanced subset
        initial_model, scaler = self._train_initial_model(
            positive_features, negative_features,
            positive_labels, negative_labels,
            model_factory
        )
        
        # Stage 2: Collect hard negatives
        hard_negative_indices = self._collect_hard_negatives(
            initial_model, scaler, negative_features, len(positive_features)
        )
        
        # Stage 3: Create final training set
        hard_negatives = negative_features[hard_negative_indices]
        hard_neg_labels = negative_labels[hard_negative_indices]
        
        final_features = np.vstack([positive_features, hard_negatives])
        final_labels = np.concatenate([positive_labels, hard_neg_labels])
        
        reduction_pct = (1 - len(hard_negatives) / len(negative_features)) * 100
        logger.info(f"Hard negative mining complete. Training set reduced by {reduction_pct:.1f}%")
        logger.info(f"Final training set: {len(positive_features)} positives, {len(hard_negatives)} hard negatives")
        
        return final_features, final_labels
    
    def _train_initial_model(self,
                            positive_features: np.ndarray,
                            negative_features: np.ndarray,
                            positive_labels: np.ndarray,
                            negative_labels: np.ndarray,
                            model_factory) -> Tuple[Any, StandardScaler]:
        """
        Train initial SVM model on a balanced subset.
        
        Args:
            positive_features: All positive training features
            negative_features: All negative training features
            positive_labels: Labels for positive samples
            negative_labels: Labels for negative samples
            model_factory: Function that creates a new SVM model
            
        Returns:
            Tuple of (initial_model, scaler)
        """
        # Sample negatives for initial training
        num_negatives_needed = int(len(positive_features) * self.initial_negative_ratio)
        
        if num_negatives_needed >= len(negative_features):
            logger.warning(f"Not enough negatives for ratio {self.initial_negative_ratio}. Using all available.")
            sampled_negatives = negative_features
            sampled_neg_labels = negative_labels
        else:
            # Random sampling
            if self.initial_sample_method == 'random':
                np.random.seed(self.random_state)
                indices = np.random.choice(len(negative_features), num_negatives_needed, replace=False)
                sampled_negatives = negative_features[indices]
                sampled_neg_labels = negative_labels[indices]
            else:
                # Stratified or other methods can be added here
                np.random.seed(self.random_state)
                indices = np.random.choice(len(negative_features), num_negatives_needed, replace=False)
                sampled_negatives = negative_features[indices]
                sampled_neg_labels = negative_labels[indices]
        
        # Combine positive and sampled negative features
        initial_features = np.vstack([positive_features, sampled_negatives])
        initial_labels = np.concatenate([positive_labels, sampled_neg_labels])
        
        # Scale features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(initial_features)
        
        # Train initial model
        logger.info(f"Training initial model on {len(positive_features)} positives and {len(sampled_negatives)} negatives")
        initial_model = model_factory()
        initial_model.fit(scaled_features, initial_labels)
        
        return initial_model, scaler
    
    def _collect_hard_negatives(self,
                               initial_model: Any,
                               scaler: StandardScaler,
                               negative_features: np.ndarray,
                               positive_count: int) -> np.ndarray:
        """
        Collect hard negative samples (false positives) from initial model.
        
        Args:
            initial_model: Trained initial SVM model
            scaler: Feature scaler used for initial model
            negative_features: All negative training features
            positive_count: Number of positive samples (for ratio calculation)
            
        Returns:
            Array of hard negative feature indices
        """
        # Scale negative features
        scaled_negatives = scaler.transform(negative_features)
        
        # Get predictions
        predictions = initial_model.predict(scaled_negatives)
        
        # Find false positives (negatives predicted as positive)
        false_positive_mask = predictions == 1
        false_positive_indices = np.where(false_positive_mask)[0]
        
        logger.info(f"Found {len(false_positive_indices)} false positives out of {len(negative_features)} negatives")
        
        if len(false_positive_indices) == 0:
            logger.warning("No false positives found. Using random hard negatives.")
            # Fallback: use decision function to find negatives closest to decision boundary
            if hasattr(initial_model, 'decision_function'):
                decision_scores = initial_model.decision_function(scaled_negatives)
                # Get negatives with highest decision scores (closest to positive class)
                sorted_indices = np.argsort(-decision_scores)  # Sort descending
            else:
                # Random fallback
                sorted_indices = np.random.permutation(len(negative_features))
            false_positive_indices = sorted_indices[:int(positive_count * 2)]
        
        # Determine how many hard negatives to keep
        num_hard_negatives = min(
            int(positive_count * self.hard_negative_ratio),
            self.max_hard_negatives,
            len(false_positive_indices)
        )
        
        # If we have decision scores, sort by confidence
        if hasattr(initial_model, 'decision_function'):
            fp_features = negative_features[false_positive_indices]
            fp_scaled = scaler.transform(fp_features)
            decision_scores = initial_model.decision_function(fp_scaled)
            
            # Sort by decision score (most confident mistakes first)
            sorted_fp_indices = np.argsort(-decision_scores)
            hard_negative_indices = false_positive_indices[sorted_fp_indices[:num_hard_negatives]]
        else:
            # Random selection from false positives
            if len(false_positive_indices) > num_hard_negatives:
                hard_negative_indices = np.random.choice(false_positive_indices, num_hard_negatives, replace=False)
            else:
                hard_negative_indices = false_positive_indices
        
        logger.info(f"Selected {len(hard_negative_indices)} hard negatives for final training")
        return hard_negative_indices
