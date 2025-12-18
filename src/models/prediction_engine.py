"""
Prediction Engine for Ensemble Nurdle Detection
================================================

Combines predictions from multiple feature-specific SVR models via meta-learner.
"""

import numpy as np
from typing import Tuple, Dict


class PredictionEngine:
    """
    Prediction engine for ensemble SVR prediction.
    
    Combines predictions from HOG, LBP, and Mask Stats base models
    using a trained meta-learner.
    """

    def __init__(self, feature_extractor, model_trainer):
        """
        Initialize prediction engine.
        Args:
            feature_extractor: FeatureExtractor instance for extracting features
            model_trainer: ModelTrainer instance with trained ensemble models
        """
        self.feature_extractor = feature_extractor
        self.model_trainer = model_trainer

    def predict_image_ensemble(
        self,
        image: np.ndarray,
        mask: np.ndarray
    ) -> Tuple[float, Dict[str, float]]:
        """
        Predict nurdle count using ensemble of feature-specific models.
        
        Args:
            image: Input image as numpy array (BGR)
            mask: Binary segmentation mask
            
        Returns:
            Tuple of (ensemble_prediction, dict_of_base_predictions)
            dict_of_base_predictions maps model names to their individual predictions
        """
        # Extract separate feature vectors
        hog_features = self.feature_extractor.extract_hog_features(image)
        lbp_features = self.feature_extractor.extract_lbp_features(image)
        mask_stats = self.feature_extractor.extract_mask_stats_features(mask)
        
        feature_dict = {}
        if len(hog_features) > 0:
            feature_dict['hog'] = hog_features
        if len(lbp_features) > 0:
            feature_dict['lbp'] = lbp_features
        if len(mask_stats) > 0:
            feature_dict['mask_stats'] = mask_stats
        
        # Get base model predictions
        base_predictions = self.model_trainer.predict_ensemble(feature_dict)
        
        # Get ensemble prediction
        ensemble_pred = self.model_trainer.predict_ensemble_meta(base_predictions)
        
        return float(ensemble_pred), base_predictions
