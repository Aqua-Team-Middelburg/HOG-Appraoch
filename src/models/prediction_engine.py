"""
Prediction Engine for Nurdle Detection
======================================

Single-stage SVM multi-class count prediction using whole-image features.
"""

import numpy as np
from typing import Tuple, List
from pathlib import Path


class PredictionEngine:
    """
    Prediction engine for single-stage SVM multi-class count prediction.
    Uses whole-image features for count prediction.
    """

    def __init__(self, feature_extractor, model_trainer):
        """
        Initialize prediction engine.
        Args:
            feature_extractor: FeatureExtractor instance for extracting features
            model_trainer: ModelTrainer instance with trained SVM classifier
        """
        self.feature_extractor = feature_extractor
        self.model_trainer = model_trainer

    def predict_image(self, image: np.ndarray) -> int:
        """
        Predict nurdle count in an image using SVM classifier.
        Args:
            image: Input image as numpy array
        Returns:
            Predicted nurdle count (int)
        """
        features = self.feature_extractor.extract_image_features(image)
        features_2d = features.reshape(1, -1)
        pred_count = self.model_trainer.svm_classifier.predict(features_2d)[0]
        return int(pred_count)

    def predict_image_from_path(self, image_path: str, data_loader) -> int:
        """
        Predict nurdle count from image path.
        Args:
            image_path: Path to image file
            data_loader: DataLoader instance for loading images
        Returns:
            Predicted nurdle count (int)
        """
        image = data_loader.load_image(image_path)
        return self.predict_image(image)
