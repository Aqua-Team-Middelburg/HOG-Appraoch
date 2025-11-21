"""
Prediction Engine for Nurdle Detection
======================================

Handles sliding window prediction with NMS for inference.
"""

import numpy as np
from typing import Tuple, List
from pathlib import Path


class PredictionEngine:
    """
    Prediction engine that performs sliding window inference with NMS.
    
    Combines feature extraction, model prediction, and NMS filtering
    for end-to-end nurdle detection on images.
    """
    
    def __init__(self, feature_extractor, model_trainer, nms):
        """
        Initialize prediction engine.
        
        Args:
            feature_extractor: FeatureExtractor instance for extracting features
            model_trainer: ModelTrainer instance with trained models
            nms: NonMaximumSuppression instance for filtering detections
        """
        self.feature_extractor = feature_extractor
        self.model_trainer = model_trainer
        self.nms = nms
    
    def predict_image(self, image: np.ndarray, nms_distance: float = 20.0) -> Tuple[int, List[Tuple[float, float]]]:
        """
        Predict nurdles in an image using sliding window + NMS.
        
        Args:
            image: Input image as numpy array
            nms_distance: Distance threshold for NMS filtering
            
        Returns:
            Tuple of (count, coordinates) where coordinates is list of (x, y) tuples
        """
        h, w = image.shape[:2]
        
        # Sliding window parameters
        window_size = self.feature_extractor.window_size
        window_stride = self.feature_extractor.window_stride
        
        detections = []
        offsets = []  # Store SVR offsets for NMS ranking
        
        # Slide window across image
        for y in range(0, h - window_size[1] + 1, window_stride):
            for x in range(0, w - window_size[0] + 1, window_stride):
                window = image[y:y+window_size[1], x:x+window_size[0]]
                
                if window.shape[:2] == window_size:
                    # Extract features
                    features = self.feature_extractor.extract_hog_lbp_features(window)
                    
                    # Two-stage prediction: SVM + SVR
                    is_nurdle, offset_x, offset_y = self.model_trainer.predict(features)
                    
                    if is_nurdle:
                        window_center_x = x + window_size[0] / 2.0
                        window_center_y = y + window_size[1] / 2.0
                        
                        final_x = window_center_x + offset_x
                        final_y = window_center_y + offset_y
                        
                        detections.append((final_x, final_y))
                        offsets.append((offset_x, offset_y))  # Store offset for NMS
        
        # Apply offset-magnitude-based NMS
        filtered_detections = self.nms.apply_nms(
            detections, 
            offsets=offsets,
            min_distance=nms_distance
        )
        
        # Return count and coordinates
        return len(filtered_detections), filtered_detections
    
    def predict_image_from_path(self, image_path: str, data_loader, nms_distance: float = 20.0) -> Tuple[int, List[Tuple[float, float]]]:
        """
        Predict nurdles from image path.
        
        Args:
            image_path: Path to image file
            data_loader: DataLoader instance for loading images
            nms_distance: Distance threshold for NMS filtering
            
        Returns:
            Tuple of (count, coordinates)
        """
        image = data_loader.load_image(image_path)
        return self.predict_image(image, nms_distance)
