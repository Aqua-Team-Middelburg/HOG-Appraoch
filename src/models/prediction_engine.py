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
    
    def predict_image(self, image: np.ndarray, iou_threshold: float = 0.3) -> Tuple[int, List[Tuple[float, float]]]:
        """
        Predict nurdles in an image using sliding window + IoU NMS after SVM.
        
        Pipeline: Sliding Window → SVM Classification → IoU NMS → SVR Regression
        
        Args:
            image: Input image as numpy array
            iou_threshold: IoU threshold for NMS filtering after SVM
            
        Returns:
            Tuple of (count, coordinates) where coordinates is list of (x, y) tuples
        """
        h, w = image.shape[:2]
        
        # Sliding window parameters
        window_size = self.feature_extractor.window_size
        window_stride = self.feature_extractor.window_stride
        
        positive_windows = []  # Store positive windows with confidence scores
        
        # Slide window across image
        for y in range(0, h - window_size[1] + 1, window_stride):
            for x in range(0, w - window_size[0] + 1, window_stride):
                window = image[y:y+window_size[1], x:x+window_size[0]]
                
                if window.shape[:2] == window_size:
                    # Extract features
                    features = self.feature_extractor.extract_hog_lbp_features(window)
                    
                    # SVM classification only
                    features_2d = features.reshape(1, -1)
                    is_nurdle = self.model_trainer.svm_classifier.predict(features_2d)[0]
                    
                    if is_nurdle:
                        # Get SVM confidence score
                        confidence = self.model_trainer.svm_classifier.predict_proba(features_2d)[0][1]
                        
                        # Create bounding box (x1, y1, x2, y2, confidence)
                        box = (
                            x,
                            y,
                            x + window_size[0],
                            y + window_size[1],
                            confidence
                        )
                        positive_windows.append((box, features, x, y))
        
        # Log window counts before NMS
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f"Windows before NMS: {len(positive_windows)}")
        
        # Apply IoU-based NMS on positive windows
        if not positive_windows:
            logger.info("No positive windows detected by SVM.")
            return 0, []
        
        boxes_for_nms = [pw[0] for pw in positive_windows]
        filtered_boxes = self.nms.apply_nms_with_boxes(boxes_for_nms, iou_threshold=iou_threshold)
        
        # Log window counts after NMS
        logger.debug(f"Windows after NMS: {len(filtered_boxes)}")
        logger.info(f"NMS filtering: {len(positive_windows)} → {len(filtered_boxes)} windows")
        
        # Handle empty NMS results
        if not filtered_boxes:
            logger.warning(f"No windows passed NMS filtering with IoU threshold {iou_threshold:.2f}")
            return 0, []
        
        # Create mapping from filtered boxes to original window data
        filtered_window_data = []
        for filtered_box in filtered_boxes:
            # Find matching original window
            for box, features, x, y in positive_windows:
                if box == filtered_box:
                    filtered_window_data.append((features, x, y))
                    break
        
        # Apply SVR only on NMS-filtered windows
        detections = []
        for features, x, y in filtered_window_data:
            # SVR regression for coordinate refinement
            features_2d = features.reshape(1, -1)
            offsets = self.model_trainer.svr_regressor.predict(features_2d)[0]
            offset_x, offset_y = offsets[0], offsets[1]
            
            window_center_x = x + window_size[0] / 2.0
            window_center_y = y + window_size[1] / 2.0
            
            final_x = window_center_x + offset_x
            final_y = window_center_y + offset_y
            
            detections.append((final_x, final_y))
        
        # Return count and coordinates
        return len(detections), detections
    
    def predict_image_from_path(self, image_path: str, data_loader, iou_threshold: float = 0.3) -> Tuple[int, List[Tuple[float, float]]]:
        """
        Predict nurdles from image path.
        
        Args:
            image_path: Path to image file
            data_loader: DataLoader instance for loading images
            iou_threshold: IoU threshold for NMS filtering
            
        Returns:
            Tuple of (count, coordinates)
        """
        image = data_loader.load_image(image_path)
        return self.predict_image(image, iou_threshold)
