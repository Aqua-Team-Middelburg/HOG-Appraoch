"""
Feature extraction for nurdle detection pipeline.

This module provides combined HOG and LBP feature extraction from image windows,
optimized for memory efficiency and computational performance.
"""

import numpy as np
import cv2
from typing import Dict, Any, Iterator, List, Tuple
from dataclasses import dataclass
import logging
from skimage.feature import hog, local_binary_pattern

from ..data import ImageAnnotation, NurdleAnnotation


@dataclass 
class TrainingWindow:
    """Single training window with features and labels."""
    features: np.ndarray
    is_nurdle: bool           # For SVM classifier (binary label)
    offset_x: float           # For SVR regressor (nurdle_center_x - window_center_x)
    offset_y: float           # For SVR regressor (nurdle_center_y - window_center_y)
    image_id: str
    window_x: int             # Window top-left x position
    window_y: int             # Window top-left y position
    window_center_x: float    # For inference coordinate calculation
    window_center_y: float    # For inference coordinate calculation


class FeatureExtractor:
    """
    Extracts combined HOG and LBP features from image windows.
    
    This class provides efficient feature extraction for training and inference,
    with integrated sliding window processing for memory efficiency.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize feature extractor with configuration."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Configuration parameters
        self.window_size = tuple(config.get('size', [20, 20]))
        self.window_stride = config.get('stride', 8)
        self.hog_cell_size = tuple(config.get('hog_cell_size', [4, 4]))
        self.negative_positive_ratio = config.get('negative_positive_ratio', 3.0)
        self.min_distance_from_nurdles = config.get('min_distance_from_nurdles', 25)
    
    def extract_hog_lbp_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract combined HOG and LBP features from image window.
        
        This replaces separate feature extraction with a single combined approach.
        
        Args:
            image: Image window to extract features from
            
        Returns:
            Combined feature vector
        """
        # Ensure window is correct size
        if image.shape[:2] != self.window_size:
            image = cv2.resize(image, self.window_size)
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # HOG Features
        hog_features = hog(
            gray,
            orientations=9,
            pixels_per_cell=self.hog_cell_size,
            cells_per_block=(2, 2),
            block_norm='L2-Hys',
            feature_vector=True
        )
        
        # LBP Features  
        lbp = local_binary_pattern(gray, 8, 1, method='uniform')
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 9))
        lbp_features = lbp_hist.astype(float)
        
        # Combine features
        combined_features = np.concatenate([hog_features, lbp_features])
        
        return combined_features
    
    def generate_training_windows_batch(self, 
                                      annotations: List[ImageAnnotation],
                                      load_image_func,
                                      transform_coords_func=None) -> Iterator[List[TrainingWindow]]:
        """
        Generate training windows using sliding window approach.
        
        This method slides a window across the entire image and checks if each window
        contains any nurdle center coordinates. Windows containing nurdles are labeled
        as positive, and a ratio-based number of negative windows (far from nurdles)
        are randomly sampled.
        
        Args:
            annotations: List of image annotations to process (with original coordinates)
            load_image_func: Function to load and normalize images
            transform_coords_func: Optional function to transform annotation coordinates
                                  to match normalized image space
            
        Yields:
            Batches of training windows
        """
        batch_windows = []
        
        for annotation in annotations:
            try:
                # Load original image to get scale factors
                original_image = cv2.imread(annotation.image_path)
                if original_image is None:
                    self.logger.error(f"Could not load image: {annotation.image_path}")
                    continue
                
                # Load and normalize image
                image = load_image_func(annotation.image_path)
                h, w = image.shape[:2]
                orig_h, orig_w = original_image.shape[:2]
                
                # Calculate coordinate transformation scale
                scale_x = w / orig_w
                scale_y = h / orig_h
                
                # Transform annotation coordinates to normalized space
                if transform_coords_func:
                    transformed_annotation = transform_coords_func(annotation, original_image)
                else:
                    # Manual transformation
                    transformed_nurdles = [
                        NurdleAnnotation(x=n.x * scale_x, y=n.y * scale_y) 
                        for n in annotation.nurdles
                    ]
                    transformed_annotation = ImageAnnotation(
                        image_path=annotation.image_path,
                        nurdles=transformed_nurdles,
                        image_id=annotation.image_id
                    )
                
                # Convert transformed nurdle coordinates to numpy array
                if transformed_annotation.nurdles:
                    nurdle_coords = np.array([[n.x, n.y] for n in transformed_annotation.nurdles])
                else:
                    nurdle_coords = np.empty((0, 2))
                
                self.logger.debug(f"Image {annotation.image_id}: Original {orig_w}x{orig_h} -> "
                                f"Normalized {w}x{h}, Scale: {scale_x:.4f}x{scale_y:.4f}")
                
                # Slide window across entire image to find all positive windows
                positive_windows = []
                potential_negative_windows = []
                
                for y in range(0, h - self.window_size[1] + 1, self.window_stride):
                    for x in range(0, w - self.window_size[0] + 1, self.window_stride):
                        # Calculate window center (use float for precise offset calculation)
                        window_center_x = x + self.window_size[0] / 2.0
                        window_center_y = y + self.window_size[1] / 2.0
                        
                        # Check if any nurdle center is inside window bounds
                        nurdles_in_window = []
                        if len(nurdle_coords) > 0:
                            for nurdle_x, nurdle_y in nurdle_coords:
                                if (x <= nurdle_x < x + self.window_size[0] and 
                                    y <= nurdle_y < y + self.window_size[1]):
                                    nurdles_in_window.append((nurdle_x, nurdle_y))
                        
                        # Extract window
                        window = image[y:y+self.window_size[1], x:x+self.window_size[0]]
                        
                        if window.shape[:2] == self.window_size:
                            features = self.extract_hog_lbp_features(window)
                            
                            if len(nurdles_in_window) > 0:
                                # POSITIVE WINDOW - Find nearest nurdle to window center
                                distances = [
                                    np.sqrt((nx - window_center_x)**2 + (ny - window_center_y)**2)
                                    for nx, ny in nurdles_in_window
                                ]
                                nearest_idx = np.argmin(distances)
                                nearest_nurdle = nurdles_in_window[nearest_idx]
                                
                                # Calculate offset from window center to nearest nurdle
                                offset_x = nearest_nurdle[0] - window_center_x
                                offset_y = nearest_nurdle[1] - window_center_y
                                
                                positive_windows.append(TrainingWindow(
                                    features=features,
                                    is_nurdle=True,
                                    offset_x=offset_x,
                                    offset_y=offset_y,
                                    image_id=annotation.image_id,
                                    window_x=x,
                                    window_y=y,
                                    window_center_x=window_center_x,
                                    window_center_y=window_center_y
                                ))
                            else:
                                # NEGATIVE WINDOW (only if far enough from all nurdles)
                                if len(nurdle_coords) > 0:
                                    distances = np.sqrt(
                                        (nurdle_coords[:, 0] - window_center_x)**2 + 
                                        (nurdle_coords[:, 1] - window_center_y)**2
                                    )
                                    min_distance = np.min(distances)
                                    
                                    if min_distance >= self.min_distance_from_nurdles:
                                        potential_negative_windows.append(TrainingWindow(
                                            features=features,
                                            is_nurdle=False,
                                            offset_x=0.0,  # Unused for negative windows
                                            offset_y=0.0,
                                            image_id=annotation.image_id,
                                            window_x=x,
                                            window_y=y,
                                            window_center_x=window_center_x,
                                            window_center_y=window_center_y
                                        ))
                                else:
                                    # No nurdles in image - all windows are negative
                                    potential_negative_windows.append(TrainingWindow(
                                        features=features,
                                        is_nurdle=False,
                                        offset_x=0.0,
                                        offset_y=0.0,
                                        image_id=annotation.image_id,
                                        window_x=x,
                                        window_y=y,
                                        window_center_x=window_center_x,
                                        window_center_y=window_center_y
                                    ))
                
                # Sample negative windows based on ratio
                n_negative = int(len(positive_windows) * self.negative_positive_ratio)
                
                if len(potential_negative_windows) > n_negative:
                    # Randomly sample the required number of negative windows
                    indices = np.random.choice(len(potential_negative_windows), 
                                              size=n_negative, 
                                              replace=False)
                    negative_windows = [potential_negative_windows[i] for i in indices]
                else:
                    # Use all available negative windows
                    negative_windows = potential_negative_windows
                
                self.logger.info(f"Image {annotation.image_id}: {len(positive_windows)} positive, "
                               f"{len(negative_windows)} negative windows (from {len(potential_negative_windows)} candidates)")
                
                # Add all windows from this image to batch
                batch_windows.extend(positive_windows)
                batch_windows.extend(negative_windows)
                
                # Yield batch if it's getting large (memory management)
                if len(batch_windows) >= 1000:  # 1000 windows per batch
                    yield batch_windows
                    batch_windows = []
                    
            except Exception as e:
                self.logger.error(f"Error processing image {annotation.image_path}: {e}")
                continue
        
        # Yield remaining windows
        if batch_windows:
            yield batch_windows
    
    def extract_sliding_window_features(self, image: np.ndarray) -> Iterator[Tuple[np.ndarray, int, int]]:
        """
        Extract features from sliding windows across an image.
        
        Used for inference - generates features for all possible window positions.
        
        Args:
            image: Input image
            
        Yields:
            Tuples of (features, x_position, y_position)
        """
        h, w = image.shape[:2]
        
        for y in range(0, h - self.window_size[1], self.window_stride):
            for x in range(0, w - self.window_size[0], self.window_stride):
                window = image[y:y+self.window_size[1], x:x+self.window_size[0]]
                
                if window.shape[:2] == self.window_size:
                    features = self.extract_hog_lbp_features(window)
                    yield features, x, y