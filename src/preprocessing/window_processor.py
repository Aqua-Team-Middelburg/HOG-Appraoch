"""
Sliding window processing module for generating candidate windows.

This module implements sliding window extraction from images for both
training and inference, with consistent parameters across the pipeline.
"""

import os
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Generator
import logging
from tqdm import tqdm
import json

from ..utils.config import ConfigLoader
from .image_pyramid import ImagePyramid
from .window_labeler import WindowLabeler

logger = logging.getLogger(__name__)


class SlidingWindowProcessor:
    """
    Processes images using sliding window approach to extract candidate windows.
    
    This class handles:
    - Sliding window extraction with configurable size and stride
    - IoU calculation for training label assignment
    - Ground truth normalization and matching
    - Balanced dataset generation for training
    """
    
    def __init__(self, config: ConfigLoader):
        """
        Initialize the sliding window processor.
        
        Args:
            config: Configuration loader instance
        """
        self.config = config
        
        # Window configuration
        sliding_config = config.get_section('preprocessing').get('sliding_window', {})
        self.window_size = tuple(sliding_config.get('window_size', [40, 40]))
        self.stride = sliding_config.get('stride', 20)
        
        # Separate stride for training vs inference (Phase 3.2 optimization)
        self.use_separate_strides = sliding_config.get('use_separate_strides', False)
        self.stride_train = sliding_config.get('stride_train', 30)
        self.stride_inference = sliding_config.get('stride_inference', 15)
        
        self.pad_image = sliding_config.get('pad_image', True)
        self.min_coverage = sliding_config.get('min_window_coverage', 0.8)
        
        # Training configuration
        training_config = config.get_section('training')
        self.iou_threshold = training_config.get('iou_threshold', 0.5)
        self.gt_bbox_size = training_config.get('gt_bbox_size', 40)
        
        # Initialize window labeler with training configuration
        self.labeler = WindowLabeler(training_config)
        
        # Paths
        self.normalized_dir = Path(config.get('paths.temp_dir')) / 'normalized_images'
        self.candidate_dir = Path(config.get('paths.candidate_windows_dir', 'temp/candidate_windows'))
        self.candidate_dir.mkdir(parents=True, exist_ok=True)
        
        # Statistics
        self.stats = {
            'total_images': 0,
            'total_windows': 0,
            'positive_windows': 0,
            'negative_windows': 0,
            'processing_times': []
        }
    
    def calculate_iou(self, box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
        """
        Calculate Intersection over Union (IoU) between two bounding boxes.
        
        DEPRECATED: This method is kept for backward compatibility.
        Use center-point labeling instead for better training data quality.
        
        Args:
            box1: First bounding box as (x, y, width, height)
            box2: Second bounding box as (x, y, width, height)
            
        Returns:
            IoU value between 0 and 1
        """
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # Calculate intersection
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        if union == 0:
            return 0.0
            
        return intersection / union
    
    def normalize_ground_truth_bbox(self, bbox: Dict[str, Any]) -> Tuple[int, int, int, int]:
        """
        Normalize ground truth bounding box to consistent size.
        
        Args:
            bbox: Ground truth bounding box dictionary
            
        Returns:
            Normalized bounding box as (x, y, width, height)
        """
        # Delegate to WindowLabeler
        return self.labeler.normalize_ground_truth_bbox(bbox)
    
    def _get_stride(self, mode: str = 'default') -> int:
        """
        Get stride value based on mode (training vs inference).
        
        Args:
            mode: 'train', 'inference', or 'default'
            
        Returns:
            Stride value
        """
        if not self.use_separate_strides:
            return self.stride
        
        if mode == 'train':
            return self.stride_train
        elif mode == 'inference':
            return self.stride_inference
        else:
            return self.stride
    
    def extract_sliding_windows(self, image: np.ndarray, mode: str = 'default') -> Generator[Tuple[np.ndarray, Tuple[int, int, int, int]], None, None]:
        """
        Extract sliding windows from an image.
        
        Args:
            image: Input image as numpy array
            mode: 'train', 'inference', or 'default' (affects stride if separate strides enabled)
            
        Yields:
            Tuples of (window_image, bbox) where bbox is (x, y, width, height)
        """
        height, width = image.shape[:2]
        window_w, window_h = self.window_size
        
        # Get appropriate stride for mode
        stride = self._get_stride(mode)
        
        # Calculate padding if needed
        if self.pad_image:
            pad_w = window_w - (width % stride) if width % stride != 0 else 0
            pad_h = window_h - (height % stride) if height % stride != 0 else 0
            
            if pad_w > 0 or pad_h > 0:
                if len(image.shape) == 3:
                    padded = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')
                else:
                    padded = np.pad(image, ((0, pad_h), (0, pad_w)), mode='constant')
                image = padded
                height, width = image.shape[:2]
        
        # Extract windows
        for y in range(0, height - window_h + 1, stride):
            for x in range(0, width - window_w + 1, stride):
                window = image[y:y + window_h, x:x + window_w]
                
                # Check minimum coverage
                coverage = (window.shape[0] * window.shape[1]) / (window_h * window_w)
                if coverage >= self.min_coverage:
                    yield window, (x, y, window_w, window_h)
    
    def load_ground_truth_annotations(self, json_path: Path) -> List[Dict[str, Any]]:
        """
        Load ground truth annotations from JSON file.
        
        Args:
            json_path: Path to JSON annotation file
            
        Returns:
            List of ground truth object dictionaries
        """
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            return data.get('objects', [])
        except Exception as e:
            logger.error(f"Error loading annotations from {json_path}: {e}")
            return []
    
    def label_window(self, window_bbox: Tuple[int, int, int, int], 
                    ground_truth_objects: List[Dict[str, Any]]) -> bool:
        """
        Determine if a window is positive or negative based on ground truth.
        
        Delegates to WindowLabeler for labeling logic.
        
        Args:
            window_bbox: Window bounding box as (x, y, width, height)
            ground_truth_objects: List of ground truth object annotations
            
        Returns:
            True if positive window, False otherwise
        """
        return self.labeler.label_window(window_bbox, ground_truth_objects)
    
    def process_image_for_training(self, image_path: Path) -> Dict[str, Any]:
        """
        Process a single image to extract labeled training windows.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing processing results
        """
        logger.debug(f"Processing {image_path.name} for training data")
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            logger.error(f"Could not load image: {image_path}")
            return {'success': False, 'error': 'Could not load image'}
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load ground truth annotations
        json_path = image_path.with_suffix('.json')
        ground_truth_objects = self.load_ground_truth_annotations(json_path)
        
        if not ground_truth_objects:
            logger.warning(f"No ground truth annotations found for {image_path.name}")
        
        # Extract windows and labels
        positive_windows = []
        negative_windows = []
        
        # Track which ground truth centers were captured
        centers_captured = set()
        
        # Use training stride for training data
        for window, bbox in self.extract_sliding_windows(image, mode='train'):
            is_positive = self.label_window(bbox, ground_truth_objects)
            
            window_data = {
                'window': window,
                'bbox': bbox,
                'image_name': image_path.name
            }
            
            if is_positive:
                positive_windows.append(window_data)
                
                # Track which centers this window captured
                if self.labeler.labeling_method == 'center_point':
                    for idx, obj in enumerate(ground_truth_objects):
                        bbox_data = obj.get('bbox', obj)
                        center = self.labeler.calculate_nurdle_center(bbox_data)
                        if self.labeler.contains_center_point(bbox, center):
                            centers_captured.add(idx)
            else:
                negative_windows.append(window_data)
        
        # Calculate capture statistics
        total_ground_truth = len(ground_truth_objects)
        captured_count = len(centers_captured)
        capture_rate = (captured_count / total_ground_truth * 100) if total_ground_truth > 0 else 0
        
        results = {
            'success': True,
            'image_name': image_path.name,
            'total_windows': len(positive_windows) + len(negative_windows),
            'positive_windows': len(positive_windows),
            'negative_windows': len(negative_windows),
            'positive_data': positive_windows,
            'negative_data': negative_windows,
            'ground_truth_objects': total_ground_truth,
            'centers_captured': captured_count,
            'capture_rate': capture_rate,
            'labeling_method': self.labeler.labeling_method
        }
        
        if self.labeler.labeling_method == 'center_point':
            logger.debug(f"Processed {image_path.name}: {len(positive_windows)} positive windows, "
                        f"{captured_count}/{total_ground_truth} centers captured ({capture_rate:.1f}%)")
        else:
            logger.debug(f"Processed {image_path.name}: {len(positive_windows)} positive, {len(negative_windows)} negative windows")
        
        return results
    
    def process_image_for_inference(self, image_path: Path) -> Dict[str, Any]:
        """
        Process a single image to extract all candidate windows for inference.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing all extracted windows and their positions
        """
        logger.debug(f"Processing {image_path.name} for inference")
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            logger.error(f"Could not load image: {image_path}")
            return {'success': False, 'error': 'Could not load image'}
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Extract all windows
        windows = []
        bboxes = []
        
        # Use inference stride for detection
        for window, bbox in self.extract_sliding_windows(image, mode='inference'):
            windows.append(window)
            bboxes.append(bbox)
        
        results = {
            'success': True,
            'image_name': image_path.name,
            'image_shape': image.shape,
            'total_windows': len(windows),
            'windows': windows,
            'bboxes': bboxes
        }
        
        logger.debug(f"Extracted {len(windows)} windows from {image_path.name}")
        return results
    
    def process_image_multi_scale(self, image_path: Path, 
                                  pyramid: Optional[ImagePyramid] = None) -> Dict[str, Any]:
        """
        Process image at multiple scales using image pyramids.
        
        Extracts candidate windows from each pyramid level, enabling detection
        of objects at different scales. Window coordinates are mapped back to
        original image coordinates.
        
        Args:
            image_path: Path to the image file
            pyramid: Optional ImagePyramid instance (creates new if None)
            
        Returns:
            Dictionary containing windows from all scales with metadata
        """
        logger.debug(f"Processing {image_path.name} at multiple scales")
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            logger.error(f"Could not load image: {image_path}")
            return {'success': False, 'error': 'Could not load image'}
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create pyramid if not provided
        if pyramid is None:
            pyramid = ImagePyramid(self.config)
        
        # Generate pyramid levels
        pyramid_levels = pyramid.generate_pyramid(image)
        
        # Storage for multi-scale windows
        all_windows = []
        all_bboxes = []
        all_scales = []
        all_pyramid_levels = []
        
        # Process each pyramid level
        for pyr_level in pyramid_levels:
            # Extract windows from this scale
            level_windows = []
            level_bboxes = []
            
            for window, bbox in self.extract_sliding_windows(pyr_level.image):
                # Map bbox coordinates back to original image
                orig_bbox = pyramid.map_coordinates_to_original(
                    bbox[0], bbox[1], bbox[2], bbox[3],
                    pyr_level.scale
                )
                
                level_windows.append(window)
                level_bboxes.append(orig_bbox)  # Store in original coordinates
            
            # Append to all windows
            all_windows.extend(level_windows)
            all_bboxes.extend(level_bboxes)
            all_scales.extend([pyr_level.scale] * len(level_windows))
            all_pyramid_levels.extend([pyr_level.level] * len(level_windows))
            
            logger.debug(f"Level {pyr_level.level} (scale={pyr_level.scale:.3f}): "
                        f"extracted {len(level_windows)} windows")
        
        results = {
            'success': True,
            'image_name': image_path.name,
            'image_shape': image.shape,
            'num_pyramid_levels': len(pyramid_levels),
            'pyramid_scales': [level.scale for level in pyramid_levels],
            'total_windows': len(all_windows),
            'windows': all_windows,
            'bboxes': all_bboxes,  # All in original image coordinates
            'scales': all_scales,  # Which scale each window came from
            'pyramid_levels': all_pyramid_levels  # Which pyramid level
        }
        
        logger.info(f"Multi-scale processing: {len(all_windows)} windows from "
                   f"{len(pyramid_levels)} pyramid levels for {image_path.name}")
        
        return results
    
    def generate_balanced_training_dataset(self, positive_ratio: float = 0.3) -> Dict[str, Any]:
        """
        Generate a balanced training dataset from all normalized images.
        
        Delegates to TrainingDatasetBuilder for batch processing.
        
        Args:
            positive_ratio: Desired ratio of positive to total samples
            
        Returns:
            Dictionary containing balanced training data and statistics
        """
        from .training_dataset_builder import TrainingDatasetBuilder
        
        builder = TrainingDatasetBuilder(self.config, self)
        return builder.generate_balanced_dataset(positive_ratio)
    
    def _save_processed_windows(self, positive_windows: List[Dict], negative_windows: List[Dict]) -> None:
        """
        Save processed windows to disk for training.
        
        Delegates to TrainingDatasetBuilder.
        
        Args:
            positive_windows: List of positive window data
            negative_windows: List of negative window data
        """
        from .training_dataset_builder import TrainingDatasetBuilder
        
        builder = TrainingDatasetBuilder(self.config, self)
        builder.save_processed_windows(positive_windows, negative_windows)
    
    def load_processed_windows(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load previously processed training windows.
        
        Delegates to TrainingDatasetBuilder.
        
        Returns:
            Tuple of (positive_windows, negative_windows, all_windows, labels)
        """
        from .training_dataset_builder import TrainingDatasetBuilder
        
        builder = TrainingDatasetBuilder(self.config, self)
        return builder.load_processed_windows()