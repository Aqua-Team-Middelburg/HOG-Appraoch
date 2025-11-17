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
        self.pad_image = sliding_config.get('pad_image', True)
        self.min_coverage = sliding_config.get('min_window_coverage', 0.8)
        
        # Training configuration
        training_config = config.get_section('training')
        self.iou_threshold = training_config.get('iou_threshold', 0.5)
        self.gt_bbox_size = training_config.get('gt_bbox_size', 40)
        
        # Center-point labeling configuration
        self.labeling_method = training_config.get('labeling_method', 'center_point')
        center_point_config = training_config.get('center_point', {})
        self.allow_multiple_centers = center_point_config.get('allow_multiple_centers', False)
        self.center_tolerance_px = center_point_config.get('center_tolerance_px', 0)
        
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
    
    def _calculate_nurdle_center(self, bbox: Dict[str, Any]) -> Tuple[int, int]:
        """
        Calculate the center point of a nurdle from its bounding box annotation.
        
        Args:
            bbox: Ground truth bounding box dictionary
            
        Returns:
            Tuple of (center_x, center_y) coordinates
        """
        # Try to get explicit center coordinates
        if 'center_x' in bbox and 'center_y' in bbox:
            return (int(bbox['center_x']), int(bbox['center_y']))
        
        # Calculate from bbox coordinates
        if 'x' in bbox and 'y' in bbox:
            x = bbox['x']
            y = bbox['y']
            width = bbox.get('width', self.gt_bbox_size)
            height = bbox.get('height', self.gt_bbox_size)
            
            center_x = int(x + width / 2)
            center_y = int(y + height / 2)
            return (center_x, center_y)
        
        # Fallback: use radius if available
        if 'radius' in bbox:
            center_x = bbox.get('center_x', 0)
            center_y = bbox.get('center_y', 0)
            return (int(center_x), int(center_y))
        
        logger.warning(f"Could not calculate center for bbox: {bbox}")
        return (0, 0)
    
    def _contains_center_point(self, window_bbox: Tuple[int, int, int, int], 
                               center_point: Tuple[int, int]) -> bool:
        """
        Check if a window contains a ground truth center point.
        
        Args:
            window_bbox: Window bounding box as (x, y, width, height)
            center_point: Ground truth center as (center_x, center_y)
            
        Returns:
            True if center point falls within window bounds (with tolerance)
        """
        win_x, win_y, win_w, win_h = window_bbox
        center_x, center_y = center_point
        
        # Apply tolerance
        tolerance = self.center_tolerance_px
        
        # Check if center is within window bounds (with tolerance)
        x_in_bounds = (win_x - tolerance) <= center_x <= (win_x + win_w + tolerance)
        y_in_bounds = (win_y - tolerance) <= center_y <= (win_y + win_h + tolerance)
        
        return x_in_bounds and y_in_bounds
    
    def normalize_ground_truth_bbox(self, bbox: Dict[str, Any]) -> Tuple[int, int, int, int]:
        """
        Normalize ground truth bounding box to consistent size.
        
        Args:
            bbox: Ground truth bounding box dictionary
            
        Returns:
            Normalized bounding box as (x, y, width, height)
        """
        # Extract center coordinates
        center_x = bbox.get('center_x', bbox.get('x', 0) + bbox.get('width', 0) // 2)
        center_y = bbox.get('center_y', bbox.get('y', 0) + bbox.get('height', 0) // 2)
        
        # Create normalized bbox centered at the same position
        half_size = self.gt_bbox_size // 2
        x = center_x - half_size
        y = center_y - half_size
        
        return (x, y, self.gt_bbox_size, self.gt_bbox_size)
    
    def extract_sliding_windows(self, image: np.ndarray) -> Generator[Tuple[np.ndarray, Tuple[int, int, int, int]], None, None]:
        """
        Extract sliding windows from an image.
        
        Args:
            image: Input image as numpy array
            
        Yields:
            Tuples of (window_image, bbox) where bbox is (x, y, width, height)
        """
        height, width = image.shape[:2]
        window_w, window_h = self.window_size
        
        # Calculate padding if needed
        if self.pad_image:
            pad_w = window_w - (width % self.stride) if width % self.stride != 0 else 0
            pad_h = window_h - (height % self.stride) if height % self.stride != 0 else 0
            
            if pad_w > 0 or pad_h > 0:
                if len(image.shape) == 3:
                    padded = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')
                else:
                    padded = np.pad(image, ((0, pad_h), (0, pad_w)), mode='constant')
                image = padded
                height, width = image.shape[:2]
        
        # Extract windows
        for y in range(0, height - window_h + 1, self.stride):
            for x in range(0, width - window_w + 1, self.stride):
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
        
        Uses center-point labeling by default (configurable to use IoU for legacy support).
        
        Args:
            window_bbox: Window bounding box as (x, y, width, height)
            ground_truth_objects: List of ground truth object annotations
            
        Returns:
            True if positive window, False otherwise
        """
        if self.labeling_method == 'center_point':
            return self._label_window_center_point(window_bbox, ground_truth_objects)
        else:
            # Legacy IoU-based labeling
            return self._label_window_iou(window_bbox, ground_truth_objects)
    
    def _label_window_center_point(self, window_bbox: Tuple[int, int, int, int],
                                   ground_truth_objects: List[Dict[str, Any]]) -> bool:
        """
        Label window as positive if it contains a ground truth center point.
        
        Args:
            window_bbox: Window bounding box as (x, y, width, height)
            ground_truth_objects: List of ground truth object annotations
            
        Returns:
            True if window contains at least one ground truth center point
        """
        centers_found = 0
        
        for obj in ground_truth_objects:
            # Get bounding box from object
            if 'bbox' in obj:
                bbox = obj['bbox']
            else:
                # Construct bbox from object attributes
                bbox = obj
            
            # Calculate center point
            center_point = self._calculate_nurdle_center(bbox)
            
            # Check if this window contains the center
            if self._contains_center_point(window_bbox, center_point):
                centers_found += 1
                
                # If we don't allow multiple centers and found one, return immediately
                if not self.allow_multiple_centers:
                    return True
        
        # If allowing multiple centers, return True if at least one found
        if self.allow_multiple_centers and centers_found > 0:
            if centers_found > 1:
                logger.debug(f"Window contains {centers_found} nurdle centers")
            return True
        
        return centers_found > 0
    
    def _label_window_iou(self, window_bbox: Tuple[int, int, int, int],
                         ground_truth_objects: List[Dict[str, Any]]) -> bool:
        """
        Legacy IoU-based window labeling (kept for backward compatibility).
        
        Args:
            window_bbox: Window bounding box as (x, y, width, height)
            ground_truth_objects: List of ground truth object annotations
            
        Returns:
            True if positive window (IoU >= threshold), False otherwise
        """
        for obj in ground_truth_objects:
            # Normalize ground truth bbox
            if 'bbox' in obj:
                gt_bbox = self.normalize_ground_truth_bbox(obj['bbox'])
            else:
                # Create bbox from center coordinates and radius if available
                center_x = obj.get('center_x', 0)
                center_y = obj.get('center_y', 0)
                radius = obj.get('radius', self.gt_bbox_size // 2)
                gt_bbox = (center_x - radius, center_y - radius, radius * 2, radius * 2)
            
            # Calculate IoU
            iou = self.calculate_iou(window_bbox, gt_bbox)
            if iou >= self.iou_threshold:
                return True
        
        return False
    
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
        
        for window, bbox in self.extract_sliding_windows(image):
            is_positive = self.label_window(bbox, ground_truth_objects)
            
            window_data = {
                'window': window,
                'bbox': bbox,
                'image_name': image_path.name
            }
            
            if is_positive:
                positive_windows.append(window_data)
                
                # Track which centers this window captured
                if self.labeling_method == 'center_point':
                    for idx, obj in enumerate(ground_truth_objects):
                        bbox_data = obj.get('bbox', obj)
                        center = self._calculate_nurdle_center(bbox_data)
                        if self._contains_center_point(bbox, center):
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
            'labeling_method': self.labeling_method
        }
        
        if self.labeling_method == 'center_point':
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
        
        for window, bbox in self.extract_sliding_windows(image):
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
    
    def generate_balanced_training_dataset(self, positive_ratio: float = 0.3) -> Dict[str, Any]:
        """
        Generate a balanced training dataset from all normalized images.
        
        Args:
            positive_ratio: Desired ratio of positive to total samples
            
        Returns:
            Dictionary containing balanced training data and statistics
        """
        logger.info("Generating balanced training dataset...")
        
        # Get all normalized image files
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png']:
            image_files.extend(list(self.normalized_dir.glob(f"*{ext}")))
        
        if not image_files:
            raise ValueError("No normalized images found. Run normalization first.")
        
        logger.info(f"Processing {len(image_files)} images for training data")
        
        all_positive_windows = []
        all_negative_windows = []
        
        # Process each image
        for image_path in tqdm(image_files, desc="Processing images"):
            result = self.process_image_for_training(image_path)
            
            if result['success']:
                all_positive_windows.extend(result['positive_data'])
                all_negative_windows.extend(result['negative_data'])
        
        # Balance the dataset
        num_positive = len(all_positive_windows)
        num_negative_needed = int(num_positive * (1 - positive_ratio) / positive_ratio)
        
        if num_negative_needed > len(all_negative_windows):
            logger.warning(f"Not enough negative samples. Using all {len(all_negative_windows)} available.")
            balanced_negative_windows = all_negative_windows
        else:
            # Randomly sample negative windows
            np.random.seed(self.config.get('system.reproducibility.random_seed', 42))
            indices = np.random.choice(len(all_negative_windows), num_negative_needed, replace=False)
            balanced_negative_windows = [all_negative_windows[i] for i in indices]
        
        # Save processed windows
        self._save_processed_windows(all_positive_windows, balanced_negative_windows)
        
        results = {
            'total_images_processed': len(image_files),
            'total_positive_windows': num_positive,
            'total_negative_windows_available': len(all_negative_windows),
            'balanced_negative_windows': len(balanced_negative_windows),
            'final_positive_ratio': num_positive / (num_positive + len(balanced_negative_windows)),
            'output_directory': str(self.candidate_dir)
        }
        
        logger.info(f"Training dataset generated: {results}")
        return results
    
    def _save_processed_windows(self, positive_windows: List[Dict], negative_windows: List[Dict]) -> None:
        """
        Save processed windows to disk for training.
        
        Args:
            positive_windows: List of positive window data
            negative_windows: List of negative window data
        """
        logger.info("Saving processed windows...")
        
        # Extract window arrays
        positive_arrays = [w['window'] for w in positive_windows]
        negative_arrays = [w['window'] for w in negative_windows]
        
        # Save as numpy arrays
        if positive_arrays:
            np.save(self.candidate_dir / 'processed_windows_positive.npy', positive_arrays)
        if negative_arrays:
            np.save(self.candidate_dir / 'processed_windows_negative.npy', negative_arrays)
        
        # Save metadata
        metadata = {
            'window_size': self.window_size,
            'stride': self.stride,
            'labeling_method': self.labeling_method,
            'iou_threshold': self.iou_threshold if self.labeling_method == 'iou' else None,
            'center_tolerance_px': self.center_tolerance_px if self.labeling_method == 'center_point' else None,
            'allow_multiple_centers': self.allow_multiple_centers if self.labeling_method == 'center_point' else None,
            'gt_bbox_size': self.gt_bbox_size,
            'positive_count': len(positive_windows),
            'negative_count': len(negative_windows)
        }
        
        with open(self.candidate_dir / 'window_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save detailed statistics
        stats = {
            'processing_stats': self.stats,
            'positive_samples': [
                {
                    'image_name': w['image_name'],
                    'bbox': w['bbox']
                }
                for w in positive_windows
            ],
            'negative_samples': [
                {
                    'image_name': w['image_name'],
                    'bbox': w['bbox']
                }
                for w in negative_windows
            ]
        }
        
        with open(self.candidate_dir / 'processing_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Saved {len(positive_windows)} positive and {len(negative_windows)} negative windows "
                   f"(labeling method: {self.labeling_method})")

    
    def load_processed_windows(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load previously processed training windows.
        
        Returns:
            Tuple of (positive_windows, negative_windows, all_windows, labels)
        """
        positive_path = self.candidate_dir / 'processed_windows_positive.npy'
        negative_path = self.candidate_dir / 'processed_windows_negative.npy'
        
        if not positive_path.exists() or not negative_path.exists():
            raise FileNotFoundError("Processed windows not found. Run generate_balanced_training_dataset first.")
        
        positive_windows = np.load(positive_path)
        negative_windows = np.load(negative_path)
        
        # Combine and create labels
        all_windows = np.concatenate([positive_windows, negative_windows], axis=0)
        labels = np.concatenate([
            np.ones(len(positive_windows)),
            np.zeros(len(negative_windows))
        ])
        
        logger.info(f"Loaded {len(positive_windows)} positive and {len(negative_windows)} negative windows")
        return positive_windows, negative_windows, all_windows, labels