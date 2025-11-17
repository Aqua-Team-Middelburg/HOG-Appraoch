"""
Window labeling module for training data generation.

This module implements window labeling logic including center-point based
and IoU-based labeling methods for generating training datasets.
"""

import numpy as np
import logging
from typing import List, Dict, Any, Tuple
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class WindowLabeler:
    """
    Labels sliding windows as positive or negative based on ground truth.
    
    Supports two labeling methods:
    - Center-point: Window is positive if it contains GT object center
    - IoU: Window is positive if IoU with GT box exceeds threshold (legacy)
    """
    
    def __init__(self, config: dict):
        """
        Initialize window labeler.
        
        Args:
            config: Labeling configuration including method and parameters
        """
        self.labeling_method = config.get('labeling_method', 'center_point')
        self.iou_threshold = config.get('iou_threshold', 0.5)
        self.gt_bbox_size = config.get('gt_bbox_size', 40)
        
        # Center-point labeling configuration
        center_point_config = config.get('center_point', {})
        self.allow_multiple_centers = center_point_config.get('allow_multiple_centers', False)
        self.center_tolerance_px = center_point_config.get('center_tolerance_px', 0)
        
        logger.info(f"Window labeler initialized: method={self.labeling_method}")
    
    def calculate_iou(self, box1: Tuple[int, int, int, int], 
                     box2: Tuple[int, int, int, int]) -> float:
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
    
    def calculate_nurdle_center(self, bbox: Dict[str, Any]) -> Tuple[int, int]:
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
    
    def contains_center_point(self, window_bbox: Tuple[int, int, int, int], 
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
            center_point = self.calculate_nurdle_center(bbox)
            
            # Check if this window contains the center
            if self.contains_center_point(window_bbox, center_point):
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
    
    def calculate_capture_statistics(self, 
                                     positive_windows: List[Tuple[int, int, int, int]],
                                     ground_truth_objects: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate statistics about how many GT centers were captured.
        
        Args:
            positive_windows: List of positive window bboxes
            ground_truth_objects: List of ground truth annotations
            
        Returns:
            Dictionary with capture statistics
        """
        centers_captured = set()
        
        for window_bbox in positive_windows:
            for idx, obj in enumerate(ground_truth_objects):
                bbox_data = obj.get('bbox', obj)
                center = self.calculate_nurdle_center(bbox_data)
                if self.contains_center_point(window_bbox, center):
                    centers_captured.add(idx)
        
        total_ground_truth = len(ground_truth_objects)
        captured_count = len(centers_captured)
        capture_rate = (captured_count / total_ground_truth * 100) if total_ground_truth > 0 else 0
        
        return {
            'total_ground_truth': total_ground_truth,
            'centers_captured': captured_count,
            'capture_rate': capture_rate
        }
