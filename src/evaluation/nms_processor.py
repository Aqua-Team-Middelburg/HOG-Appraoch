"""
Non-Maximum Suppression module for removing duplicate detections.

This module implements standard and multi-scale NMS algorithms for
post-processing detection results from the nurdle detection pipeline.
"""

import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class NonMaximumSuppression:
    """
    Non-Maximum Suppression for removing duplicate detections.
    
    Provides both standard NMS and scale-aware NMS for multi-scale detection.
    """
    
    @staticmethod
    def calculate_iou(box1: List[float], box2: List[float]) -> float:
        """
        Calculate Intersection over Union (IoU) between two bounding boxes.
        
        Args:
            box1: [x1, y1, x2, y2] format
            box2: [x1, y1, x2, y2] format
            
        Returns:
            IoU value between 0 and 1
        """
        # Calculate intersection coordinates
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        # Calculate intersection area
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        
        # Calculate union area
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    @staticmethod
    def apply_nms(detections: List[Dict[str, Any]], iou_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Apply Non-Maximum Suppression to detection results.
        
        Args:
            detections: List of detection dictionaries with 'bbox', 'score', 'label'
            iou_threshold: IoU threshold for suppression
            
        Returns:
            Filtered list of detections after NMS
        """
        if not detections:
            return []
        
        # Sort detections by score in descending order
        detections = sorted(detections, key=lambda x: x['score'], reverse=True)
        
        kept_detections = []
        
        while detections:
            # Keep the detection with highest score
            current = detections.pop(0)
            kept_detections.append(current)
            
            # Remove detections with high IoU overlap
            remaining = []
            for detection in detections:
                iou = NonMaximumSuppression.calculate_iou(
                    current['bbox'], 
                    detection['bbox']
                )
                
                if iou <= iou_threshold:
                    remaining.append(detection)
                # else: detection is suppressed (too much overlap)
            
            detections = remaining
        
        return kept_detections
    
    @staticmethod
    def apply_nms_multi_scale(detections: List[Dict[str, Any]], 
                             iou_threshold: float = 0.5,
                             scale_penalty: float = 0.1) -> List[Dict[str, Any]]:
        """
        Apply scale-aware Non-Maximum Suppression for multi-scale detections.
        
        This method performs NMS across detections from different pyramid scales,
        with optional scale penalty to favor detections from finer (larger) scales.
        
        Args:
            detections: List of detection dicts with 'bbox', 'score', 'label', 'scale'
            iou_threshold: IoU threshold for suppression
            scale_penalty: Score penalty for coarser scales (0=no penalty, 1=strong)
            
        Returns:
            Filtered list of detections after scale-aware NMS
        """
        if not detections:
            return []
        
        # Apply scale penalty to scores
        adjusted_detections = []
        for det in detections:
            det_copy = det.copy()
            
            # Scale penalty: coarser scales (smaller scale values) get lower scores
            # scale=1.0 (original) gets no penalty
            # scale=0.67 (first downsample with factor 1.5) gets small penalty
            scale = det.get('scale', 1.0)
            scale_adjustment = 1.0 - (scale_penalty * (1.0 - scale))
            
            det_copy['adjusted_score'] = det_copy['score'] * scale_adjustment
            det_copy['original_score'] = det_copy['score']
            
            adjusted_detections.append(det_copy)
        
        # Sort by adjusted scores (highest first)
        adjusted_detections = sorted(adjusted_detections, 
                                     key=lambda x: x['adjusted_score'], 
                                     reverse=True)
        
        kept_detections = []
        
        while adjusted_detections:
            # Keep highest scoring detection
            current = adjusted_detections.pop(0)
            kept_detections.append(current)
            
            # Remove overlapping detections from all scales
            remaining = []
            for detection in adjusted_detections:
                iou = NonMaximumSuppression.calculate_iou(
                    current['bbox'],
                    detection['bbox']
                )
                
                if iou <= iou_threshold:
                    remaining.append(detection)
                # else: suppressed due to overlap
            
            adjusted_detections = remaining
        
        # Restore original scores in output
        for det in kept_detections:
            det['score'] = det['original_score']
            del det['adjusted_score']
            del det['original_score']
        
        return kept_detections
