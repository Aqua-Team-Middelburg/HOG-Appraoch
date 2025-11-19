"""
Non-Maximum Suppression Module for Nurdle Detection Pipeline
===========================================================

This module provides Non-Maximum Suppression (NMS) functionality
to remove duplicate/overlapping detections from model predictions.
"""

from typing import List, Tuple, Optional
import numpy as np


class NonMaximumSuppression:
    """
    Non-Maximum Suppression implementation for nurdle detection.
    
    Handles removal of overlapping detections based on confidence scores
    and spatial proximity.
    """
    
    def __init__(self, default_iou_threshold: float = 0.5, default_min_distance: float = 20.0):
        """
        Initialize NMS with default parameters.
        
        Args:
            default_iou_threshold: Default IoU threshold for suppression
            default_min_distance: Default minimum distance between detections (pixels)
        """
        self.default_iou_threshold = default_iou_threshold
        self.default_min_distance = default_min_distance
    
    def apply_nms(self, 
                  detections: List[Tuple[float, float, float]], 
                  iou_threshold: Optional[float] = None,
                  min_distance: Optional[float] = None) -> List[Tuple[float, float, float]]:
        """
        Apply Non-Maximum Suppression to remove overlapping detections.
        
        For point detections (nurdles), we use distance-based suppression
        rather than traditional bounding box IoU.
        
        Args:
            detections: List of (x, y, confidence) tuples
            iou_threshold: IoU threshold for suppression (not used in distance-based NMS)
            min_distance: Minimum distance between detections (pixels)
        
        Returns:
            Filtered detections
        """
        if not detections:
            return []
        
        if min_distance is None:
            min_distance = self.default_min_distance
        
        # Sort by confidence (highest first)
        sorted_detections = sorted(detections, key=lambda x: x[2], reverse=True)
        
        keep = []
        remaining = sorted_detections.copy()
        
        while remaining:
            # Keep the highest confidence detection
            current = remaining.pop(0)
            keep.append(current)
            
            # Remove overlapping detections
            filtered_remaining = []
            for det in remaining:
                # Calculate distance between current and candidate detection
                distance = self._calculate_distance(current, det)
                
                # If detections are far enough apart, keep the candidate
                if distance > min_distance:
                    filtered_remaining.append(det)
            
            remaining = filtered_remaining
        
        return keep
    
    def apply_nms_with_boxes(self, 
                           detections: List[Tuple[float, float, float, float, float]], 
                           iou_threshold: Optional[float] = None) -> List[Tuple[float, float, float, float, float]]:
        """
        Apply NMS with bounding boxes using traditional IoU calculation.
        
        Args:
            detections: List of (x1, y1, x2, y2, confidence) tuples
            iou_threshold: IoU threshold for suppression
        
        Returns:
            Filtered detections
        """
        if not detections:
            return []
        
        if iou_threshold is None:
            iou_threshold = self.default_iou_threshold
        
        # Sort by confidence (highest first)
        sorted_detections = sorted(detections, key=lambda x: x[4], reverse=True)
        
        keep = []
        remaining = sorted_detections.copy()
        
        while remaining:
            # Keep the highest confidence detection
            current = remaining.pop(0)
            keep.append(current)
            
            # Remove overlapping detections
            filtered_remaining = []
            for det in remaining:
                # Calculate IoU between current and candidate detection
                iou = self._calculate_iou(current[:4], det[:4])
                
                # If IoU is below threshold, keep the candidate
                if iou < iou_threshold:
                    filtered_remaining.append(det)
            
            remaining = filtered_remaining
        
        return keep
    
    def apply_adaptive_nms(self, 
                          detections: List[Tuple[float, float, float]], 
                          base_distance: float = 20.0,
                          confidence_factor: float = 0.1) -> List[Tuple[float, float, float]]:
        """
        Apply adaptive NMS where suppression distance varies with confidence.
        
        Higher confidence detections suppress with larger radius.
        
        Args:
            detections: List of (x, y, confidence) tuples
            base_distance: Base suppression distance
            confidence_factor: Factor for confidence-based distance scaling
        
        Returns:
            Filtered detections
        """
        if not detections:
            return []
        
        # Sort by confidence (highest first)
        sorted_detections = sorted(detections, key=lambda x: x[2], reverse=True)
        
        keep = []
        remaining = sorted_detections.copy()
        
        while remaining:
            # Keep the highest confidence detection
            current = remaining.pop(0)
            keep.append(current)
            
            # Calculate adaptive suppression distance
            adaptive_distance = base_distance * (1 + confidence_factor * current[2])
            
            # Remove overlapping detections
            filtered_remaining = []
            for det in remaining:
                distance = self._calculate_distance(current, det)
                
                if distance > adaptive_distance:
                    filtered_remaining.append(det)
            
            remaining = filtered_remaining
        
        return keep
    
    def cluster_detections(self, 
                          detections: List[Tuple[float, float, float]], 
                          cluster_distance: float = 15.0) -> List[Tuple[float, float, float]]:
        """
        Cluster nearby detections and keep the highest confidence in each cluster.
        
        Args:
            detections: List of (x, y, confidence) tuples
            cluster_distance: Maximum distance for clustering
        
        Returns:
            Clustered detections
        """
        if not detections:
            return []
        
        if len(detections) == 1:
            return detections
        
        # Convert to numpy array for easier processing
        det_array = np.array(detections)
        n_detections = len(detections)
        
        # Create adjacency matrix based on distance
        adjacency = np.zeros((n_detections, n_detections), dtype=bool)
        
        for i in range(n_detections):
            for j in range(i + 1, n_detections):
                distance = self._calculate_distance(detections[i], detections[j])
                if distance <= cluster_distance:
                    adjacency[i, j] = True
                    adjacency[j, i] = True
        
        # Find connected components (clusters)
        visited = [False] * n_detections
        clusters = []
        
        for i in range(n_detections):
            if not visited[i]:
                cluster = self._dfs_cluster(i, adjacency, visited)
                clusters.append(cluster)
        
        # Keep highest confidence detection from each cluster
        result = []
        for cluster_indices in clusters:
            cluster_detections = [detections[i] for i in cluster_indices]
            best_detection = max(cluster_detections, key=lambda x: x[2])
            result.append(best_detection)
        
        return result
    
    def _calculate_distance(self, 
                          det1: Tuple[float, float, float], 
                          det2: Tuple[float, float, float]) -> float:
        """Calculate Euclidean distance between two point detections."""
        return np.sqrt((det1[0] - det2[0])**2 + (det1[1] - det2[1])**2)
    
    def _calculate_iou(self, 
                      box1: Tuple[float, float, float, float], 
                      box2: Tuple[float, float, float, float]) -> float:
        """
        Calculate Intersection over Union (IoU) between two bounding boxes.
        
        Args:
            box1: (x1, y1, x2, y2) for first box
            box2: (x1, y1, x2, y2) for second box
        
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
            intersection = 0.0
        else:
            intersection = (x2 - x1) * (y2 - y1)
        
        # Calculate areas of both boxes
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        # Calculate union area
        union = area1 + area2 - intersection
        
        # Avoid division by zero
        if union <= 0:
            return 0.0
        
        return intersection / union
    
    def _dfs_cluster(self, 
                    start_idx: int, 
                    adjacency: np.ndarray, 
                    visited: List[bool]) -> List[int]:
        """Depth-first search to find connected components in adjacency graph."""
        cluster = []
        stack = [start_idx]
        
        while stack:
            idx = stack.pop()
            if not visited[idx]:
                visited[idx] = True
                cluster.append(idx)
                
                # Add unvisited neighbors to stack
                for neighbor_idx in range(len(adjacency)):
                    if adjacency[idx, neighbor_idx] and not visited[neighbor_idx]:
                        stack.append(neighbor_idx)
        
        return cluster
    
    def get_suppression_stats(self, 
                            original_detections: List[Tuple[float, float, float]], 
                            filtered_detections: List[Tuple[float, float, float]]) -> dict:
        """
        Get statistics about the suppression process.
        
        Args:
            original_detections: Detections before NMS
            filtered_detections: Detections after NMS
        
        Returns:
            Dictionary with suppression statistics
        """
        original_count = len(original_detections)
        filtered_count = len(filtered_detections)
        suppressed_count = original_count - filtered_count
        suppression_rate = suppressed_count / original_count if original_count > 0 else 0.0
        
        # Calculate confidence statistics
        if original_detections:
            original_confidences = [det[2] for det in original_detections]
            avg_original_confidence = np.mean(original_confidences)
            max_original_confidence = max(original_confidences)
            min_original_confidence = min(original_confidences)
        else:
            avg_original_confidence = max_original_confidence = min_original_confidence = 0.0
        
        if filtered_detections:
            filtered_confidences = [det[2] for det in filtered_detections]
            avg_filtered_confidence = np.mean(filtered_confidences)
            max_filtered_confidence = max(filtered_confidences)
            min_filtered_confidence = min(filtered_confidences)
        else:
            avg_filtered_confidence = max_filtered_confidence = min_filtered_confidence = 0.0
        
        return {
            'original_count': original_count,
            'filtered_count': filtered_count,
            'suppressed_count': suppressed_count,
            'suppression_rate': suppression_rate,
            'avg_original_confidence': avg_original_confidence,
            'avg_filtered_confidence': avg_filtered_confidence,
            'max_original_confidence': max_original_confidence,
            'max_filtered_confidence': max_filtered_confidence,
            'min_original_confidence': min_original_confidence,
            'min_filtered_confidence': min_filtered_confidence
        }