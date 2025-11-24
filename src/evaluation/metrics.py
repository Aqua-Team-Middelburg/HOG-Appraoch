"""
Metrics Calculation Module for Nurdle Detection Pipeline
=======================================================

This module provides metrics calculation for single-stage SVM multi-class count prediction.
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error


class EvaluationMetrics:
    """
    Metrics calculation for single-stage SVM multi-class count prediction.
    """

    def __init__(self):
        pass

    def calculate_count_metrics(self,
                                ground_truth: List[int],
                                predictions: List[int],
                                tolerance_percent: float = 10.0) -> Dict[str, float]:
        """
        Calculate count prediction metrics.
        Args:
            ground_truth: List of actual nurdle counts
            predictions: List of predicted nurdle counts
            tolerance_percent: Percentage tolerance for count accuracy (default 10%)
        Returns:
            Dictionary of count-related metrics
        """
        if not ground_truth or not predictions:
            return {
                'count_accuracy': 0.0,
                'count_mae': 0.0,
                'count_rmse': 0.0,
                'count_bias': 0.0
            }
        accurate_predictions = 0
        for gt, pred in zip(ground_truth, predictions):
            tolerance = max(1, int(gt * tolerance_percent / 100))
            if abs(pred - gt) <= tolerance:
                accurate_predictions += 1
        accuracy = accurate_predictions / len(ground_truth)
        mae = mean_absolute_error(ground_truth, predictions)
        rmse = np.sqrt(mean_squared_error(ground_truth, predictions))
        bias = np.mean(np.array(predictions) - np.array(ground_truth))
        return {
            'count_accuracy': float(accuracy),
            'count_mae': float(mae),
            'count_rmse': float(rmse),
            'count_bias': float(bias)
        }


class CoordinateMatching:
    """
    Handles coordinate matching between predictions and ground truth.
    
    Implements various matching strategies for coordinate-based evaluation.
    """
    
    def __init__(self, default_match_threshold: float = 25.0):
        """
        Initialize coordinate matcher.
        
        Args:
            default_match_threshold: Default distance threshold for coordinate matching (pixels)
        """
        self.default_match_threshold = default_match_threshold
    
    def match_coordinates(self,
                        predicted_coords: List[Tuple[float, float]],
                        ground_truth_coords: List[Tuple[float, float]],
                        match_threshold: Optional[float] = None) -> Tuple[List[float], int, int, int]:
        """
        Match predicted coordinates to ground truth coordinates.
        
        Uses greedy matching: each ground truth coordinate is matched to the
        closest predicted coordinate within the threshold distance.
        
        Args:
            predicted_coords: List of (x, y) predicted coordinates
            ground_truth_coords: List of (x, y) ground truth coordinates
            match_threshold: Maximum distance for matching (pixels)
            
        Returns:
            Tuple of (coordinate_errors, true_positives, false_positives, false_negatives)
        """
        if match_threshold is None:
            match_threshold = self.default_match_threshold
        
        if not ground_truth_coords and not predicted_coords:
            return [], 0, 0, 0
        
        if not ground_truth_coords:
            # No ground truth, all predictions are false positives
            return [], 0, len(predicted_coords), 0
        
        if not predicted_coords:
            # No predictions, all ground truth are false negatives
            return [], 0, 0, len(ground_truth_coords)
        
        # Convert to numpy arrays for easier computation
        pred_coords = np.array(predicted_coords)
        gt_coords = np.array(ground_truth_coords)
        
        # Track which predictions have been matched
        pred_matched = [False] * len(predicted_coords)
        coordinate_errors = []
        true_positives = 0
        
        # Match each ground truth coordinate to closest prediction
        for gt_coord in gt_coords:
            # Calculate distances to all unmatched predictions
            distances = []
            valid_indices = []
            
            for i, pred_coord in enumerate(pred_coords):
                if not pred_matched[i]:
                    distance = np.sqrt((gt_coord[0] - pred_coord[0])**2 + (gt_coord[1] - pred_coord[1])**2)
                    if distance <= match_threshold:
                        distances.append(distance)
                        valid_indices.append(i)
            
            # Match to closest prediction if any valid matches exist
            if distances:
                min_distance_idx = np.argmin(distances)
                matched_pred_idx = valid_indices[min_distance_idx]
                
                # Mark prediction as matched and record error
                pred_matched[matched_pred_idx] = True
                coordinate_errors.append(distances[min_distance_idx])
                true_positives += 1
        
        # Count false positives (unmatched predictions)
        false_positives = sum(1 for matched in pred_matched if not matched)
        
        # Count false negatives (unmatched ground truth)
        false_negatives = len(ground_truth_coords) - true_positives
        
        return coordinate_errors, true_positives, false_positives, false_negatives
    
    def calculate_matching_matrix(self,
                                predicted_coords: List[Tuple[float, float]],
                                ground_truth_coords: List[Tuple[float, float]]) -> np.ndarray:
        """
        Calculate distance matrix between all predicted and ground truth coordinates.
        
        Args:
            predicted_coords: List of (x, y) predicted coordinates
            ground_truth_coords: List of (x, y) ground truth coordinates
            
        Returns:
            Distance matrix of shape (n_predictions, n_ground_truth)
        """
        if not predicted_coords or not ground_truth_coords:
            return np.array([])
        
        pred_coords = np.array(predicted_coords)
        gt_coords = np.array(ground_truth_coords)
        
        # Calculate pairwise distances
        distances = np.zeros((len(pred_coords), len(gt_coords)))
        
        for i, pred_coord in enumerate(pred_coords):
            for j, gt_coord in enumerate(gt_coords):
                distances[i, j] = np.sqrt((pred_coord[0] - gt_coord[0])**2 + (pred_coord[1] - gt_coord[1])**2)
        
        return distances
    
    def optimal_matching(self,
                       predicted_coords: List[Tuple[float, float]],
                       ground_truth_coords: List[Tuple[float, float]],
                       match_threshold: Optional[float] = None) -> Tuple[List[float], int, int, int]:
        """
        Perform optimal (Hungarian algorithm) coordinate matching.
        
        This provides optimal matching but is more computationally expensive
        than greedy matching for large numbers of coordinates.
        
        Args:
            predicted_coords: List of (x, y) predicted coordinates
            ground_truth_coords: List of (x, y) ground truth coordinates
            match_threshold: Maximum distance for matching (pixels)
            
        Returns:
            Tuple of (coordinate_errors, true_positives, false_positives, false_negatives)
        """
        try:
            from scipy.optimize import linear_sum_assignment
        except ImportError:
            # Fall back to greedy matching if scipy is not available
            return self.match_coordinates(predicted_coords, ground_truth_coords, match_threshold)
        
        if match_threshold is None:
            match_threshold = self.default_match_threshold
        
        if not ground_truth_coords and not predicted_coords:
            return [], 0, 0, 0
        
        if not ground_truth_coords:
            return [], 0, len(predicted_coords), 0
        
        if not predicted_coords:
            return [], 0, 0, len(ground_truth_coords)
        
        # Calculate distance matrix
        distance_matrix = self.calculate_matching_matrix(predicted_coords, ground_truth_coords)
        
        # Apply Hungarian algorithm
        pred_indices, gt_indices = linear_sum_assignment(distance_matrix)
        
        coordinate_errors = []
        true_positives = 0
        
        # Process matches
        for pred_idx, gt_idx in zip(pred_indices, gt_indices):
            distance = distance_matrix[pred_idx, gt_idx]
            if distance <= match_threshold:
                coordinate_errors.append(distance)
                true_positives += 1
        
        # Calculate false positives and false negatives
        false_positives = len(predicted_coords) - true_positives
        false_negatives = len(ground_truth_coords) - true_positives
        
        return coordinate_errors, true_positives, false_positives, false_negatives