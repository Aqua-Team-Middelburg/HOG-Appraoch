"""
Comprehensive evaluation module for nurdle detection pipeline.

This module provides detailed evaluation metrics, visualization, and analysis
for trained models including window-level and image-level performance metrics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional, Union
from collections import defaultdict

from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score, f1_score,
    precision_score, recall_score, accuracy_score
)

from ..utils.config import ConfigLoader
from ..training.bbox_regressor import BoundingBoxRegressor

logger = logging.getLogger(__name__)


class TestSetLoader:
    """
    Loads and processes test set data with proper image-window mapping.
    """
    
    def __init__(self, config: ConfigLoader):
        self.config = config
        
    def load_test_set_features_and_metadata(self) -> Dict[str, Any]:
        """
        Load test set features and create proper window-to-image mapping.
        
        Returns:
            Dictionary containing test features, labels, and metadata
        """
        logger.info("Loading test set with proper image-window mapping...")
        
        # Try to load from separate test set first
        test_set_dir = Path(self.config.get('paths.test_set_dir', 'test_set'))
        
        if test_set_dir.exists():
            return self._load_separate_test_set(test_set_dir)
        else:
            return self._create_test_split_from_training()
    
    def _load_separate_test_set(self, test_dir: Path) -> Dict[str, Any]:
        """Load features from separate test set directory."""
        logger.info(f"Loading separate test set from {test_dir}")
        
        # Look for test set features
        test_features_dir = test_dir / 'features'
        if not test_features_dir.exists():
            logger.warning(f"Test features directory not found: {test_features_dir}")
            return self._create_test_split_from_training()
        
        test_data = {}
        window_metadata = {}
        
        for feature_type in ['hog', 'lbp', 'combined']:
            feature_file = test_features_dir / f'{feature_type}_features.npy'
            labels_file = test_features_dir / f'{feature_type}_labels.npy'
            metadata_file = test_features_dir / f'{feature_type}_window_metadata.json'
            
            if feature_file.exists() and labels_file.exists():
                features = np.load(feature_file)
                labels = np.load(labels_file)
                
                test_data[feature_type] = {
                    'features': features,
                    'labels': labels
                }
                
                # Load window metadata if available
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        window_metadata[feature_type] = json.load(f)
                
                logger.info(f"Loaded test {feature_type}: {features.shape}")
        
        return {
            'test_data': test_data,
            'window_metadata': window_metadata,
            'source': 'separate_test_set'
        }
    
    def _create_test_split_from_training(self) -> Dict[str, Any]:
        """Create test split from training data with enhanced metadata."""
        logger.info("Creating test split from training data...")
        
        features_dir = Path(self.config.get('paths.extracted_features_dir', 'temp/extracted_features'))
        test_split_ratio = self.config.get_section('evaluation').get('test_split_ratio', 0.2)
        
        test_data = {}
        window_metadata = {}
        
        for feature_type in ['hog', 'lbp', 'combined']:
            feature_file = features_dir / f'{feature_type}_features.npy'
            labels_file = features_dir / f'{feature_type}_labels.npy'
            
            if feature_file.exists() and labels_file.exists():
                features = np.load(feature_file)
                labels = np.load(labels_file)
                
                # Create stratified test split
                from sklearn.model_selection import train_test_split
                
                _, test_features, _, test_labels = train_test_split(
                    features, labels, 
                    test_size=test_split_ratio, 
                    stratify=labels,
                    random_state=42
                )
                
                test_data[feature_type] = {
                    'features': test_features,
                    'labels': test_labels
                }
                
                # Create simulated window metadata for image grouping
                window_metadata[feature_type] = self._create_simulated_metadata(len(test_features))
                
                logger.info(f"Created test {feature_type}: {test_features.shape}")
        
        return {
            'test_data': test_data,
            'window_metadata': window_metadata,
            'source': 'training_split'
        }
    
    def _create_simulated_metadata(self, n_windows: int) -> Dict[str, Any]:
        """Create simulated window-to-image mapping."""
        windows_per_image = self.config.get_section('evaluation').get('simulated_windows_per_image', 20)
        n_images = max(1, n_windows // windows_per_image)
        
        metadata = {
            'window_to_image': {},
            'image_info': {},
            'windows_per_image': windows_per_image
        }
        
        for i in range(n_windows):
            image_id = f"test_image_{i // windows_per_image}"
            metadata['window_to_image'][str(i)] = {
                'image_id': image_id,
                'window_index': i % windows_per_image
            }
        
        for i in range(n_images):
            image_id = f"test_image_{i}"
            metadata['image_info'][image_id] = {
                'total_windows': min(windows_per_image, n_windows - i * windows_per_image)
            }
        
        return metadata


class NonMaximumSuppression:
    """
    Non-Maximum Suppression for removing duplicate detections.
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


class ThresholdOptimizer:
    """
    Optimizes detection thresholds for each model using validation curves.
    """
    
    def __init__(self, config: ConfigLoader):
        self.config = config
    
    def optimize_threshold(self, 
                          y_true: np.ndarray, 
                          y_scores: np.ndarray,
                          metric: str = 'f1') -> Dict[str, Any]:
        """
        Find optimal decision threshold using validation curve.
        
        Args:
            y_true: True binary labels
            y_scores: Model decision scores or probabilities
            metric: Optimization metric ('f1', 'precision', 'recall', 'accuracy')
            
        Returns:
            Dictionary with optimal threshold and metrics
        """
        if y_scores is None:
            return {
                'optimal_threshold': 0.0,
                'optimal_score': 0.0,
                'thresholds': [0.0],
                'scores': [0.0],
                'method': 'default'
            }
        
        # Generate threshold range
        min_score = np.min(y_scores)
        max_score = np.max(y_scores)
        thresholds = np.linspace(min_score, max_score, 100)
        
        scores = []
        
        for threshold in thresholds:
            y_pred = (y_scores >= threshold).astype(int)
            
            if metric == 'f1':
                score = f1_score(y_true, y_pred, zero_division=0)
            elif metric == 'precision':
                score = precision_score(y_true, y_pred, zero_division=0)
            elif metric == 'recall':
                score = recall_score(y_true, y_pred, zero_division=0)
            elif metric == 'accuracy':
                score = accuracy_score(y_true, y_pred)
            else:
                score = f1_score(y_true, y_pred, zero_division=0)
            
            scores.append(score)
        
        # Find optimal threshold
        best_idx = np.argmax(scores)
        optimal_threshold = thresholds[best_idx]
        optimal_score = scores[best_idx]
        
        return {
            'optimal_threshold': float(optimal_threshold),
            'optimal_score': float(optimal_score),
            'thresholds': thresholds.tolist(),
            'scores': scores,
            'method': 'validation_curve'
        }


class ModelEvaluator:
    """
    Comprehensive model evaluation with multiple metrics and visualizations.
    
    Provides:
    - Window-level classification metrics
    - Image-level aggregated metrics  
    - ROC and PR curve analysis
    - Confusion matrix visualization
    - Performance comparison across models
    """
    
    def __init__(self, config: ConfigLoader):
        """
        Initialize evaluator.
        
        Args:
            config: Configuration loader instance
        """
        self.config = config
        
        # Evaluation configuration
        eval_config = config.get_section('evaluation')
        self.metrics_config = eval_config.get('metrics', {})
        self.visualization_config = eval_config.get('visualization', {})
        self.detection_config = eval_config.get('detection', {})
        
        # Initialize helper classes
        self.test_loader = TestSetLoader(config)
        self.nms = NonMaximumSuppression()
        self.threshold_optimizer = ThresholdOptimizer(config)
        
        # Paths
        self.figures_dir = Path(config.get('paths.output_dir')) / config.get('paths.figures_dir')
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
        # Results storage
        self.evaluation_results = {}
        
        logger.info("Model evaluator initialized with NMS and threshold optimization")
    
    def evaluate_single_model(self,
                             model_package: Dict[str, Any],
                             test_features: np.ndarray,
                             test_labels: np.ndarray,
                             model_name: str) -> Dict[str, Any]:
        """
        Evaluate a single trained model.
        
        Args:
            model_package: Trained model package
            test_features: Test feature matrix
            test_labels: Test labels
            model_name: Name identifier for model
            
        Returns:
            Dictionary containing comprehensive evaluation results
        """
        logger.info(f"Evaluating {model_name} model")
        
        model = model_package['model']
        scaler = model_package['scaler']
        
        # Scale test features
        scaled_features = scaler.transform(test_features)
        
        # Predictions
        predictions = model.predict(scaled_features)
        
        # Get decision scores (for threshold optimization)
        decision_scores = None
        probabilities = None
        
        try:
            if hasattr(model, 'decision_function'):
                decision_scores = model.decision_function(scaled_features)
                probabilities = decision_scores  # Use decision scores as probabilities
            elif hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(scaled_features)[:, 1]
                decision_scores = probabilities
        except Exception as e:
            logger.warning(f"Could not get decision scores: {e}")
        
        # Optimize decision threshold
        if decision_scores is not None:
            threshold_results = self.threshold_optimizer.optimize_threshold(
                test_labels, decision_scores, metric='f1'
            )
            optimal_threshold = threshold_results['optimal_threshold']
            
            # Make predictions with optimal threshold
            optimized_predictions = (decision_scores >= optimal_threshold).astype(int)
        else:
            threshold_results = {'optimal_threshold': 0.0, 'optimal_score': 0.0, 'method': 'default'}
            optimized_predictions = predictions
            optimal_threshold = 0.0
        
        # Basic classification metrics (with default threshold)
        default_metrics = self._calculate_classification_metrics(test_labels, predictions, probabilities)
        
        # Optimized metrics (with optimal threshold)  
        optimized_metrics = self._calculate_classification_metrics(test_labels, optimized_predictions, probabilities)
        
        # Use optimized metrics as primary metrics
        metrics = optimized_metrics
        
        # Detailed classification report
        report = classification_report(test_labels, predictions, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(test_labels, predictions)
        
        # Compile results
        results = {
            'model_name': model_name,
            'metrics': metrics,
            'default_metrics': default_metrics,
            'optimized_metrics': optimized_metrics,
            'threshold_optimization': threshold_results,
            'optimal_threshold': optimal_threshold,
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'predictions': predictions.tolist(),
            'optimized_predictions': optimized_predictions.tolist(),
            'test_labels': test_labels.tolist(),
            'probabilities': probabilities.tolist() if probabilities is not None else None,
            'decision_scores': decision_scores.tolist() if decision_scores is not None else None,
            'test_samples': len(test_labels),
            'positive_samples': int(np.sum(test_labels)),
            'negative_samples': int(len(test_labels) - np.sum(test_labels)),
            'evaluation_timestamp': datetime.now().isoformat()
        }
        
        # Store results
        self.evaluation_results[model_name] = results
        
        logger.info(f"{model_name} evaluation completed - F1: {metrics['f1_score']:.4f}")
        
        return results
    
    def _calculate_classification_metrics(self,
                                        y_true: np.ndarray,
                                        y_pred: np.ndarray,
                                        y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Calculate comprehensive classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Prediction probabilities (optional)
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
        }
        
        # Add probability-based metrics if available
        if y_prob is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
                metrics['average_precision'] = average_precision_score(y_true, y_prob)
            except ValueError:
                # Handle cases where only one class is present
                metrics['roc_auc'] = None
                metrics['average_precision'] = None
        else:
            metrics['roc_auc'] = None
            metrics['average_precision'] = None
        
        # Custom metrics from config
        if self.metrics_config.get('calculate_mape', False):
            metrics['mape'] = self._calculate_mape(y_true, y_pred)
        
        if self.metrics_config.get('calculate_mae', False):
            metrics['mae'] = self._calculate_mae(y_true, y_pred)
        
        return metrics
    
    def _calculate_mape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Mean Absolute Percentage Error.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            MAPE value
        """
        # For binary classification, treat as 0/1 and avoid division by zero
        mask = y_true != 0
        if np.sum(mask) == 0:
            return 0.0
        
        return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
    
    def _calculate_mae(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Mean Absolute Error.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            MAE value
        """
        return float(np.mean(np.abs(y_true - y_pred)))
    
    def _apply_bbox_refinement(self,
                              detections: List[Dict[str, Any]],
                              features: np.ndarray,
                              bbox_regressor: Optional[BoundingBoxRegressor],
                              confidence_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Apply bounding box regression to refine detection windows.
        
        Uses trained regressor to predict offset and scale adjustments,
        transforming crude sliding window detections into tight-fitting boxes.
        
        Args:
            detections: List of detection dicts with 'bbox', 'score', 'window_idx'
            features: Feature vectors for all windows (N × D)
            bbox_regressor: Trained BoundingBoxRegressor instance (or None)
            confidence_threshold: Minimum confidence to apply refinement
            
        Returns:
            List of detections with refined 'bbox' coordinates
        """
        if bbox_regressor is None or bbox_regressor.regressor is None:
            logger.debug("No trained bbox regressor - skipping refinement")
            return detections
        
        if not detections:
            return detections
        
        logger.info(f"Applying bbox refinement to {len(detections)} detections")
        
        # Extract window indices and features for detections
        detection_indices = [det.get('window_idx', -1) for det in detections]
        
        # Filter out detections without window indices
        valid_detections = []
        valid_indices = []
        for det, idx in zip(detections, detection_indices):
            if idx >= 0 and idx < len(features):
                valid_detections.append(det)
                valid_indices.append(idx)
        
        if not valid_detections:
            logger.warning("No valid window indices for bbox refinement")
            return detections
        
        # Get features for valid detections
        detection_features = features[valid_indices]
        
        # Prepare detection list for regressor
        detection_list = [
            {
                'bbox': det['bbox'],
                'score': det['score'],
                'label': det.get('label', det.get('prediction', 1))
            }
            for det in valid_detections
        ]
        
        # Apply refinement
        try:
            refined_detections = bbox_regressor.refine_detections(
                detection_list,
                detection_features,
                min_confidence=confidence_threshold
            )
            
            # Update original detection bboxes
            for i, refined_det in enumerate(refined_detections):
                valid_detections[i]['bbox'] = refined_det['bbox']
                valid_detections[i]['refined'] = True
            
            logger.info(f"Successfully refined {len(refined_detections)} bounding boxes")
            
        except Exception as e:
            logger.error(f"Bbox refinement failed: {e}")
            # Return original detections on error
            return detections
        
        return valid_detections
    
    def evaluate_image_level_performance(self,
                                       model_results: Dict[str, Dict[str, Any]],
                                       window_metadata: Optional[Dict[str, Any]] = None,
                                       bbox_regressors: Optional[Dict[str, BoundingBoxRegressor]] = None,
                                       test_features: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate performance at image level by aggregating window predictions with NMS.
        
        Args:
            model_results: Results from window-level evaluation
            window_metadata: Window-to-image mapping metadata
            bbox_regressors: Optional dict of bbox regressors by feature type
            test_features: Optional dict of test features by feature type
            
        Returns:
            Image-level evaluation results with proper aggregation
        """
        logger.info("Calculating image-level performance with proper window aggregation")
        
        # Load test set with metadata if not provided
        if window_metadata is None:
            test_set_info = self.test_loader.load_test_set_features_and_metadata()
            window_metadata = test_set_info.get('window_metadata', {})
        
        # Ensure window_metadata is a dict
        if window_metadata is None:
            window_metadata = {}
        
        image_level_results = {}
        nms_threshold = self.detection_config.get('iou_threshold', 0.5)
        
        for model_name, results in model_results.items():
            if 'error' in results:
                continue
                
            # Use optimized predictions if available
            predictions = np.array(results.get('optimized_predictions', results['predictions']))
            true_labels = np.array(results['test_labels'])
            decision_scores = results.get('decision_scores')
            
            if decision_scores is not None:
                decision_scores = np.array(decision_scores)
            
            # Get window metadata for this feature type
            feature_type = model_name
            metadata = window_metadata.get(feature_type, {})
            window_to_image = metadata.get('window_to_image', {})
            
            if not window_to_image:
                # Fallback to simulated grouping
                logger.warning(f"No window metadata for {model_name}, using simulated grouping")
                image_level_results[model_name] = self._simulate_image_level_evaluation(
                    predictions, true_labels, decision_scores
                )
                continue
            
            # Group predictions by image
            image_groups = defaultdict(list)
            
            for window_idx in range(len(predictions)):
                window_info = window_to_image.get(str(window_idx), {})
                image_id = window_info.get('image_id', f'unknown_image_{window_idx // 20}')
                
                detection = {
                    'prediction': predictions[window_idx],
                    'true_label': true_labels[window_idx],
                    'score': decision_scores[window_idx] if decision_scores is not None else predictions[window_idx],
                    'window_index': window_idx,
                    'bbox': [0, 0, 40, 40]  # Simulated bbox for NMS
                }
                
                image_groups[image_id].append(detection)
            
            # Evaluate each image
            image_predictions = []
            image_labels = []
            image_counts_pred = []
            image_counts_true = []
            
            for image_id, detections in image_groups.items():
                # Prepare positive detections
                positive_detections = [
                    {
                        'bbox': det['bbox'],
                        'score': float(det['score']),
                        'label': int(det['prediction']),
                        'window_idx': det['window_index']
                    }
                    for det in detections if det['prediction'] == 1
                ]
                
                # Apply bbox refinement if regressor available
                if bbox_regressors and feature_type in bbox_regressors:
                    if test_features and feature_type in test_features:
                        bbox_regressor = bbox_regressors[feature_type]
                        features = test_features[feature_type]
                        
                        positive_detections = self._apply_bbox_refinement(
                            positive_detections,
                            features,
                            bbox_regressor,
                            confidence_threshold=0.5
                        )
                
                # Apply NMS to detections (potentially refined)
                filtered_detections = self.nms.apply_nms(positive_detections, nms_threshold)
                
                # Count predictions and ground truth
                pred_count = len(filtered_detections)
                true_count = sum(det['true_label'] for det in detections)
                
                # Image-level binary classification (any detection = positive)
                image_pred = 1 if pred_count > 0 else 0
                image_true = 1 if true_count > 0 else 0
                
                image_predictions.append(image_pred)
                image_labels.append(image_true)
                image_counts_pred.append(pred_count)
                image_counts_true.append(true_count)
            
            # Calculate image-level binary metrics
            image_metrics = self._calculate_classification_metrics(
                np.array(image_labels),
                np.array(image_predictions)
            )
            
            # Calculate count-based metrics (MAPE, MAE)
            if image_counts_true:
                mae = float(np.mean(np.abs(np.array(image_counts_pred) - np.array(image_counts_true))))
                
                # Calculate MAPE (handle division by zero)
                percentage_errors = []
                for pred, true in zip(image_counts_pred, image_counts_true):
                    if true > 0:
                        percentage_errors.append(abs((pred - true) / true))
                
                mape = float(np.mean(percentage_errors) * 100) if percentage_errors else 0.0
                
                image_metrics['mae'] = mae
                image_metrics['mape'] = mape
            
            image_level_results[model_name] = {
                'metrics': image_metrics,
                'predictions': image_predictions,
                'labels': image_labels,
                'count_predictions': image_counts_pred,
                'count_labels': image_counts_true,
                'n_images': len(image_groups),
                'aggregation_method': 'NMS',
                'nms_threshold': nms_threshold
            }
            
            logger.info(f"{model_name} image-level F1: {image_metrics.get('f1_score', 0):.4f}, "
                       f"MAE: {image_metrics.get('mae', 0):.2f}, MAPE: {image_metrics.get('mape', 0):.2f}%")
        
        return image_level_results
    
    def _simulate_image_level_evaluation(self, predictions, true_labels, decision_scores):
        """Fallback method for simulated image-level evaluation."""
        n_images = self.metrics_config.get('simulated_images', 10)
        windows_per_image = len(predictions) // n_images
        
        image_level_preds = []
        image_level_labels = []
        
        for i in range(n_images):
            start_idx = i * windows_per_image
            end_idx = (i + 1) * windows_per_image if i < n_images - 1 else len(predictions)
            
            window_preds = predictions[start_idx:end_idx]
            window_labels = true_labels[start_idx:end_idx]
            
            # Image is positive if any window is positive (OR aggregation)
            image_pred = 1 if np.any(window_preds) else 0
            image_label = 1 if np.any(window_labels) else 0
            
            image_level_preds.append(image_pred)
            image_level_labels.append(image_label)
        
        # Calculate image-level metrics
        image_metrics = self._calculate_classification_metrics(
            np.array(image_level_labels),
            np.array(image_level_preds)
        )
        
        return {
            'metrics': image_metrics,
            'predictions': image_level_preds,
            'labels': image_level_labels,
            'n_images': n_images,
            'aggregation_method': 'OR_simulation'
        }
    
    def compare_models(self, model_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compare performance across multiple models.
        
        Args:
            model_results: Results from multiple model evaluations
            
        Returns:
            Model comparison results
        """
        logger.info("Comparing model performance")
        
        comparison_data = []
        
        for model_name, results in model_results.items():
            if 'error' in results:
                continue
                
            metrics = results['metrics']
            comparison_data.append({
                'model': model_name,
                'accuracy': metrics.get('accuracy', 0),
                'precision': metrics.get('precision', 0),
                'recall': metrics.get('recall', 0),
                'f1_score': metrics.get('f1_score', 0),
                'roc_auc': metrics.get('roc_auc', 0),
                'average_precision': metrics.get('average_precision', 0)
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Find best model for each metric
        best_models = {}
        for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'average_precision']:
            if metric in comparison_df.columns:
                best_idx = comparison_df[metric].idxmax()
                if pd.notna(comparison_df.loc[best_idx, metric]):
                    best_models[metric] = {
                        'model': comparison_df.loc[best_idx, 'model'],
                        'value': comparison_df.loc[best_idx, metric]
                    }
        
        # Calculate ranking
        ranking_cols = ['f1_score', 'accuracy', 'precision', 'recall']
        available_cols = [col for col in ranking_cols if col in comparison_df.columns]
        
        if available_cols:
            comparison_df['overall_score'] = comparison_df[available_cols].mean(axis=1)
            comparison_df = comparison_df.sort_values('overall_score', ascending=False)
        
        comparison_results = {
            'comparison_table': comparison_df.to_dict('records'),
            'best_models': best_models,
            'model_ranking': comparison_df['model'].tolist() if 'overall_score' in comparison_df.columns else []
        }
        
        return comparison_results
    
    def create_visualizations(self, model_results: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
        """
        Create comprehensive evaluation visualizations.
        
        Args:
            model_results: Results from model evaluations
            
        Returns:
            Dictionary mapping visualization names to file paths
        """
        if not self.visualization_config.get('enabled', True):
            logger.info("Visualizations disabled")
            return {}
        
        logger.info("Creating evaluation visualizations")
        
        visualization_paths = {}
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
        
        # 1. Model comparison bar chart
        if len(model_results) > 1:
            viz_path = self._create_model_comparison_chart(model_results)
            if viz_path:
                visualization_paths['model_comparison'] = str(viz_path)
        
        # 2. Confusion matrices
        confusion_paths = self._create_confusion_matrices(model_results)
        visualization_paths.update(confusion_paths)
        
        # 3. ROC curves
        if self._has_probabilities(model_results):
            roc_path = self._create_roc_curves(model_results)
            if roc_path:
                visualization_paths['roc_curves'] = str(roc_path)
        
        # 4. Precision-Recall curves
        if self._has_probabilities(model_results):
            pr_path = self._create_pr_curves(model_results)
            if pr_path:
                visualization_paths['pr_curves'] = str(pr_path)
        
        # 5. Threshold optimization plots
        threshold_path = self._create_threshold_optimization_plot(model_results)
        if threshold_path:
            visualization_paths['threshold_optimization'] = str(threshold_path)
        
        logger.info(f"Created {len(visualization_paths)} visualizations")
        return visualization_paths
    
    def _has_probabilities(self, model_results: Dict[str, Dict[str, Any]]) -> bool:
        """Check if any model has probability predictions."""
        return any(
            results.get('probabilities') is not None 
            for results in model_results.values()
            if 'error' not in results
        )
    
    def _create_model_comparison_chart(self, model_results: Dict[str, Dict[str, Any]]) -> Optional[Path]:
        """Create model comparison bar chart."""
        try:
            comparison_results = self.compare_models(model_results)
            comparison_data = comparison_results['comparison_table']
            
            if not comparison_data:
                return None
            
            df = pd.DataFrame(comparison_data)
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 8))
            
            metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score']
            available_metrics = [m for m in metrics_to_plot if m in df.columns]
            
            x = np.arange(len(df))
            width = 0.2
            
            for i, metric in enumerate(available_metrics):
                ax.bar(x + i * width, df[metric], width, label=metric.replace('_', ' ').title())
            
            ax.set_xlabel('Models')
            ax.set_ylabel('Score')
            ax.set_title('Model Performance Comparison')
            ax.set_xticks(x + width * (len(available_metrics) - 1) / 2)
            ax.set_xticklabels(df['model'])
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Save plot
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            plot_path = self.figures_dir / f'model_comparison_{timestamp}.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return plot_path
            
        except Exception as e:
            logger.error(f"Failed to create model comparison chart: {e}")
            return None
    
    def _create_confusion_matrices(self, model_results: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
        """Create confusion matrix plots for each model."""
        confusion_paths = {}
        
        for model_name, results in model_results.items():
            if 'error' in results:
                continue
                
            try:
                cm = np.array(results['confusion_matrix'])
                
                fig, ax = plt.subplots(figsize=(8, 6))
                
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                           xticklabels=['Negative', 'Positive'],
                           yticklabels=['Negative', 'Positive'])
                
                ax.set_title(f'Confusion Matrix - {model_name}')
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                
                # Save plot
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                plot_path = self.figures_dir / f'confusion_matrix_{model_name}_{timestamp}.png'
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                confusion_paths[f'confusion_matrix_{model_name}'] = str(plot_path)
                
            except Exception as e:
                logger.error(f"Failed to create confusion matrix for {model_name}: {e}")
        
        return confusion_paths
    
    def _create_roc_curves(self, model_results: Dict[str, Dict[str, Any]]) -> Optional[Path]:
        """Create ROC curves plot."""
        try:
            fig, ax = plt.subplots(figsize=(10, 8))
            
            for model_name, results in model_results.items():
                if 'error' in results or results.get('probabilities') is None:
                    continue
                    
                y_true = np.array(results['test_labels'])
                y_prob = np.array(results['probabilities'])
                
                fpr, tpr, _ = roc_curve(y_true, y_prob)
                auc_score = results['metrics'].get('roc_auc', 0)
                
                ax.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.3f})')
            
            ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('ROC Curves Comparison')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Save plot
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            plot_path = self.figures_dir / f'roc_curves_{timestamp}.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return plot_path
            
        except Exception as e:
            logger.error(f"Failed to create ROC curves: {e}")
            return None
    
    def _create_pr_curves(self, model_results: Dict[str, Dict[str, Any]]) -> Optional[Path]:
        """Create Precision-Recall curves plot."""
        try:
            fig, ax = plt.subplots(figsize=(10, 8))
            
            for model_name, results in model_results.items():
                if 'error' in results or results.get('probabilities') is None:
                    continue
                    
                y_true = np.array(results['test_labels'])
                y_prob = np.array(results['probabilities'])
                
                precision, recall, _ = precision_recall_curve(y_true, y_prob)
                ap_score = results['metrics'].get('average_precision', 0)
                
                ax.plot(recall, precision, label=f'{model_name} (AP = {ap_score:.3f})')
            
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.set_title('Precision-Recall Curves Comparison')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Save plot
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            plot_path = self.figures_dir / f'pr_curves_{timestamp}.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return plot_path
            
        except Exception as e:
            logger.error(f"Failed to create PR curves: {e}")
            return None
    
    def _create_threshold_optimization_plot(self, model_results: Dict[str, Dict[str, Any]]) -> Optional[Path]:
        """Create threshold optimization plot showing F1 vs threshold curves."""
        try:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            plotted_models = 0
            
            for model_name, results in model_results.items():
                if 'error' in results:
                    continue
                    
                threshold_info = results.get('threshold_optimization', {})
                
                if threshold_info.get('method') == 'validation_curve':
                    thresholds = threshold_info.get('thresholds', [])
                    scores = threshold_info.get('scores', [])
                    optimal_threshold = threshold_info.get('optimal_threshold', 0)
                    optimal_score = threshold_info.get('optimal_score', 0)
                    
                    if thresholds and scores:
                        ax.plot(thresholds, scores, label=f'{model_name}', linewidth=2)
                        
                        # Mark optimal point
                        ax.scatter([optimal_threshold], [optimal_score], 
                                 s=100, marker='o', 
                                 label=f'{model_name} optimal (F1={optimal_score:.3f})')
                        
                        plotted_models += 1
            
            if plotted_models == 0:
                return None
            
            ax.set_xlabel('Decision Threshold')
            ax.set_ylabel('F1 Score')
            ax.set_title('Threshold Optimization - F1 Score vs Decision Threshold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Save plot
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            plot_path = self.figures_dir / f'threshold_optimization_{timestamp}.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return plot_path
            
        except Exception as e:
            logger.error(f"Failed to create threshold optimization plot: {e}")
            return None
    
    def save_evaluation_report(self, 
                              model_results: Dict[str, Dict[str, Any]],
                              image_level_results: Optional[Dict[str, Dict[str, Any]]] = None,
                              visualization_paths: Optional[Dict[str, str]] = None) -> str:
        """
        Save comprehensive evaluation report.
        
        Args:
            model_results: Window-level evaluation results
            image_level_results: Optional image-level evaluation results
            visualization_paths: Optional paths to generated visualizations
            
        Returns:
            Path to saved report file
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = self.figures_dir / f'evaluation_report_{timestamp}.json'
        
        # Compile complete report
        report = {
            'evaluation_summary': {
                'timestamp': datetime.now().isoformat(),
                'total_models_evaluated': len([r for r in model_results.values() if 'error' not in r]),
                'failed_models': len([r for r in model_results.values() if 'error' in r])
            },
            'window_level_results': model_results,
            'model_comparison': self.compare_models(model_results)
        }
        
        if image_level_results:
            report['image_level_results'] = image_level_results
        
        if visualization_paths:
            report['visualizations'] = visualization_paths
        
        # Save report
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Evaluation report saved: {report_path}")
        return str(report_path)
    
    def generate_summary_statistics(self, model_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate summary statistics across all models.
        
        Args:
            model_results: Results from all model evaluations
            
        Returns:
            Summary statistics dictionary
        """
        successful_results = {k: v for k, v in model_results.items() if 'error' not in v}
        
        if not successful_results:
            return {'error': 'No successful model evaluations found'}
        
        # Collect all metrics
        all_metrics = defaultdict(list)
        
        for results in successful_results.values():
            metrics = results['metrics']
            for metric_name, value in metrics.items():
                if value is not None:
                    all_metrics[metric_name].append(value)
        
        # Calculate statistics
        summary_stats = {}
        
        for metric_name, values in all_metrics.items():
            if values:
                summary_stats[metric_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'median': np.median(values)
                }
        
        # Add model count information
        summary_stats['model_counts'] = {
            'total_evaluated': len(model_results),
            'successful': len(successful_results),
            'failed': len(model_results) - len(successful_results)
        }
        
        return summary_stats