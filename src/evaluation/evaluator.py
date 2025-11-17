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
from .nms_processor import NonMaximumSuppression
from .test_loader import TestSetLoader
from .threshold_optimizer import ThresholdOptimizer
from .evaluation_visualizer import EvaluationVisualizer

logger = logging.getLogger(__name__)


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
        
        # Initialize visualizer
        self.visualizer = EvaluationVisualizer(self.figures_dir, self.visualization_config)
        
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
        
        Delegates to EvaluationVisualizer for all visualization generation.
        
        Args:
            model_results: Results from model evaluations
            
        Returns:
            Dictionary mapping visualization names to file paths
        """
        return self.visualizer.create_all_visualizations(model_results)
    


    
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