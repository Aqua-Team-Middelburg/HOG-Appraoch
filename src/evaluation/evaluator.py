"""
Main Evaluation Module for Nurdle Detection Pipeline
===================================================

This module handles the complete evaluation process for trained models,
including prediction, coordinate matching, and metrics calculation.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Callable
import numpy as np

from .metrics import EvaluationMetrics, CoordinateMatching
from .nms import NonMaximumSuppression


class ModelEvaluator:
    """
    Main evaluator for nurdle detection models.
    
    Handles evaluation of both classification (count prediction) and
    coordinate prediction tasks.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize evaluator with optional logger."""
        self.logger = logger or logging.getLogger(__name__)
        self.metrics_calculator = EvaluationMetrics()
        self.coordinate_matcher = CoordinateMatching()
        self.nms = NonMaximumSuppression()
        
    def evaluate_models(self, 
                       test_annotations: List[Any],
                       predict_image_func: Callable[[str], Tuple[int, List[Tuple[float, float]]]]) -> Dict[str, float]:
        """
        Evaluate models on test set (combined two-stage approach).
        
        Args:
            test_annotations: List of ImageAnnotation objects for testing
            predict_image_func: Function that takes image_path and returns (count, coordinates)
        
        Returns:
            Dictionary of evaluation metrics
        """
        self.logger.info("Evaluating models...")
        
        count_predictions = []
        count_ground_truth = []
        coordinate_errors = []
        detection_stats = {
            'true_positives': 0,
            'false_positives': 0,
            'false_negatives': 0
        }
        
        for annotation in test_annotations:
            try:
                predicted_count, predicted_coords = predict_image_func(annotation.image_path)
                actual_count = annotation.nurdle_count
                actual_coords = annotation.coordinates
                
                count_predictions.append(predicted_count)
                count_ground_truth.append(actual_count)
                
                # Calculate coordinate matching and detection statistics
                # Convert coordinates to list format for matching
                actual_coords_list = actual_coords.tolist() if hasattr(actual_coords, 'tolist') else list(actual_coords)
                
                if len(actual_coords_list) > 0 or len(predicted_coords) > 0:
                    coord_errors, tp, fp, fn = self.coordinate_matcher.match_coordinates(
                        predicted_coords, actual_coords_list
                    )
                    
                    coordinate_errors.extend(coord_errors)
                    detection_stats['true_positives'] += tp
                    detection_stats['false_positives'] += fp
                    detection_stats['false_negatives'] += fn
                
            except Exception as e:
                self.logger.error(f"Error evaluating {annotation.image_path}: {e}")
                continue
        
        # Calculate comprehensive metrics
        metrics = self.metrics_calculator.calculate_all_metrics(
            count_ground_truth=count_ground_truth,
            count_predictions=count_predictions,
            coordinate_errors=coordinate_errors,
            detection_stats=detection_stats,
            n_test_images=len(test_annotations)
        )
        
        # Log results
        self._log_evaluation_results(metrics)
        
        return metrics
    
    def evaluate_models_detailed(self,
                                test_annotations: List[Any],
                                prediction_engine: Any,
                                data_loader: Any) -> Dict[str, Any]:
        """
        Evaluate models with detailed breakdown: SVM only, SVR only, and combined.
        
        Args:
            test_annotations: List of ImageAnnotation objects for testing
            prediction_engine: PredictionEngine instance with trained models
            data_loader: DataLoader instance for loading images
        
        Returns:
            Dictionary with separate metrics for SVM, SVR, and combined approach
        """
        self.logger.info("Performing detailed model evaluation (SVM + SVR + Combined)...")
        
        results = {
            'svm_only': self._evaluate_svm_only(test_annotations, prediction_engine, data_loader),
            'svr_refinement': {},  # Will be populated with SVR contribution analysis
            'combined': {}  # Will be populated with full pipeline metrics
        }
        
        # Evaluate combined approach (already done in evaluate_models)
        self.logger.info("Evaluating combined two-stage approach...")
        count_predictions = []
        count_ground_truth = []
        coordinate_errors = []
        svr_offset_magnitudes = []  # Track SVR offset magnitudes
        detection_stats = {
            'true_positives': 0,
            'false_positives': 0,
            'false_negatives': 0
        }
        
        for annotation in test_annotations:
            try:
                # Get combined predictions
                predicted_count, predicted_coords = prediction_engine.predict_image_from_path(
                    annotation.image_path,
                    data_loader,
                    prediction_engine.nms.default_iou_threshold
                )
                
                actual_count = annotation.nurdle_count
                actual_coords = annotation.coordinates
                
                count_predictions.append(predicted_count)
                count_ground_truth.append(actual_count)
                
                # Calculate coordinate matching and detection statistics
                actual_coords_list = actual_coords.tolist() if hasattr(actual_coords, 'tolist') else list(actual_coords)
                
                if len(actual_coords_list) > 0 or len(predicted_coords) > 0:
                    coord_errors, tp, fp, fn = self.coordinate_matcher.match_coordinates(
                        predicted_coords, actual_coords_list
                    )
                    
                    coordinate_errors.extend(coord_errors)
                    detection_stats['true_positives'] += tp
                    detection_stats['false_positives'] += fp
                    detection_stats['false_negatives'] += fn
                
            except Exception as e:
                self.logger.error(f"Error evaluating combined approach for {annotation.image_path}: {e}")
                continue
        
        results['combined'] = self.metrics_calculator.calculate_all_metrics(
            count_ground_truth=count_ground_truth,
            count_predictions=count_predictions,
            coordinate_errors=coordinate_errors,
            detection_stats=detection_stats,
            n_test_images=len(test_annotations)
        )
        
        # Analyze SVR contribution
        results['svr_refinement'] = self._analyze_svr_contribution(
            test_annotations, prediction_engine, data_loader
        )
        
        # Log detailed results
        self._log_detailed_evaluation_results(results)
        
        return results
    
    def _evaluate_svm_only(self,
                          test_annotations: List[Any],
                          prediction_engine: Any,
                          data_loader: Any) -> Dict[str, Any]:
        """
        Evaluate SVM classifier detection performance only (no coordinate accuracy).
        
        Measures: Precision, Recall, F1, TP, FP, FN after NMS filtering.
        Does NOT measure coordinate accuracy since SVM only does classification.
        
        Args:
            test_annotations: List of ImageAnnotation objects
            prediction_engine: PredictionEngine instance
            data_loader: DataLoader instance
        
        Returns:
            Dictionary with SVM detection metrics only
        """
        self.logger.info("Evaluating SVM classifier detection performance...")
        
        detection_stats = {
            'true_positives': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'total_svm_positives': 0,  # Before NMS
            'total_after_nms': 0        # After NMS
        }
        
        for annotation in test_annotations:
            try:
                image = data_loader.load_image(annotation.image_path)
                h, w = image.shape[:2]
                
                window_size = prediction_engine.feature_extractor.window_size
                window_stride = prediction_engine.feature_extractor.window_stride
                
                positive_windows = []
                
                # Slide window and collect SVM positive predictions
                for y in range(0, h - window_size[1] + 1, window_stride):
                    for x in range(0, w - window_size[0] + 1, window_stride):
                        window = image[y:y+window_size[1], x:x+window_size[0]]
                        
                        if window.shape[:2] == tuple(window_size):
                            features = prediction_engine.feature_extractor.extract_hog_lbp_features(window)
                            features_2d = features.reshape(1, -1)
                            is_nurdle = prediction_engine.model_trainer.svm_classifier.predict(features_2d)[0]
                            
                            if is_nurdle:
                                confidence = prediction_engine.model_trainer.svm_classifier.predict_proba(features_2d)[0][1]
                                box = (x, y, x + window_size[0], y + window_size[1], confidence)
                                window_center_x = x + window_size[0] / 2.0
                                window_center_y = y + window_size[1] / 2.0
                                positive_windows.append((box, window_center_x, window_center_y))
                
                detection_stats['total_svm_positives'] += len(positive_windows)
                
                # Apply NMS on positive windows
                if positive_windows:
                    boxes_for_nms = [pw[0] for pw in positive_windows]
                    filtered_boxes = prediction_engine.nms.apply_nms_with_boxes(
                        boxes_for_nms,
                        iou_threshold=prediction_engine.nms.default_iou_threshold
                    )
                    
                    # Get window centers for filtered boxes
                    svm_predictions = []
                    for filtered_box in filtered_boxes:
                        for box, cx, cy in positive_windows:
                            if box == filtered_box:
                                svm_predictions.append((cx, cy))
                                break
                else:
                    svm_predictions = []
                
                detection_stats['total_after_nms'] += len(svm_predictions)
                
                # Calculate detection matching (TP, FP, FN only - no coordinate errors)
                actual_coords = annotation.coordinates
                actual_coords_list = actual_coords.tolist() if hasattr(actual_coords, 'tolist') else list(actual_coords)
                
                if len(actual_coords_list) > 0 or len(svm_predictions) > 0:
                    _, tp, fp, fn = self.coordinate_matcher.match_coordinates(
                        svm_predictions, actual_coords_list
                    )
                    
                    detection_stats['true_positives'] += tp
                    detection_stats['false_positives'] += fp
                    detection_stats['false_negatives'] += fn
                
            except Exception as e:
                self.logger.error(f"Error evaluating SVM-only for {annotation.image_path}: {e}")
                continue
        
        # Calculate only detection metrics (no coordinate or count metrics)
        return self.metrics_calculator.calculate_detection_metrics(detection_stats)
    
    def _analyze_svr_contribution(self,
                                 test_annotations: List[Any],
                                 prediction_engine: Any,
                                 data_loader: Any) -> Dict[str, Any]:
        """
        Analyze SVR's contribution to coordinate refinement.
        
        Args:
            test_annotations: List of ImageAnnotation objects
            prediction_engine: PredictionEngine instance
            data_loader: DataLoader instance
        
        Returns:
            Dictionary with SVR contribution analysis
        """
        self.logger.info("Analyzing SVR coordinate refinement contribution...")
        
        offset_magnitudes = []
        improvements = []  # How much SVR improved coordinate error
        
        for annotation in test_annotations:
            try:
                image = data_loader.load_image(annotation.image_path)
                h, w = image.shape[:2]
                
                window_size = prediction_engine.feature_extractor.window_size
                window_stride = prediction_engine.feature_extractor.window_stride
                
                actual_coords = annotation.coordinates
                actual_coords_list = actual_coords.tolist() if hasattr(actual_coords, 'tolist') else list(actual_coords)
                
                # Collect positive windows with both SVM and SVR predictions
                for y in range(0, h - window_size[1] + 1, window_stride):
                    for x in range(0, w - window_size[0] + 1, window_stride):
                        window = image[y:y+window_size[1], x:x+window_size[0]]
                        
                        if window.shape[:2] == tuple(window_size):
                            features = prediction_engine.feature_extractor.extract_hog_lbp_features(window)
                            features_2d = features.reshape(1, -1)
                            is_nurdle = prediction_engine.model_trainer.svm_classifier.predict(features_2d)[0]
                            
                            if is_nurdle:
                                # Window center (SVM prediction)
                                window_center_x = x + window_size[0] / 2.0
                                window_center_y = y + window_size[1] / 2.0
                                
                                # SVR refinement
                                offsets = prediction_engine.model_trainer.svr_regressor.predict(features_2d)[0]
                                offset_x, offset_y = offsets[0], offsets[1]
                                offset_magnitude = np.sqrt(offset_x**2 + offset_y**2)
                                offset_magnitudes.append(offset_magnitude)
                                
                                # Refined coordinates
                                refined_x = window_center_x + offset_x
                                refined_y = window_center_y + offset_y
                                
                                # Find nearest ground truth
                                if actual_coords_list:
                                    min_dist_before = float('inf')
                                    min_dist_after = float('inf')
                                    
                                    for gt_x, gt_y in actual_coords_list:
                                        dist_before = np.sqrt((window_center_x - gt_x)**2 + (window_center_y - gt_y)**2)
                                        dist_after = np.sqrt((refined_x - gt_x)**2 + (refined_y - gt_y)**2)
                                        
                                        if dist_before < min_dist_before:
                                            min_dist_before = dist_before
                                        if dist_after < min_dist_after:
                                            min_dist_after = dist_after
                                    
                                    improvement = min_dist_before - min_dist_after
                                    improvements.append(improvement)
                
            except Exception as e:
                self.logger.error(f"Error analyzing SVR for {annotation.image_path}: {e}")
                continue
        
        return {
            'avg_offset_magnitude': float(np.mean(offset_magnitudes)) if offset_magnitudes else 0.0,
            'median_offset_magnitude': float(np.median(offset_magnitudes)) if offset_magnitudes else 0.0,
            'max_offset_magnitude': float(np.max(offset_magnitudes)) if offset_magnitudes else 0.0,
            'avg_improvement': float(np.mean(improvements)) if improvements else 0.0,
            'median_improvement': float(np.median(improvements)) if improvements else 0.0,
            'positive_improvements': int(np.sum(np.array(improvements) > 0)) if improvements else 0,
            'negative_improvements': int(np.sum(np.array(improvements) < 0)) if improvements else 0,
            'total_predictions': len(improvements)
        }
    

    
    def save_evaluation_results(self, 
                              metrics: Dict[str, float], 
                              output_dir: str,
                              filename: str = "evaluation_results.json") -> None:
        """
        Save evaluation results to organized folder structure.
        
        Args:
            metrics: Dictionary of evaluation metrics
            output_dir: Output directory path
            filename: Output filename (default: "evaluation_results.json")
        """
        output_path = Path(output_dir)
        
        # Save evaluation results to evaluations subfolder
        evaluations_dir = output_path / "evaluations"
        evaluations_dir.mkdir(parents=True, exist_ok=True)
        
        results_path = evaluations_dir / filename
        
        # Convert numpy types to Python types for JSON serialization
        serializable_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, np.ndarray):
                serializable_metrics[key] = value.tolist()
            elif isinstance(value, (np.integer, np.floating)):
                serializable_metrics[key] = float(value)
            else:
                serializable_metrics[key] = value
        
        with open(results_path, 'w') as f:
            json.dump(serializable_metrics, f, indent=2)
        
        self.logger.info(f"Saved evaluation results to {results_path}")
    

    
    def generate_evaluation_report(self, 
                                 metrics: Dict[str, float],
                                 output_dir: str) -> str:
        """
        Generate a human-readable evaluation report in organized folder structure.
        
        Args:
            metrics: Dictionary of evaluation metrics
            output_dir: Output directory path
            
        Returns:
            Path to generated report file
        """
        output_path = Path(output_dir)
        
        # Save evaluation report to evaluations subfolder
        evaluations_dir = output_path / "evaluations"
        evaluations_dir.mkdir(parents=True, exist_ok=True)
        
        report_path = evaluations_dir / "evaluation_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("Nurdle Detection Pipeline - Evaluation Report\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("COUNT PREDICTION METRICS:\n")
            f.write(f"  Accuracy: {metrics.get('count_accuracy', 0):.3f}\n")
            f.write(f"  Mean Absolute Error: {metrics.get('count_mae', 0):.3f}\n")
            f.write(f"  Root Mean Square Error: {metrics.get('count_rmse', 0):.3f}\n\n")
            
            f.write("COORDINATE PREDICTION METRICS:\n")
            f.write(f"  Average Coordinate Error: {metrics.get('avg_coordinate_error', 0):.1f} pixels\n")
            f.write(f"  Median Coordinate Error: {metrics.get('median_coordinate_error', 0):.1f} pixels\n")
            f.write(f"  Max Coordinate Error: {metrics.get('max_coordinate_error', 0):.1f} pixels\n\n")
            
            f.write("DETECTION METRICS:\n")
            f.write(f"  Precision: {metrics.get('precision', 0):.3f}\n")
            f.write(f"  Recall: {metrics.get('recall', 0):.3f}\n")
            f.write(f"  F1-Score: {metrics.get('f1_score', 0):.3f}\n\n")
            
            f.write("DATASET INFORMATION:\n")
            f.write(f"  Test Images: {metrics.get('n_test_images', 0)}\n")
            f.write(f"  True Positives: {metrics.get('true_positives', 0)}\n")
            f.write(f"  False Positives: {metrics.get('false_positives', 0)}\n")
            f.write(f"  False Negatives: {metrics.get('false_negatives', 0)}\n")
        
        self.logger.info(f"Generated evaluation report: {report_path}")
        return str(report_path)
    

    
    def generate_evaluation_visualizations(self,
                                         test_annotations: List[Any],
                                         predict_image_func: Callable[[str], Tuple[int, List[Tuple[float, float]]]],
                                         data_loader: Any,
                                         metrics: Dict[str, float],
                                         output_dir: str) -> None:
        """
        Generate comprehensive evaluation visualizations.
        
        Args:
            test_annotations: List of test ImageAnnotation objects
            predict_image_func: Function for generating predictions
            data_loader: DataLoader instance for loading images
            metrics: Computed metrics dictionary
            output_dir: Directory to save visualizations
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        from pathlib import Path
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 1. Generate count accuracy bar chart
        self.logger.info("Generating count accuracy visualization...")
        self._generate_count_accuracy_chart(metrics, output_path)
        
        # 2. Generate coordinate error histogram
        self.logger.info("Generating coordinate error histogram...")
        self._generate_coordinate_error_histogram(test_annotations, predict_image_func, output_path)
        
        # 3. Generate detection examples for all test images
        self.logger.info("Generating detection examples...")
        predictions_dir = output_path / 'predictions'
        predictions_dir.mkdir(exist_ok=True)
        
        for annotation in test_annotations:
            try:
                predicted_count, predicted_coords = predict_image_func(annotation.image_path)
                actual_coords = annotation.coordinates.tolist() if hasattr(annotation.coordinates, 'tolist') else list(annotation.coordinates)
                
                self._create_detection_visualization(
                    image_path=annotation.image_path,
                    ground_truth_coords=actual_coords,
                    predicted_coords=predicted_coords,
                    data_loader=data_loader,
                    output_path=predictions_dir / f"{Path(annotation.image_path).stem}_detection.png"
                )
            except Exception as e:
                self.logger.warning(f"Could not generate detection visualization for {annotation.image_path}: {e}")
        
        # 4. Generate metrics summary visualization
        self.logger.info("Generating metrics summary...")
        self._generate_metrics_summary(metrics, output_path)
        
        self.logger.info(f"All evaluation visualizations saved to {output_path}")
    
    def _generate_count_accuracy_chart(self, metrics: Dict[str, float], output_dir: Path) -> None:
        """Generate bar chart showing count prediction metrics."""
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        metric_names = ['Accuracy', 'MAE', 'RMSE']
        metric_values = [
            metrics.get('count_accuracy', 0),
            metrics.get('count_mae', 0),
            metrics.get('count_rmse', 0)
        ]
        
        colors = ['#2ecc71', '#e74c3c', '#f39c12']
        bars = ax.bar(metric_names, metric_values, color=colors, alpha=0.7, edgecolor='black')
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        ax.set_ylabel('Value', fontsize=12)
        ax.set_title('Count Prediction Metrics', fontsize=14, fontweight='bold')
        ax.set_ylim(0, max(metric_values) * 1.15)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'count_metrics.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _generate_coordinate_error_histogram(self, test_annotations: List[Any],
                                           predict_image_func: Callable,
                                           output_dir: Path) -> None:
        """Generate histogram of coordinate prediction errors."""
        import matplotlib.pyplot as plt
        
        all_errors = []
        
        for annotation in test_annotations:
            try:
                predicted_count, predicted_coords = predict_image_func(annotation.image_path)
                actual_coords = annotation.coordinates.tolist() if hasattr(annotation.coordinates, 'tolist') else list(annotation.coordinates)
                
                coord_errors, _, _, _ = self.coordinate_matcher.match_coordinates(
                    predicted_coords, actual_coords
                )
                all_errors.extend(coord_errors)
            except Exception as e:
                continue
        
        if not all_errors:
            self.logger.warning("No coordinate errors to visualize")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.hist(all_errors, bins=30, color='#3498db', alpha=0.7, edgecolor='black')
        
        # Add statistics lines
        mean_error = np.mean(all_errors)
        median_error = np.median(all_errors)
        
        ax.axvline(mean_error, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_error:.2f}px')
        ax.axvline(median_error, color='green', linestyle='--', linewidth=2, label=f'Median: {median_error:.2f}px')
        
        ax.set_xlabel('Coordinate Error (pixels)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Distribution of Coordinate Prediction Errors', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'coordinate_error_histogram.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _create_detection_visualization(self, image_path: str,
                                      ground_truth_coords: List[Tuple[float, float]],
                                      predicted_coords: List[Tuple[float, float]],
                                      data_loader: Any,
                                      output_path: Path) -> None:
        """Create visualization showing ground truth vs predictions for a single image."""
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        
        # Load image
        image = data_loader.load_image(image_path)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(image, cmap='gray')
        
        # Draw ground truth (green circles)
        for x, y in ground_truth_coords:
            circle = plt.Circle((x, y), radius=15, color='lime', fill=False, linewidth=2)
            ax.add_patch(circle)
        
        # Draw predictions (red circles)
        for x, y in predicted_coords:
            circle = plt.Circle((x, y), radius=12, color='red', fill=False, linewidth=2, linestyle='--')
            ax.add_patch(circle)
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='lime', markersize=10, label='Ground Truth'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Predictions', linestyle='--')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        # Add title with counts
        img_name = Path(image_path).stem
        ax.set_title(f'{img_name}\nGT: {len(ground_truth_coords)} | Predicted: {len(predicted_coords)}',
                    fontsize=12, fontweight='bold')
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _generate_metrics_summary(self, metrics: Dict[str, float], output_dir: Path) -> None:
        """Generate comprehensive metrics summary visualization."""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Evaluation Metrics Summary', fontsize=16, fontweight='bold')
        
        # Detection metrics (top-left)
        ax1 = axes[0, 0]
        detection_names = ['Precision', 'Recall', 'F1-Score']
        detection_values = [
            metrics.get('precision', 0),
            metrics.get('recall', 0),
            metrics.get('f1_score', 0)
        ]
        bars1 = ax1.bar(detection_names, detection_values, color=['#3498db', '#e74c3c', '#2ecc71'], alpha=0.7, edgecolor='black')
        for bar, value in zip(bars1, detection_values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height, f'{value:.3f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax1.set_ylabel('Score', fontsize=11)
        ax1.set_title('Detection Performance', fontsize=12, fontweight='bold')
        ax1.set_ylim(0, 1.1)
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Count metrics (top-right)
        ax2 = axes[0, 1]
        count_names = ['Accuracy', 'MAE', 'RMSE']
        count_values = [
            metrics.get('count_accuracy', 0),
            metrics.get('count_mae', 0) / 10,  # Scale for visualization
            metrics.get('count_rmse', 0) / 10
        ]
        bars2 = ax2.bar(count_names, count_values, color=['#9b59b6', '#f39c12', '#e67e22'], alpha=0.7, edgecolor='black')
        for bar, value, actual in zip(bars2, count_values, [metrics.get('count_accuracy', 0), metrics.get('count_mae', 0), metrics.get('count_rmse', 0)]):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height, f'{actual:.2f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax2.set_ylabel('Value (scaled)', fontsize=11)
        ax2.set_title('Count Prediction Metrics', fontsize=12, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Coordinate error stats (bottom-left)
        ax3 = axes[1, 0]
        coord_names = ['Mean', 'Median', 'Std', 'Max']
        coord_values = [
            metrics.get('avg_coordinate_error', 0),
            metrics.get('median_coordinate_error', 0),
            metrics.get('std_coordinate_error', 0),
            metrics.get('max_coordinate_error', 0)
        ]
        bars3 = ax3.bar(coord_names, coord_values, color='#1abc9c', alpha=0.7, edgecolor='black')
        for bar, value in zip(bars3, coord_values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height, f'{value:.1f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax3.set_ylabel('Error (pixels)', fontsize=11)
        ax3.set_title('Coordinate Error Statistics', fontsize=12, fontweight='bold')
        ax3.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Detection counts (bottom-right)
        ax4 = axes[1, 1]
        detection_count_names = ['True Pos', 'False Pos', 'False Neg']
        detection_count_values = [
            metrics.get('true_positives', 0),
            metrics.get('false_positives', 0),
            metrics.get('false_negatives', 0)
        ]
        bars4 = ax4.bar(detection_count_names, detection_count_values,
                       color=['#2ecc71', '#e74c3c', '#f39c12'], alpha=0.7, edgecolor='black')
        for bar, value in zip(bars4, detection_count_values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height, f'{int(value)}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax4.set_ylabel('Count', fontsize=11)
        ax4.set_title('Detection Breakdown', fontsize=12, fontweight='bold')
        ax4.grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'metrics_summary.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _log_evaluation_results(self, metrics: Dict[str, float]) -> None:
        """Log evaluation results to console."""
        self.logger.info("Evaluation Results:")
        self.logger.info(f"  Count Accuracy: {metrics.get('count_accuracy', 0):.3f}")
        self.logger.info(f"  Count MAE: {metrics.get('count_mae', 0):.3f}")
        self.logger.info(f"  Avg Coordinate Error: {metrics.get('avg_coordinate_error', 0):.1f} pixels")
        self.logger.info(f"  Precision: {metrics.get('precision', 0):.3f}")
        self.logger.info(f"  Recall: {metrics.get('recall', 0):.3f}")
        self.logger.info(f"  F1-Score: {metrics.get('f1_score', 0):.3f}")
    
    def _log_detailed_evaluation_results(self, results: Dict[str, Any]) -> None:
        """Log detailed evaluation results with component breakdown."""
        self.logger.info("="*80)
        self.logger.info("DETAILED EVALUATION RESULTS")
        self.logger.info("="*80)
        
        # SVM Only
        self.logger.info("\n[1] SVM CLASSIFIER DETECTION PERFORMANCE:")
        svm = results['svm_only']
        self.logger.info(f"  Precision: {svm.get('precision', 0):.3f}")
        self.logger.info(f"  Recall: {svm.get('recall', 0):.3f}")
        self.logger.info(f"  F1-Score: {svm.get('f1_score', 0):.3f}")
        self.logger.info(f"  True Positives: {svm.get('true_positives', 0)}")
        self.logger.info(f"  False Positives: {svm.get('false_positives', 0)}")
        self.logger.info(f"  False Negatives: {svm.get('false_negatives', 0)}")
        self.logger.info(f"  Total SVM Positives (before NMS): {svm.get('total_svm_positives', 0)}")
        self.logger.info(f"  Total After NMS: {svm.get('total_after_nms', 0)}")
        
        # SVR Contribution
        self.logger.info("\n[2] SVR COORDINATE REFINEMENT ANALYSIS:")
        svr = results['svr_refinement']
        self.logger.info(f"  Avg Offset Magnitude: {svr.get('avg_offset_magnitude', 0):.2f} pixels")
        self.logger.info(f"  Median Offset Magnitude: {svr.get('median_offset_magnitude', 0):.2f} pixels")
        self.logger.info(f"  Max Offset Magnitude: {svr.get('max_offset_magnitude', 0):.2f} pixels")
        self.logger.info(f"  Avg Improvement: {svr.get('avg_improvement', 0):+.2f} pixels")
        pos_pct = 100 * svr.get('positive_improvements', 0) / svr.get('total_predictions', 1)
        neg_pct = 100 * svr.get('negative_improvements', 0) / svr.get('total_predictions', 1)
        self.logger.info(f"  Predictions Improved: {svr.get('positive_improvements', 0)} ({pos_pct:.1f}%)")
        self.logger.info(f"  Predictions Degraded: {svr.get('negative_improvements', 0)} ({neg_pct:.1f}%)")
        
        # Combined
        self.logger.info("\n[3] COMBINED TWO-STAGE APPROACH (SVM + SVR):")
        combined = results['combined']
        self.logger.info(f"  Precision: {combined.get('precision', 0):.3f}")
        self.logger.info(f"  Recall: {combined.get('recall', 0):.3f}")
        self.logger.info(f"  F1-Score: {combined.get('f1_score', 0):.3f}")
        self.logger.info(f"  Count MAE: {combined.get('count_mae', 0):.1f}")
        self.logger.info(f"  Avg Coordinate Error: {combined.get('avg_coordinate_error', 0):.1f} pixels")
        self.logger.info(f"  True Positives: {combined.get('true_positives', 0)}")
        self.logger.info(f"  False Positives: {combined.get('false_positives', 0)}")
        self.logger.info(f"  False Negatives: {combined.get('false_negatives', 0)}")
        
        # Comparison
        self.logger.info("\n[4] KEY INSIGHTS:")
        f1_change = combined.get('f1_score', 0) - svm.get('f1_score', 0)
        precision_change = combined.get('precision', 0) - svm.get('precision', 0)
        recall_change = combined.get('recall', 0) - svm.get('recall', 0)
        self.logger.info(f"  F1-Score Change (SVM->Combined): {f1_change:+.3f}")
        self.logger.info(f"  Precision Change: {precision_change:+.3f}")
        self.logger.info(f"  Recall Change: {recall_change:+.3f}")
        if svr.get('avg_improvement', 0) < 0:
            self.logger.info(f"  WARNING: SVR degrades coordinates by {abs(svr.get('avg_improvement', 0)):.2f}px on average")
        self.logger.info("="*80)