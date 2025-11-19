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
        Evaluate models on test set.
        
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
    
    def evaluate_single_image(self,
                            image_path: str,
                            ground_truth_coords: List[Tuple[float, float]],
                            predict_image_func: Callable[[str], Tuple[int, List[Tuple[float, float]]]],
                            match_threshold: float = 25.0) -> Dict[str, Any]:
        """
        Evaluate model performance on a single image.
        
        Args:
            image_path: Path to image file
            ground_truth_coords: List of (x, y) ground truth coordinates
            predict_image_func: Function that takes image_path and returns (count, coordinates)
            match_threshold: Maximum distance for matching coordinates (pixels)
            
        Returns:
            Dictionary containing detailed evaluation results for this image
        """
        try:
            predicted_count, predicted_coords = predict_image_func(image_path)
            actual_count = len(ground_truth_coords)
            
            # Match coordinates and calculate errors
            coord_errors, tp, fp, fn = self.coordinate_matcher.match_coordinates(
                predicted_coords, ground_truth_coords, match_threshold
            )
            
            # Calculate basic metrics for this image
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            results = {
                'image_path': image_path,
                'predicted_count': predicted_count,
                'actual_count': actual_count,
                'count_error': abs(predicted_count - actual_count),
                'coordinate_errors': coord_errors,
                'avg_coordinate_error': np.mean(coord_errors) if coord_errors else 0.0,
                'true_positives': tp,
                'false_positives': fp,
                'false_negatives': fn,
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'predicted_coords': predicted_coords,
                'ground_truth_coords': ground_truth_coords
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error evaluating single image {image_path}: {e}")
            return {
                'image_path': image_path,
                'error': str(e)
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
    
    def save_detailed_results(self,
                            detailed_results: List[Dict[str, Any]],
                            output_dir: str,
                            filename: str = "detailed_evaluation.json") -> None:
        """
        Save detailed per-image evaluation results.
        
        Args:
            detailed_results: List of per-image evaluation dictionaries
            output_dir: Output directory path
            filename: Output filename (default: "detailed_evaluation.json")
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        results_path = output_path / filename
        
        # Convert numpy types for JSON serialization
        serializable_results = []
        for result in detailed_results:
            serializable_result = {}
            for key, value in result.items():
                if isinstance(value, np.ndarray):
                    serializable_result[key] = value.tolist()
                elif isinstance(value, (np.integer, np.floating)):
                    serializable_result[key] = float(value)
                elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], tuple):
                    # Handle coordinate tuples
                    serializable_result[key] = [list(coord) for coord in value]
                else:
                    serializable_result[key] = value
            serializable_results.append(serializable_result)
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        self.logger.info(f"Saved detailed evaluation results to {results_path}")
    
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
    
    def save_comparison_results(self,
                               main_metrics: Dict[str, float],
                               stacked_metrics: Dict[str, float],
                               comparison: Dict[str, float],
                               output_dir: str) -> None:
        """
        Save side-by-side comparison results.
        
        Args:
            main_metrics: Metrics from main pipeline
            stacked_metrics: Metrics from stacked model
            comparison: Improvement metrics
            output_dir: Output directory
        """
        output_path = Path(output_dir)
        evaluations_dir = output_path / "evaluations"
        evaluations_dir.mkdir(parents=True, exist_ok=True)
        
        # Save combined results
        combined_results = {
            'main_pipeline': main_metrics,
            'stacked_model': stacked_metrics,
            'comparison': comparison
        }
        
        results_path = evaluations_dir / "comparison_results.json"
        with open(results_path, 'w') as f:
            json.dump(combined_results, f, indent=2)
        
        self.logger.info(f"Saved comparison results to {results_path}")
        
        # Generate comparison report
        self.generate_comparison_report(combined_results, output_dir)
    
    def generate_comparison_report(self,
                                  results: Dict[str, Any],
                                  output_dir: str) -> str:
        """
        Generate human-readable comparison report.
        
        Args:
            results: Combined results dictionary
            output_dir: Output directory
            
        Returns:
            Path to generated report
        """
        output_path = Path(output_dir)
        evaluations_dir = output_path / "evaluations"
        evaluations_dir.mkdir(parents=True, exist_ok=True)
        
        report_path = evaluations_dir / "comparison_report.txt"
        
        main = results['main_pipeline']
        stacked = results['stacked_model']
        comp = results['comparison']
        
        with open(report_path, 'w') as f:
            f.write("Nurdle Detection Pipeline - Model Comparison Report\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("MAIN PIPELINE (SVM + SVR):\n")
            f.write("-" * 30 + "\n")
            f.write(f"  F1-Score: {main['f1_score']:.3f}\n")
            f.write(f"  Precision: {main['precision']:.3f}\n")
            f.write(f"  Recall: {main['recall']:.3f}\n")
            f.write(f"  Avg Coordinate Error: {main['avg_coordinate_error']:.1f} pixels\n")
            f.write(f"  Count MAE: {main['count_mae']:.1f}\n\n")
            
            f.write("STACKED MODEL (SVM + SVR + Meta-Learner):\n")
            f.write("-" * 30 + "\n")
            f.write(f"  F1-Score: {stacked['f1_score']:.3f}\n")
            f.write(f"  Precision: {stacked['precision']:.3f}\n")
            f.write(f"  Recall: {stacked['recall']:.3f}\n")
            f.write(f"  Avg Coordinate Error: {stacked['avg_coordinate_error']:.1f} pixels\n")
            f.write(f"  Count MAE: {stacked['count_mae']:.1f}\n\n")
            
            f.write("IMPROVEMENT (Stacked vs Main):\n")
            f.write("-" * 30 + "\n")
            f.write(f"  F1-Score: {comp['f1_improvement']:+.3f} ({comp['f1_improvement']/main['f1_score']*100:+.1f}%)\n")
            f.write(f"  Precision: {comp['precision_improvement']:+.3f}\n")
            f.write(f"  Recall: {comp['recall_improvement']:+.3f}\n")
            f.write(f"  Coordinate Error: {comp['coord_error_improvement']:+.1f} pixels\n\n")
            
            # Determine winner
            if comp['f1_improvement'] > 0.01 and comp['coord_error_improvement'] > 0.5:
                winner = "Stacked Model (significant improvement)"
            elif comp['f1_improvement'] < -0.01 or comp['coord_error_improvement'] < -0.5:
                winner = "Main Pipeline (stacking degraded performance)"
            else:
                winner = "Similar performance (no clear winner)"
            
            f.write(f"CONCLUSION: {winner}\n")
        
        self.logger.info(f"Generated comparison report: {report_path}")
        return str(report_path)
    
    def _log_evaluation_results(self, metrics: Dict[str, float]) -> None:
        """Log evaluation results to console."""
        self.logger.info("Evaluation Results:")
        self.logger.info(f"  Count Accuracy: {metrics.get('count_accuracy', 0):.3f}")
        self.logger.info(f"  Count MAE: {metrics.get('count_mae', 0):.3f}")
        self.logger.info(f"  Avg Coordinate Error: {metrics.get('avg_coordinate_error', 0):.1f} pixels")
        self.logger.info(f"  Precision: {metrics.get('precision', 0):.3f}")
        self.logger.info(f"  Recall: {metrics.get('recall', 0):.3f}")
        self.logger.info(f"  F1-Score: {metrics.get('f1_score', 0):.3f}")