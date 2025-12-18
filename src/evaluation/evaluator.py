"""
Model Evaluation for Nurdle Detection Pipeline
==============================================

Evaluates trained ensemble SVR models for nurdle count prediction.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
import numpy as np

from .metrics import EvaluationMetrics


class ModelEvaluator:
    """
    Evaluator for ensemble SVR nurdle count prediction models.
    Handles evaluation of count prediction metrics.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.metrics_calculator = EvaluationMetrics()

    def evaluate(self,
                 test_annotations: List[Any],
                 predict_image_func: Callable[[str], int]) -> Dict[str, float]:
        """
        Evaluate count predictions on test set.

        Args:
            test_annotations: List of ImageAnnotation objects for testing
            predict_image_func: Function that takes image_path and returns predicted count

        Returns:
            Dictionary of evaluation metrics
        """
        self.logger.info("Evaluating count predictions...")

        count_predictions = []
        count_ground_truth = []

        for annotation in test_annotations:
            try:
                predicted_count = predict_image_func(annotation.image_path)
                actual_count = annotation.nurdle_count
                count_predictions.append(predicted_count)
                count_ground_truth.append(actual_count)
            except Exception as e:
                self.logger.error(f"Error evaluating {annotation.image_path}: {e}")
                continue

        metrics = self.metrics_calculator.calculate_count_metrics(
            ground_truth=count_ground_truth,
            predictions=count_predictions,
            n_test_images=len(test_annotations)
        )

        self._log_evaluation_results(metrics)
        return metrics

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
        evaluations_dir = output_path / "evaluations"
        evaluations_dir.mkdir(parents=True, exist_ok=True)
        report_path = evaluations_dir / "evaluation_report.txt"

        with open(report_path, 'w') as f:
            f.write("Nurdle Count Prediction - Evaluation Report\n")
            f.write("=" * 50 + "\n\n")
            f.write("COUNT PREDICTION METRICS:\n")
            f.write(f"  Accuracy: {metrics.get('count_accuracy', 0):.3f}\n")
            f.write(f"  Mean Absolute Error: {metrics.get('count_mae', 0):.3f}\n")
            f.write(f"  Root Mean Square Error: {metrics.get('count_rmse', 0):.3f}\n")
            f.write(f"  Bias: {metrics.get('count_bias', 0):.3f}\n")
            f.write("\nDATASET INFORMATION:\n")
            f.write(f"  Test Images: {metrics.get('n_test_images', 0)}\n")

        self.logger.info(f"Generated evaluation report: {report_path}")
        return str(report_path)

    def _log_evaluation_results(self, metrics: Dict[str, float]) -> None:
        """Log evaluation results to console."""
        self.logger.info("Evaluation Results:")
        self.logger.info(f"  Count MAE: {metrics.get('mae', 0):.3f}")
        self.logger.info(f"  Count RMSE: {metrics.get('rmse', 0):.3f}")
        self.logger.info(f"  Count MAPE: {metrics.get('mape', 0):.2f}%")
        self.logger.info(f"  Test Samples: {metrics.get('n_test', 0)}")
