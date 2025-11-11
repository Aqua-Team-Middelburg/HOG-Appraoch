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
        
        # Paths
        self.figures_dir = Path(config.get('paths.output_dir')) / config.get('paths.figures_dir')
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
        # Results storage
        self.evaluation_results = {}
        
        logger.info("Model evaluator initialized")
    
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
        
        # Probabilities (if available)
        try:
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(scaled_features)[:, 1]
            elif hasattr(model, 'decision_function'):
                probabilities = model.decision_function(scaled_features)
                # Normalize to [0,1] for consistency
                probabilities = (probabilities - probabilities.min()) / (probabilities.max() - probabilities.min())
            else:
                probabilities = None
        except Exception:
            probabilities = None
        
        # Basic classification metrics
        metrics = self._calculate_classification_metrics(test_labels, predictions, probabilities)
        
        # Detailed classification report
        report = classification_report(test_labels, predictions, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(test_labels, predictions)
        
        # Compile results
        results = {
            'model_name': model_name,
            'metrics': metrics,
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'predictions': predictions.tolist(),
            'test_labels': test_labels.tolist(),
            'probabilities': probabilities.tolist() if probabilities is not None else None,
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
    
    def evaluate_image_level_performance(self,
                                       model_results: Dict[str, Dict[str, Any]],
                                       image_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate performance at image level by aggregating window predictions.
        
        Args:
            model_results: Results from window-level evaluation
            image_metadata: Optional metadata mapping windows to images
            
        Returns:
            Image-level evaluation results
        """
        logger.info("Calculating image-level performance metrics")
        
        image_level_results = {}
        
        for model_name, results in model_results.items():
            if 'error' in results:
                continue
                
            predictions = np.array(results['predictions'])
            true_labels = np.array(results['test_labels'])
            probabilities = results.get('probabilities')
            
            if probabilities is not None:
                probabilities = np.array(probabilities)
            
            # If no image metadata provided, treat each prediction as separate image
            if image_metadata is None:
                logger.warning("No image metadata provided for image-level evaluation")
                image_level_results[model_name] = results  # Return window-level results
                continue
            
            # Group predictions by image
            image_predictions = defaultdict(list)
            image_labels = defaultdict(list)
            image_probabilities = defaultdict(list) if probabilities is not None else None
            
            # This would require proper window-to-image mapping
            # For now, simulate image-level aggregation
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
            
            image_level_results[model_name] = {
                'metrics': image_metrics,
                'predictions': image_level_preds,
                'labels': image_level_labels,
                'n_images': n_images,
                'aggregation_method': 'OR'
            }
            
            logger.info(f"{model_name} image-level F1: {image_metrics['f1_score']:.4f}")
        
        return image_level_results
    
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