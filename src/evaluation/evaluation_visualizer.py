"""
Evaluation visualization module for creating charts and plots.

This module handles all visualization generation for model evaluation results,
separating presentation logic from evaluation logic following the Single 
Responsibility Principle.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

from sklearn.metrics import roc_curve, precision_recall_curve

logger = logging.getLogger(__name__)


class EvaluationVisualizer:
    """
    Creates comprehensive evaluation visualizations for model results.
    
    Handles:
    - Model comparison bar charts
    - Confusion matrix heatmaps
    - ROC curves
    - Precision-Recall curves  
    - Threshold optimization plots
    """
    
    def __init__(self, figures_dir: Path, visualization_config: Dict[str, Any]):
        """
        Initialize evaluation visualizer.
        
        Args:
            figures_dir: Directory for saving visualization files
            visualization_config: Configuration for visualizations
        """
        self.figures_dir = Path(figures_dir)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.visualization_config = visualization_config
    
    def create_all_visualizations(self, model_results: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
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
            viz_path = self.create_model_comparison_chart(model_results)
            if viz_path:
                visualization_paths['model_comparison'] = str(viz_path)
        
        # 2. Confusion matrices
        confusion_paths = self.create_confusion_matrices(model_results)
        visualization_paths.update(confusion_paths)
        
        # 3. ROC curves
        if self._has_probabilities(model_results):
            roc_path = self.create_roc_curves(model_results)
            if roc_path:
                visualization_paths['roc_curves'] = str(roc_path)
        
        # 4. Precision-Recall curves
        if self._has_probabilities(model_results):
            pr_path = self.create_pr_curves(model_results)
            if pr_path:
                visualization_paths['pr_curves'] = str(pr_path)
        
        # 5. Threshold optimization plots
        threshold_path = self.create_threshold_optimization_plot(model_results)
        if threshold_path:
            visualization_paths['threshold_optimization'] = str(threshold_path)
        
        logger.info(f"Created {len(visualization_paths)} visualizations")
        return visualization_paths
    
    def _has_probabilities(self, model_results: Dict[str, Dict[str, Any]]) -> bool:
        """
        Check if any model has probability predictions.
        
        Args:
            model_results: Model evaluation results
            
        Returns:
            True if any model has probabilities
        """
        return any(
            results.get('probabilities') is not None 
            for results in model_results.values()
            if 'error' not in results
        )
    
    def create_model_comparison_chart(self, model_results: Dict[str, Dict[str, Any]]) -> Optional[Path]:
        """
        Create model comparison bar chart.
        
        Args:
            model_results: Results from multiple model evaluations
            
        Returns:
            Path to saved chart or None if failed
        """
        try:
            # Extract comparison data
            comparison_data = []
            for model_name, results in model_results.items():
                if 'error' not in results and 'metrics' in results:
                    metrics = results['metrics']
                    comparison_data.append({
                        'model': model_name,
                        'accuracy': metrics.get('accuracy', 0),
                        'precision': metrics.get('precision', 0),
                        'recall': metrics.get('recall', 0),
                        'f1_score': metrics.get('f1_score', 0)
                    })
            
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
    
    def create_confusion_matrices(self, model_results: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
        """
        Create confusion matrix plots for each model.
        
        Args:
            model_results: Results from model evaluations
            
        Returns:
            Dictionary mapping model names to confusion matrix file paths
        """
        confusion_paths = {}
        
        for model_name, results in model_results.items():
            if 'error' in results or 'confusion_matrix' not in results:
                continue
                
            try:
                cm = np.array(results['confusion_matrix'])
                
                # Create confusion matrix plot
                fig, ax = plt.subplots(figsize=(8, 6))
                
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                           xticklabels=['Negative', 'Positive'],
                           yticklabels=['Negative', 'Positive'],
                           ax=ax)
                
                ax.set_title(f'Confusion Matrix - {model_name}')
                ax.set_ylabel('Actual Label')
                ax.set_xlabel('Predicted Label')
                
                # Save plot
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                plot_path = self.figures_dir / f'confusion_matrix_{model_name}_{timestamp}.png'
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                confusion_paths[f'confusion_matrix_{model_name}'] = str(plot_path)
                
            except Exception as e:
                logger.error(f"Failed to create confusion matrix for {model_name}: {e}")
        
        return confusion_paths
    
    def create_roc_curves(self, model_results: Dict[str, Dict[str, Any]]) -> Optional[Path]:
        """
        Create ROC curves plot for models with probabilities.
        
        Args:
            model_results: Results from model evaluations
            
        Returns:
            Path to saved ROC curves plot or None if failed
        """
        try:
            fig, ax = plt.subplots(figsize=(10, 8))
            
            for model_name, results in model_results.items():
                if ('error' not in results and 
                    'probabilities' in results and 
                    'true_labels' in results):
                    
                    y_true = np.array(results['true_labels'])
                    y_prob = np.array(results['probabilities'])
                    
                    fpr, tpr, _ = roc_curve(y_true, y_prob)
                    auc_score = results.get('metrics', {}).get('roc_auc', 0)
                    
                    ax.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.3f})')
            
            # Plot random classifier line
            ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
            
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('ROC Curves - Model Comparison')
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
    
    def create_pr_curves(self, model_results: Dict[str, Dict[str, Any]]) -> Optional[Path]:
        """
        Create Precision-Recall curves plot for models with probabilities.
        
        Args:
            model_results: Results from model evaluations
            
        Returns:
            Path to saved PR curves plot or None if failed
        """
        try:
            fig, ax = plt.subplots(figsize=(10, 8))
            
            for model_name, results in model_results.items():
                if ('error' not in results and 
                    'probabilities' in results and 
                    'true_labels' in results):
                    
                    y_true = np.array(results['true_labels'])
                    y_prob = np.array(results['probabilities'])
                    
                    precision, recall, _ = precision_recall_curve(y_true, y_prob)
                    ap_score = results.get('metrics', {}).get('average_precision', 0)
                    
                    ax.plot(recall, precision, label=f'{model_name} (AP = {ap_score:.3f})')
            
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.set_title('Precision-Recall Curves - Model Comparison')
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
    
    def create_threshold_optimization_plot(self, model_results: Dict[str, Dict[str, Any]]) -> Optional[Path]:
        """
        Create threshold optimization plots showing F1 vs threshold.
        
        Args:
            model_results: Results from model evaluations
            
        Returns:
            Path to saved threshold plot or None if failed
        """
        try:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            for model_name, results in model_results.items():
                if ('error' not in results and 
                    'threshold_optimization' in results):
                    
                    opt_results = results['threshold_optimization']
                    thresholds = opt_results.get('thresholds', [])
                    scores = opt_results.get('scores', [])
                    optimal_threshold = opt_results.get('optimal_threshold', 0.5)
                    optimal_score = opt_results.get('optimal_score', 0)
                    
                    if thresholds and scores:
                        ax.plot(thresholds, scores, label=f'{model_name}')
                        
                        # Mark optimal point
                        ax.scatter([optimal_threshold], [optimal_score], 
                                 marker='o', s=100, 
                                 label=f'{model_name} Optimal (t={optimal_threshold:.3f}, F1={optimal_score:.3f})')
            
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