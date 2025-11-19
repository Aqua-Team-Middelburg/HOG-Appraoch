"""
Pipeline Visualization Module
============================

This module provides visualization capabilities for the nurdle detection pipeline,
including predicted bounding boxes on test images in PNG format.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Dict, Any
import logging


class PipelineVisualizer:
    """
    Visualization class for nurdle detection pipeline results.
    
    Creates PNG images showing:
    - Original test images
    - Ground truth bounding boxes (green)
    - Predicted bounding boxes (red)
    - Confidence scores
    """
    
    def __init__(self, logger=None):
        """Initialize visualizer."""
        self.logger = logger or logging.getLogger(__name__)
        
        # Color scheme
        self.colors = {
            'ground_truth': 'lime',
            'prediction': 'red', 
            'high_confidence': 'orange',
            'low_confidence': 'yellow'
        }
    
    def create_prediction_visualization(self,
                                     image_path: str,
                                     ground_truth_coords: List[Tuple[float, float]],
                                     predicted_coords: List[Tuple[float, float]],
                                     confidences: List[float] = None,
                                     output_path: str = None) -> str:
        """
        Create a visualization showing predictions vs ground truth.
        
        Args:
            image_path: Path to the original image
            ground_truth_coords: List of (x, y) ground truth coordinates
            predicted_coords: List of (x, y) predicted coordinates
            confidences: Optional confidence scores for predictions
            output_path: Where to save the visualization
            
        Returns:
            Path to saved visualization
        """
        try:
            # Load image
            image = Image.open(image_path)
            
            # Create figure
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            ax.imshow(image)
            
            # Add ground truth annotations (green circles)
            for i, (x, y) in enumerate(ground_truth_coords):
                circle = plt.Circle((x, y), radius=15, color=self.colors['ground_truth'], 
                                  fill=False, linewidth=2, label='Ground Truth' if i == 0 else "")
                ax.add_patch(circle)
                
                # Add ground truth number
                ax.text(x + 20, y, f'GT{i+1}', color=self.colors['ground_truth'], 
                       fontsize=10, weight='bold')
            
            # Add predictions (red circles)
            confidences = confidences or [1.0] * len(predicted_coords)
            for i, ((x, y), conf) in enumerate(zip(predicted_coords, confidences)):
                # Color based on confidence
                if conf > 0.7:
                    color = self.colors['prediction']
                elif conf > 0.5:
                    color = self.colors['high_confidence'] 
                else:
                    color = self.colors['low_confidence']
                
                circle = plt.Circle((x, y), radius=12, color=color, 
                                  fill=False, linewidth=2, linestyle='--',
                                  label='Prediction' if i == 0 else "")
                ax.add_patch(circle)
                
                # Add prediction info
                ax.text(x + 20, y - 15, f'P{i+1}', color=color, fontsize=10, weight='bold')
                ax.text(x + 20, y + 5, f'{conf:.2f}', color=color, fontsize=8)
            
            # Set title and labels
            image_name = Path(image_path).stem
            ax.set_title(f'Nurdle Detection Results: {image_name}\n'
                        f'GT: {len(ground_truth_coords)} nurdles, '
                        f'Predicted: {len(predicted_coords)} nurdles', 
                        fontsize=14, pad=20)
            
            # Add legend
            ax.legend(loc='upper right')
            
            # Remove axes
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Add stats text box
            stats_text = (f'Ground Truth Count: {len(ground_truth_coords)}\n'
                         f'Predicted Count: {len(predicted_coords)}\n'
                         f'Count Accuracy: {"✓" if len(predicted_coords) == len(ground_truth_coords) else "✗"}')
            
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            # Save figure
            if output_path is None:
                output_path = f"prediction_{image_name}.png"
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Saved prediction visualization to {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Failed to create visualization for {image_path}: {e}")
            return None
    
    def create_batch_visualizations(self,
                                  test_annotations: Dict[str, Any],
                                  predict_func,
                                  output_dir: str) -> List[str]:
        """
        Create visualizations for all test images.
        
        Args:
            test_annotations: Dictionary of test image annotations
            predict_func: Function that takes image path and returns (count, coordinates)
            output_dir: Directory to save visualizations
            
        Returns:
            List of paths to created visualizations
        """
        self.logger.info("Creating batch visualizations for test images...")
        
        # Setup output directory
        viz_dir = Path(output_dir) / "visualizations" / "predictions"
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        created_files = []
        
        for image_path, annotation in test_annotations.items():
            try:
                self.logger.debug(f"Processing visualization for {image_path}")
                
                # Get ground truth
                gt_coords = []
                if 'objects' in annotation:
                    gt_coords = [(obj['center_x'], obj['center_y']) for obj in annotation['objects']]
                elif 'nurdles' in annotation:
                    gt_coords = [(nurdle['center_x'], nurdle['center_y']) for nurdle in annotation['nurdles']]
                
                # Get predictions
                pred_count, pred_coords = predict_func(image_path)
                
                # Extract confidences (if available)
                confidences = []
                if isinstance(pred_coords, list) and len(pred_coords) > 0:
                    if isinstance(pred_coords[0], tuple) and len(pred_coords[0]) > 2:
                        # Coordinates include confidence
                        confidences = [coord[2] for coord in pred_coords]
                        pred_coords = [(coord[0], coord[1]) for coord in pred_coords]
                
                # Create visualization
                output_path = viz_dir / f"{Path(image_path).stem}_prediction.png"
                
                viz_path = self.create_prediction_visualization(
                    image_path=image_path,
                    ground_truth_coords=gt_coords,
                    predicted_coords=pred_coords,
                    confidences=confidences,
                    output_path=str(output_path)
                )
                
                if viz_path:
                    created_files.append(viz_path)
                    
            except Exception as e:
                self.logger.error(f"Failed to create visualization for {image_path}: {e}")
                continue
        
        self.logger.info(f"Created {len(created_files)} prediction visualizations")
        return created_files
    
    def create_training_summary_plot(self,
                                   metrics: Dict[str, float],
                                   output_dir: str) -> str:
        """
        Create a summary plot of training metrics.
        
        Args:
            metrics: Dictionary of evaluation metrics
            output_dir: Directory to save plot
            
        Returns:
            Path to saved plot
        """
        try:
            # Setup figure
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
            
            # Metric names and values
            metric_names = ['Count Accuracy', 'Precision', 'Recall', 'F1-Score']
            metric_values = [
                metrics.get('count_accuracy', 0),
                metrics.get('precision', 0),
                metrics.get('recall', 0),
                metrics.get('f1_score', 0)
            ]
            
            axes = [ax1, ax2, ax3, ax4]
            colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold']
            
            # Create bar plots for each metric
            for ax, name, value, color in zip(axes, metric_names, metric_values, colors):
                bars = ax.bar([name], [value], color=color, alpha=0.7, edgecolor='black')
                ax.set_ylim(0, 1)
                ax.set_ylabel('Score')
                ax.set_title(f'{name}: {value:.3f}')
                
                # Add value text on bar
                ax.text(0, value + 0.02, f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
            
            # Overall title
            fig.suptitle('Nurdle Detection Pipeline - Performance Summary', fontsize=16, fontweight='bold')
            
            # Add additional metrics text
            additional_info = (
                f"Count MAE: {metrics.get('count_mae', 'N/A'):.3f}\n"
                f"Coordinate Error: {metrics.get('avg_coordinate_error', 'N/A'):.1f} pixels\n"
                f"Test Images: {metrics.get('n_test_images', 'N/A')}"
            )
            
            fig.text(0.02, 0.02, additional_info, fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
            
            # Save plot
            output_path = Path(output_dir) / "visualizations" / "training_summary.png"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Saved training summary plot to {output_path}")
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"Failed to create training summary plot: {e}")
            return None
    
    def create_comparison_visualization(self,
                                       image_path: str,
                                       ground_truth_coords: List[Tuple[float, float]],
                                       main_prediction: Tuple[int, List[Tuple[float, float]]],
                                       stacked_prediction: Tuple[int, List[Tuple[float, float]]],
                                       output_path: str) -> None:
        """
        Create side-by-side comparison visualization.
        
        Shows three panels: Ground Truth | Main Pipeline | Stacked Model
        
        Args:
            image_path: Path to image file
            ground_truth_coords: List of ground truth coordinates
            main_prediction: (count, coordinates) from main pipeline
            stacked_prediction: (count, coordinates) from stacked model
            output_path: Path to save visualization
        """
        import cv2
        
        # Load image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create figure with 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Ground Truth
        axes[0].imshow(image)
        axes[0].set_title(f'Ground Truth (n={len(ground_truth_coords)})', fontsize=14, fontweight='bold')
        for x, y in ground_truth_coords:
            circle = plt.Circle((x, y), radius=10, color='green', fill=False, linewidth=2)
            axes[0].add_patch(circle)
        axes[0].axis('off')
        
        # Main Pipeline
        main_count, main_coords = main_prediction
        axes[1].imshow(image)
        axes[1].set_title(f'Main Pipeline (n={main_count})', fontsize=14, fontweight='bold')
        for x, y in main_coords:
            circle = plt.Circle((x, y), radius=10, color='blue', fill=False, linewidth=2)
            axes[1].add_patch(circle)
        axes[1].axis('off')
        
        # Stacked Model
        stacked_count, stacked_coords = stacked_prediction
        axes[2].imshow(image)
        axes[2].set_title(f'Stacked Model (n={stacked_count})', fontsize=14, fontweight='bold')
        for x, y in stacked_coords:
            circle = plt.Circle((x, y), radius=10, color='red', fill=False, linewidth=2)
            axes[2].add_patch(circle)
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Saved comparison visualization to {output_path}")
    
    def create_optuna_summary(self,
                            best_params: Dict[str, Any], 
                            best_score: float,
                            output_dir: str) -> str:
        """
        Create a visual summary of Optuna optimization results.
        
        Args:
            best_params: Dictionary of best parameters found
            best_score: Best score achieved
            output_dir: Directory to save plot
            
        Returns:
            Path to saved plot
        """
        try:
            # Create parameter visualization
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Plot 1: Parameter values
            param_names = list(best_params.keys())
            param_values = []
            param_labels = []
            
            for name, value in best_params.items():
                if isinstance(value, (int, float)):
                    param_values.append(float(value))
                    param_labels.append(f"{name}\n{value:.3f}")
                else:
                    # For categorical parameters, use index
                    param_values.append(1.0)
                    param_labels.append(f"{name}\n{value}")
            
            # Normalize values for better visualization
            if param_values:
                max_val = max(param_values) if max(param_values) > 0 else 1
                normalized_values = [v / max_val for v in param_values]
            else:
                normalized_values = []
            
            bars = ax1.barh(param_labels, normalized_values, color='skyblue', alpha=0.7)
            ax1.set_xlabel('Normalized Parameter Value')
            ax1.set_title('Best Parameters Found by Optuna')
            ax1.grid(axis='x', alpha=0.3)
            
            # Plot 2: Performance score
            ax2.bar(['Best Score'], [best_score], color='gold', alpha=0.7, edgecolor='black', width=0.5)
            ax2.set_ylim(0, 1)
            ax2.set_ylabel('Score')
            ax2.set_title(f'Optimization Result\nScore: {best_score:.4f}')
            ax2.text(0, best_score + 0.02, f'{best_score:.4f}', ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            
            # Save plot
            output_path = Path(output_dir) / "visualizations" / "optuna_summary.png"
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Saved Optuna summary plot to {output_path}")
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"Failed to create Optuna summary plot: {e}")
            return None