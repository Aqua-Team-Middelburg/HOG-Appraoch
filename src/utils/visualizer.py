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
    

    
