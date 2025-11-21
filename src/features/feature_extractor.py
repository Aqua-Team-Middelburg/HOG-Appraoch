"""
Feature extraction for nurdle detection pipeline.

This module provides combined HOG and LBP feature extraction from image windows,
optimized for memory efficiency and computational performance.
"""

import numpy as np
import cv2
import pickle
from pathlib import Path
from typing import Dict, Any, Iterator, List, Tuple
from dataclasses import dataclass
import logging
from skimage.feature import hog, local_binary_pattern

from ..data import ImageAnnotation, NurdleAnnotation


@dataclass 
class TrainingWindow:
    """Single training window with features and labels."""
    features: np.ndarray
    is_nurdle: bool           # For SVM classifier (binary label)
    offset_x: float           # For SVR regressor (nurdle_center_x - window_center_x)
    offset_y: float           # For SVR regressor (nurdle_center_y - window_center_y)
    image_id: str
    window_x: int             # Window top-left x position
    window_y: int             # Window top-left y position
    window_center_x: float    # For inference coordinate calculation
    window_center_y: float    # For inference coordinate calculation


class FeatureExtractor:
    """
    Extracts combined HOG and LBP features from image windows.
    
    This class provides efficient feature extraction for training and inference,
    with integrated sliding window processing for memory efficiency.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize feature extractor with configuration."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Configuration parameters
        self.window_size = tuple(config.get('size', [20, 20]))
        self.window_stride = config.get('stride', 8)
        self.hog_cell_size = tuple(config.get('hog_cell_size', [4, 4]))
        self.negative_positive_ratio = config.get('negative_positive_ratio', 3.0)
        self.min_distance_from_nurdles = config.get('min_distance_from_nurdles', 25)
    
    def extract_hog_lbp_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract combined HOG and LBP features from image window.
        
        This replaces separate feature extraction with a single combined approach.
        
        Args:
            image: Image window to extract features from
            
        Returns:
            Combined feature vector
        """
        # Ensure window is correct size
        if image.shape[:2] != self.window_size:
            image = cv2.resize(image, self.window_size)
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # HOG Features
        hog_features = hog(
            gray,
            orientations=9,
            pixels_per_cell=self.hog_cell_size,
            cells_per_block=(2, 2),
            block_norm='L2-Hys',
            feature_vector=True
        )
        
        # LBP Features  
        lbp = local_binary_pattern(gray, 8, 1, method='uniform')
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 9))
        lbp_features = lbp_hist.astype(float)
        
        # Combine features
        combined_features = np.concatenate([hog_features, lbp_features])
        
        return combined_features
    
    def generate_training_windows_batch(self, 
                                      annotations: List[ImageAnnotation],
                                      load_image_func,
                                      transform_coords_func=None) -> Iterator[List[TrainingWindow]]:
        """
        Generate training windows using sliding window approach.
        
        This method slides a window across the entire image and checks if each window
        contains any nurdle center coordinates. Windows containing nurdles are labeled
        as positive, and a ratio-based number of negative windows (far from nurdles)
        are randomly sampled.
        
        Args:
            annotations: List of image annotations to process (with original coordinates)
            load_image_func: Function to load and normalize images
            transform_coords_func: Optional function to transform annotation coordinates
                                  to match normalized image space
            
        Yields:
            Batches of training windows
        """
        batch_windows = []
        
        for annotation in annotations:
            try:
                # Load original image to get scale factors
                original_image = cv2.imread(annotation.image_path)
                if original_image is None:
                    self.logger.error(f"Could not load image: {annotation.image_path}")
                    continue
                
                # Load and normalize image
                image = load_image_func(annotation.image_path)
                h, w = image.shape[:2]
                orig_h, orig_w = original_image.shape[:2]
                
                # Calculate coordinate transformation scale
                scale_x = w / orig_w
                scale_y = h / orig_h
                
                # Transform annotation coordinates to normalized space
                if transform_coords_func:
                    transformed_annotation = transform_coords_func(annotation, original_image)
                else:
                    # Manual transformation
                    transformed_nurdles = [
                        NurdleAnnotation(x=n.x * scale_x, y=n.y * scale_y) 
                        for n in annotation.nurdles
                    ]
                    transformed_annotation = ImageAnnotation(
                        image_path=annotation.image_path,
                        nurdles=transformed_nurdles,
                        image_id=annotation.image_id
                    )
                
                # Convert transformed nurdle coordinates to numpy array
                if transformed_annotation.nurdles:
                    nurdle_coords = np.array([[n.x, n.y] for n in transformed_annotation.nurdles])
                else:
                    nurdle_coords = np.empty((0, 2))
                
                self.logger.debug(f"Image {annotation.image_id}: Original {orig_w}x{orig_h} -> "
                                f"Normalized {w}x{h}, Scale: {scale_x:.4f}x{scale_y:.4f}")
                
                # Slide window across entire image to find all positive windows
                positive_windows = []
                potential_negative_windows = []
                
                for y in range(0, h - self.window_size[1] + 1, self.window_stride):
                    for x in range(0, w - self.window_size[0] + 1, self.window_stride):
                        # Calculate window center (use float for precise offset calculation)
                        window_center_x = x + self.window_size[0] / 2.0
                        window_center_y = y + self.window_size[1] / 2.0
                        
                        # Check if any nurdle center is inside window bounds
                        nurdles_in_window = []
                        if len(nurdle_coords) > 0:
                            for nurdle_x, nurdle_y in nurdle_coords:
                                if (x <= nurdle_x < x + self.window_size[0] and 
                                    y <= nurdle_y < y + self.window_size[1]):
                                    nurdles_in_window.append((nurdle_x, nurdle_y))
                        
                        # Extract window
                        window = image[y:y+self.window_size[1], x:x+self.window_size[0]]
                        
                        if window.shape[:2] == self.window_size:
                            features = self.extract_hog_lbp_features(window)
                            
                            if len(nurdles_in_window) > 0:
                                # POSITIVE WINDOW - Find nearest nurdle to window center
                                distances = [
                                    np.sqrt((nx - window_center_x)**2 + (ny - window_center_y)**2)
                                    for nx, ny in nurdles_in_window
                                ]
                                nearest_idx = np.argmin(distances)
                                nearest_nurdle = nurdles_in_window[nearest_idx]
                                
                                # Calculate offset from window center to nearest nurdle
                                offset_x = nearest_nurdle[0] - window_center_x
                                offset_y = nearest_nurdle[1] - window_center_y
                                
                                positive_windows.append(TrainingWindow(
                                    features=features,
                                    is_nurdle=True,
                                    offset_x=offset_x,
                                    offset_y=offset_y,
                                    image_id=annotation.image_id,
                                    window_x=x,
                                    window_y=y,
                                    window_center_x=window_center_x,
                                    window_center_y=window_center_y
                                ))
                            else:
                                # NEGATIVE WINDOW (only if far enough from all nurdles)
                                if len(nurdle_coords) > 0:
                                    distances = np.sqrt(
                                        (nurdle_coords[:, 0] - window_center_x)**2 + 
                                        (nurdle_coords[:, 1] - window_center_y)**2
                                    )
                                    min_distance = np.min(distances)
                                    
                                    if min_distance >= self.min_distance_from_nurdles:
                                        potential_negative_windows.append(TrainingWindow(
                                            features=features,
                                            is_nurdle=False,
                                            offset_x=0.0,  # Unused for negative windows
                                            offset_y=0.0,
                                            image_id=annotation.image_id,
                                            window_x=x,
                                            window_y=y,
                                            window_center_x=window_center_x,
                                            window_center_y=window_center_y
                                        ))
                                else:
                                    # No nurdles in image - all windows are negative
                                    potential_negative_windows.append(TrainingWindow(
                                        features=features,
                                        is_nurdle=False,
                                        offset_x=0.0,
                                        offset_y=0.0,
                                        image_id=annotation.image_id,
                                        window_x=x,
                                        window_y=y,
                                        window_center_x=window_center_x,
                                        window_center_y=window_center_y
                                    ))
                
                # Sample negative windows based on ratio
                n_negative = int(len(positive_windows) * self.negative_positive_ratio)
                
                if len(potential_negative_windows) > n_negative:
                    # Randomly sample the required number of negative windows
                    indices = np.random.choice(len(potential_negative_windows), 
                                              size=n_negative, 
                                              replace=False)
                    negative_windows = [potential_negative_windows[i] for i in indices]
                else:
                    # Use all available negative windows
                    negative_windows = potential_negative_windows
                
                self.logger.info(f"Image {annotation.image_id}: {len(positive_windows)} positive, "
                               f"{len(negative_windows)} negative windows (from {len(potential_negative_windows)} candidates)")
                
                # Add all windows from this image to batch
                batch_windows.extend(positive_windows)
                batch_windows.extend(negative_windows)
                
                # Yield batch if it's getting large (memory management)
                if len(batch_windows) >= 1000:  # 1000 windows per batch
                    yield batch_windows
                    batch_windows = []
                    
            except Exception as e:
                self.logger.error(f"Error processing image {annotation.image_path}: {e}")
                continue
        
        # Yield remaining windows
        if batch_windows:
            yield batch_windows
    
    def generate_windows_for_image(self, image: np.ndarray, annotation: ImageAnnotation) -> List[TrainingWindow]:
        """
        Generate all training windows for a single image.
        
        Args:
            image: Normalized image
            annotation: Image annotation with nurdle coordinates (in normalized space)
            
        Returns:
            List of TrainingWindow objects
        """
        h, w = image.shape[:2]
        
        # Get nurdle coordinates
        if annotation.nurdles:
            nurdle_coords = np.array([[n.x, n.y] for n in annotation.nurdles])
        else:
            nurdle_coords = np.empty((0, 2))
        
        positive_windows = []
        potential_negative_windows = []
        
        # Slide window across image
        for y in range(0, h - self.window_size[1] + 1, self.window_stride):
            for x in range(0, w - self.window_size[0] + 1, self.window_stride):
                window_center_x = x + self.window_size[0] / 2.0
                window_center_y = y + self.window_size[1] / 2.0
                
                # Check if any nurdle center is inside window bounds
                nurdles_in_window = []
                if len(nurdle_coords) > 0:
                    for nurdle_x, nurdle_y in nurdle_coords:
                        if (x <= nurdle_x < x + self.window_size[0] and 
                            y <= nurdle_y < y + self.window_size[1]):
                            nurdles_in_window.append((nurdle_x, nurdle_y))
                
                window = image[y:y+self.window_size[1], x:x+self.window_size[0]]
                
                if window.shape[:2] == self.window_size:
                    features = self.extract_hog_lbp_features(window)
                    
                    if len(nurdles_in_window) > 0:
                        # Positive window
                        distances = [
                            np.sqrt((nx - window_center_x)**2 + (ny - window_center_y)**2)
                            for nx, ny in nurdles_in_window
                        ]
                        nearest_idx = np.argmin(distances)
                        nearest_nurdle = nurdles_in_window[nearest_idx]
                        
                        offset_x = nearest_nurdle[0] - window_center_x
                        offset_y = nearest_nurdle[1] - window_center_y
                        
                        positive_windows.append(TrainingWindow(
                            features=features,
                            is_nurdle=True,
                            offset_x=offset_x,
                            offset_y=offset_y,
                            image_id=annotation.image_id,
                            window_x=x,
                            window_y=y,
                            window_center_x=window_center_x,
                            window_center_y=window_center_y
                        ))
                    else:
                        # Potential negative window
                        if len(nurdle_coords) > 0:
                            distances = np.sqrt(
                                (nurdle_coords[:, 0] - window_center_x)**2 + 
                                (nurdle_coords[:, 1] - window_center_y)**2
                            )
                            min_distance = np.min(distances)
                            
                            if min_distance >= self.min_distance_from_nurdles:
                                potential_negative_windows.append(TrainingWindow(
                                    features=features,
                                    is_nurdle=False,
                                    offset_x=0.0,
                                    offset_y=0.0,
                                    image_id=annotation.image_id,
                                    window_x=x,
                                    window_y=y,
                                    window_center_x=window_center_x,
                                    window_center_y=window_center_y
                                ))
        
        # Sample negative windows
        n_negative = int(len(positive_windows) * self.negative_positive_ratio)
        if len(potential_negative_windows) > n_negative:
            import random
            negative_windows = random.sample(potential_negative_windows, n_negative)
        else:
            negative_windows = potential_negative_windows
        
        return positive_windows + negative_windows
    
    def save_windows(self, windows: List[TrainingWindow], checkpoint_dir: Path) -> None:
        """
        Save candidate windows to checkpoint directory.
        
        Args:
            windows: List of training windows to save
            checkpoint_dir: Directory to save checkpoint
        """
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_file = checkpoint_dir / 'windows.pkl'
        
        self.logger.info(f"Saving {len(windows)} windows to {checkpoint_file}")
        
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(windows, f)
        
        # Save summary metadata
        positive_count = sum(1 for w in windows if w.is_nurdle)
        negative_count = len(windows) - positive_count
        
        metadata = {
            'total_windows': len(windows),
            'positive_windows': positive_count,
            'negative_windows': negative_count,
            'window_size': self.window_size,
            'stride': self.window_stride
        }
        
        metadata_file = checkpoint_dir / 'windows_metadata.json'
        import json
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"Saved windows: {positive_count} positive, {negative_count} negative")
    
    def load_windows(self, checkpoint_dir: Path) -> List[TrainingWindow]:
        """
        Load candidate windows from checkpoint directory.
        
        Args:
            checkpoint_dir: Directory containing checkpoint
            
        Returns:
            List of training windows
            
        Raises:
            FileNotFoundError: If checkpoint not found
        """
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_file = checkpoint_dir / 'windows.pkl'
        
        if not checkpoint_file.exists():
            raise FileNotFoundError(f"Windows checkpoint not found: {checkpoint_file}")
        
        self.logger.info(f"Loading windows from {checkpoint_file}")
        
        with open(checkpoint_file, 'rb') as f:
            windows = pickle.load(f)
        
        self.logger.info(f"Loaded {len(windows)} windows")
        return windows
    
    def visualize_features(self, windows: List[TrainingWindow], data_loader, output_dir: Path) -> None:
        """
        Generate HOG and LBP visualizations for sample windows from TEST set.
        
        IMPORTANT: Generates windows from TEST SET images only - NO data leakage!
        Test images are never used in training, only for demonstration purposes.
        
        Creates 2 combined images:
        - positive_features.png: 3 positive windows with their HOG and LBP features
        - negative_features.png: 3 negative windows with their HOG and LBP features
        
        Args:
            windows: List of training windows (not used, we generate from test set)
            data_loader: DataLoader to access test images
            output_dir: Directory to save visualizations
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        import random
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("Generating feature visualizations from TEST set (no data leakage)...")
        
        # Generate windows from TEST set images only (for visualization)
        test_windows = []
        for annotation in data_loader.test_annotations:
            try:
                image = data_loader.load_image(annotation.image_path)
                windows_for_image = self.generate_windows_for_image(image, annotation)
                test_windows.extend(windows_for_image)
            except Exception as e:
                self.logger.warning(f"Could not generate windows for test image {annotation.image_id}: {e}")
        
        if len(test_windows) == 0:
            self.logger.warning("No test windows generated for feature visualization")
            return
        
        # Separate positive and negative windows from TEST set
        positive_windows = [w for w in test_windows if w.is_nurdle]
        negative_windows = [w for w in test_windows if not w.is_nurdle]
        
        # Sample 3 of each
        sample_positive = random.sample(positive_windows, min(3, len(positive_windows)))
        sample_negative = random.sample(negative_windows, min(3, len(negative_windows)))
        
        # Process each type
        for window_type, samples in [('positive', sample_positive), ('negative', sample_negative)]:
            try:
                # Create figure with 3 rows (samples) x 3 columns (original, HOG, LBP)
                fig, axes = plt.subplots(3, 3, figsize=(12, 12))
                fig.suptitle(f'{window_type.capitalize()} Window Features (Original | HOG | LBP)', 
                           fontsize=16, fontweight='bold')
                
                for row_idx, window in enumerate(samples):
                    try:
                        # Find corresponding annotation from TEST set
                        annotation = next((a for a in data_loader.test_annotations if a.image_id == window.image_id), None)
                        if annotation is None:
                            continue
                        
                        # Load image
                        image = data_loader.load_image(annotation.image_path)
                        
                        # Extract window
                        window_img = image[
                            window.window_y:window.window_y + self.window_size[1],
                            window.window_x:window.window_x + self.window_size[0]
                        ]
                        
                        if window_img.shape[:2] != self.window_size:
                            continue
                        
                        # Convert to grayscale
                        gray = cv2.cvtColor(window_img, cv2.COLOR_BGR2GRAY) if len(window_img.shape) == 3 else window_img
                        
                        # Generate HOG visualization
                        hog_features, hog_image = hog(
                            gray,
                            orientations=9,
                            pixels_per_cell=self.hog_cell_size,
                            cells_per_block=(2, 2),
                            block_norm='L2-Hys',
                            feature_vector=True,
                            visualize=True
                        )
                        
                        # Generate LBP pattern
                        lbp = local_binary_pattern(gray, 8, 1, method='uniform')
                        
                        # Plot original window
                        axes[row_idx, 0].imshow(cv2.cvtColor(window_img, cv2.COLOR_BGR2RGB) if len(window_img.shape) == 3 else window_img, cmap='gray')
                        axes[row_idx, 0].set_title(f'Sample {row_idx+1}', fontsize=11, fontweight='bold')
                        axes[row_idx, 0].axis('off')
                        
                        # Plot HOG features
                        axes[row_idx, 1].imshow(hog_image, cmap='hot')
                        axes[row_idx, 1].set_title(f'HOG Features', fontsize=11)
                        axes[row_idx, 1].axis('off')
                        
                        # Plot LBP pattern
                        axes[row_idx, 2].imshow(lbp, cmap='gray')
                        axes[row_idx, 2].set_title(f'LBP Pattern', fontsize=11)
                        axes[row_idx, 2].axis('off')
                        
                    except Exception as e:
                        self.logger.warning(f"Could not process {window_type} window {row_idx}: {e}")
                        # Clear the row if processing failed
                        for col_idx in range(3):
                            axes[row_idx, col_idx].axis('off')
                
                plt.tight_layout()
                plt.savefig(output_dir / f'{window_type}_features.png', dpi=150, bbox_inches='tight')
                plt.close()
                
                self.logger.info(f"Generated {window_type} features visualization")
                
            except Exception as e:
                self.logger.warning(f"Could not generate {window_type} features visualization: {e}")
        
        self.logger.info(f"Generated 2 combined feature visualizations")
    
    def visualize_windows_samples(self, windows: List[TrainingWindow], data_loader, output_dir: Path, num_samples: int = 3) -> None:
        """
        Generate visualization of positive candidate windows on TEST set images.
        
        IMPORTANT: Uses TEST SET samples only for visualization - NO data leakage!
        Test images are never used in training, only for demonstration purposes.
        
        Creates one image per test sample showing only positive windows (no labels).
        
        Args:
            windows: List of training windows (not used, we generate new windows from test set)
            data_loader: DataLoader to access test images
            output_dir: Directory to save visualizations
            num_samples: Maximum number of samples (limited by test set size, max 3)
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Use TEST set samples only (no data leakage)
        if len(data_loader.test_annotations) == 0:
            self.logger.warning("No test samples available for window visualization")
            return
        
        # Limit to min(num_samples, test set size, 3)
        n_samples = min(num_samples, len(data_loader.test_annotations), 3)
        sample_annotations = data_loader.test_annotations[:n_samples]
        
        self.logger.info(f"Generating window visualizations using {n_samples} TEST set samples (no data leakage)")
        
        # Create one file per image
        for annotation in sample_annotations:
            try:
                # Load normalized test image
                image = data_loader.load_image(annotation.image_path)
                
                # Generate windows for this TEST image (for visualization only)
                test_windows = self.generate_windows_for_image(image, annotation)
                
                # Filter only positive windows
                positive_windows = [w for w in test_windows if w.is_nurdle]
                
                # Create individual figure for this image
                fig, ax = plt.subplots(1, 1, figsize=(14, 10))
                
                # Display image
                ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if len(image.shape) == 3 else image, cmap='gray')
                
                # Draw only positive windows (no labels, just rectangles)
                for window in positive_windows:
                    rect = patches.Rectangle(
                        (window.window_x, window.window_y),
                        self.window_size[0], self.window_size[1],
                        linewidth=2, edgecolor='lime', facecolor='none', alpha=0.7
                    )
                    ax.add_patch(rect)
                
                ax.set_title(
                    f'{annotation.image_id} - {len(positive_windows)} positive windows (TEST sample)',
                    fontsize=14, fontweight='bold'
                )
                ax.axis('off')
                
                plt.tight_layout()
                output_file = output_path / f'{annotation.image_id}_windows.png'
                plt.savefig(output_file, dpi=150, bbox_inches='tight')
                plt.close()
                
                self.logger.info(f"Saved window visualization: {output_file}")
                
            except Exception as e:
                self.logger.error(f"Error visualizing windows for {annotation.image_id}: {e}")
        
        self.logger.info(f"Generated {n_samples} window visualization files")