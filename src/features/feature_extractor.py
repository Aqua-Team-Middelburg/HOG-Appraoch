"""
Feature extraction for nurdle detection pipeline.

This module provides combined HOG and LBP feature extraction from full images for SVM count classification.
"""

# Required imports
import logging
from typing import Dict, Any, List, Iterator, Tuple, Optional
import numpy as np
import cv2
from skimage.feature import hog, local_binary_pattern
from pathlib import Path

# If ImageAnnotation is defined elsewhere, import it. Otherwise, define a minimal stub for type hints.
class FeatureExtractor:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.hog_cell_size = tuple(config.get('hog_cell_size', [4, 4]))
        self.image_size = tuple(config.get('image_size', [1080, 1080]))

    def _apply_mask(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Apply a binary mask to an image, resizing the mask if needed."""
        if mask.shape[:2] != image.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 0).astype(np.uint8) * 255
        return cv2.bitwise_and(image, image, mask=mask)

    def extract_image_features(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Extract combined RGB HOG (per channel) and grayscale LBP features.
        Image is resized to config.image_size before feature extraction.

        If a mask is provided, HOG/LBP are computed on the masked nurdle regions only.
        """
        if mask is not None:
            image = self._apply_mask(image, mask)

        if image.shape[:2] != self.image_size:
            image = cv2.resize(image, self.image_size)

        # Ensure RGB ordering for per-channel HOG
        if len(image.shape) == 3:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        parts = []

        # Always extract HOG (per channel)
        hog_features = []
        for c in range(3):
            channel = rgb[:, :, c]
            hog_feat = hog(
                channel,
                orientations=9,
                pixels_per_cell=self.hog_cell_size,
                cells_per_block=(2, 2),
                block_norm='L2-Hys',
                feature_vector=True
            )
            hog_features.append(hog_feat)
        hog_features = np.concatenate(hog_features)
        parts.append(hog_features)

        # Always extract LBP (on grayscale for texture)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        lbp = local_binary_pattern(gray, 8, 1, method='uniform')
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 9))
        lbp_features = lbp_hist.astype(float)
        parts.append(lbp_features)

        return np.concatenate(parts)

    def extract_mask_statistics(self, mask: np.ndarray) -> np.ndarray:
        """
        Extract statistical features from a binary segmentation mask.
        
        Returns a feature vector with:
        - Mask coverage (% of image)
        - Number of connected components (nurdles)
        - Mean and std of component areas
        - Centroid coordinates (normalized to [0, 1])
        - Aspect ratio statistics
        """
        mask_binary = (mask > 0).astype(np.uint8)
        
        # Basic coverage
        total_pixels = mask.shape[0] * mask.shape[1]
        coverage = float(np.sum(mask_binary)) / total_pixels if total_pixels > 0 else 0.0
        
        # Connected components
        from scipy import ndimage as ndi
        labeled, num_components = ndi.label(mask_binary)
        
        # Component area statistics
        if num_components > 0:
            areas = np.array([np.sum(labeled == i) for i in range(1, num_components + 1)])
            mean_area = float(np.mean(areas))
            std_area = float(np.std(areas)) if len(areas) > 1 else 0.0
            max_area = float(np.max(areas))
            min_area = float(np.min(areas))
        else:
            mean_area = std_area = max_area = min_area = 0.0
        
        # Centroid of mass
        if np.sum(mask_binary) > 0:
            coords = np.argwhere(mask_binary)
            centroid_y = float(np.mean(coords[:, 0])) / mask.shape[0]  # Normalize
            centroid_x = float(np.mean(coords[:, 1])) / mask.shape[1]  # Normalize
        else:
            centroid_x = centroid_y = 0.0
        
        # Bounding box aspect ratio
        if np.sum(mask_binary) > 0:
            coords = np.argwhere(mask_binary)
            min_y, min_x = coords.min(axis=0)
            max_y, max_x = coords.max(axis=0)
            bbox_height = max_y - min_y + 1
            bbox_width = max_x - min_x + 1
            aspect_ratio = float(bbox_width / bbox_height) if bbox_height > 0 else 0.0
        else:
            aspect_ratio = 0.0
        
        # Combine all statistics
        stats = np.array([
            coverage,
            float(num_components),
            mean_area,
            std_area,
            max_area,
            min_area,
            centroid_x,
            centroid_y,
            aspect_ratio
        ], dtype=np.float32)
        
        return stats

    def extract_hog_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract HOG features only from image.
        
        Args:
            image: Input image (BGR)
            
        Returns:
            HOG feature vector
        """
        if image.shape[:2] != self.image_size:
            image = cv2.resize(image, self.image_size)
        
        if len(image.shape) == 3:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        hog_features = []
        for c in range(3):
            channel = rgb[:, :, c]
            hog_feat = hog(
                channel,
                orientations=9,
                pixels_per_cell=self.hog_cell_size,
                cells_per_block=(2, 2),
                block_norm='L2-Hys',
                feature_vector=True
            )
            hog_features.append(hog_feat)
        
        return np.concatenate(hog_features)
    
    def extract_lbp_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract LBP features only from image.
        
        Args:
            image: Input image (BGR)
            
        Returns:
            LBP feature vector
        """
        if image.shape[:2] != self.image_size:
            image = cv2.resize(image, self.image_size)
        
        if len(image.shape) == 3:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        lbp = local_binary_pattern(gray, 8, 1, method='uniform')
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 9))
        
        return lbp_hist.astype(np.float32)
    
    def extract_mask_stats_features(self, mask: np.ndarray) -> np.ndarray:
        """
        Extract mask statistics features only.
        
        Args:
            mask: Binary segmentation mask
            
        Returns:
            Mask statistics feature vector
        """
        return self.extract_mask_statistics(mask)
    
    def extract_features_with_mask_stats(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Extract combined HOG+LBP features from raw image and mask statistics.
        
        This extracts features from the raw (unsegmented) image and concatenates
        them with statistical features derived from the segmentation mask.
        
        Args:
            image: Raw image (BGR)
            mask: Binary segmentation mask
            
        Returns:
            Concatenated feature vector: [HOG+LBP features, mask statistics]
        """
        # Extract HOG+LBP from raw image (no mask applied)
        raw_features = self.extract_image_features(image, mask=None)
        
        # Extract mask statistics
        mask_stats = self.extract_mask_statistics(mask)
        
        # Concatenate
        combined_features = np.concatenate([raw_features, mask_stats])
        
        return combined_features

    def visualize_hog_features(self, annotations: List[Any], load_image_func, output_dir: str, num_samples: int = 3) -> None:
        """
        Visualize normalized image vs HOG features for sample images.
        
        Args:
            annotations: List of ImageAnnotation objects
            load_image_func: Function to load image from path
            output_dir: Directory to save visualizations
            num_samples: Max number of samples to visualize
        """
        import matplotlib.pyplot as plt
        from skimage import exposure

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        n_samples = min(num_samples, len(annotations), 3)
        sample_annotations = annotations[:n_samples]

        for annotation in sample_annotations:
            try:
                image = load_image_func(annotation.image_path)
                if image.shape[:2] != self.image_size:
                    image = cv2.resize(image, self.image_size)

                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Compute HOG visualization (average across channels)
                hog_images = []
                for c in range(3):
                    _, hog_img = hog(
                        rgb[:, :, c],
                        orientations=9,
                        pixels_per_cell=self.hog_cell_size,
                        cells_per_block=(2, 2),
                        block_norm='L2-Hys',
                        feature_vector=True,
                        visualize=True
                    )
                    hog_images.append(exposure.rescale_intensity(hog_img, in_range='image'))
                hog_image = np.mean(hog_images, axis=0)
                
                # Create side-by-side visualization
                fig, axes = plt.subplots(1, 2, figsize=(12, 6))
                fig.suptitle(f'HOG Features: {annotation.image_id} (Count: {annotation.nurdle_count})', 
                            fontsize=14, fontweight='bold')
                
                axes[0].imshow(rgb)
                axes[0].set_title('Normalized Image', fontsize=12, fontweight='bold')
                axes[0].axis('off')
                
                axes[1].imshow(hog_image, cmap='gray')
                axes[1].set_title('HOG Features', fontsize=12, fontweight='bold')
                axes[1].axis('off')
                
                plt.tight_layout()
                output_file = output_path / f'{annotation.image_id}_hog_features.png'
                plt.savefig(output_file, dpi=150, bbox_inches='tight')
                plt.close()
                self.logger.info(f"Saved HOG visualization: {output_file}")
            except Exception as e:
                self.logger.error(f"Error visualizing HOG features for {annotation.image_id}: {e}")

    def visualize_lbp_features(self, annotations: List[Any], load_image_func, output_dir: str, num_samples: int = 3) -> None:
        """
        Visualize normalized image vs LBP features for sample images.
        
        Args:
            annotations: List of ImageAnnotation objects
            load_image_func: Function to load image from path
            output_dir: Directory to save visualizations
            num_samples: Max number of samples to visualize
        """
        import matplotlib.pyplot as plt

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        n_samples = min(num_samples, len(annotations), 3)
        sample_annotations = annotations[:n_samples]

        for annotation in sample_annotations:
            try:
                image = load_image_func(annotation.image_path)
                if image.shape[:2] != self.image_size:
                    image = cv2.resize(image, self.image_size)

                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
                
                # Compute LBP
                lbp_image = local_binary_pattern(gray, 8, 1, method='uniform')
                
                # Create side-by-side visualization
                fig, axes = plt.subplots(1, 2, figsize=(12, 6))
                fig.suptitle(f'LBP Features: {annotation.image_id} (Count: {annotation.nurdle_count})', 
                            fontsize=14, fontweight='bold')
                
                axes[0].imshow(rgb)
                axes[0].set_title('Normalized Image', fontsize=12, fontweight='bold')
                axes[0].axis('off')
                
                axes[1].imshow(lbp_image, cmap='gray')
                axes[1].set_title('LBP Features', fontsize=12, fontweight='bold')
                axes[1].axis('off')
                
                plt.tight_layout()
                output_file = output_path / f'{annotation.image_id}_lbp_features.png'
                plt.savefig(output_file, dpi=150, bbox_inches='tight')
                plt.close()
                self.logger.info(f"Saved LBP visualization: {output_file}")
            except Exception as e:
                self.logger.error(f"Error visualizing LBP features for {annotation.image_id}: {e}")

    def visualize_mask_stats_features(self, annotations: List[Any], load_image_func, segmentation_func, output_dir: str, num_samples: int = 3) -> None:
        """
        Visualize segmentation mask vs extracted mask statistics for sample images.
        
        Args:
            annotations: List of ImageAnnotation objects
            load_image_func: Function to load image from path
            segmentation_func: Function to compute segmentation mask from image
            output_dir: Directory to save visualizations
            num_samples: Max number of samples to visualize
        """
        import matplotlib.pyplot as plt

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        n_samples = min(num_samples, len(annotations), 3)
        sample_annotations = annotations[:n_samples]

        for annotation in sample_annotations:
            try:
                image = load_image_func(annotation.image_path)
                if image.shape[:2] != self.image_size:
                    image = cv2.resize(image, self.image_size)

                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                mask = segmentation_func(image)
                
                # Compute mask statistics for info text
                mask_stats = self.extract_mask_stats_features(mask)
                stats_labels = [
                    f'Coverage: {mask_stats[0]:.2%}',
                    f'Components: {int(mask_stats[1])}',
                    f'Area: {mask_stats[2]:.0f}',
                    f'Centroids: ({mask_stats[3]:.0f}, {mask_stats[4]:.0f})',
                    f'Aspect Ratio: {mask_stats[5]:.2f}',
                    f'Solidity: {mask_stats[6]:.2f}',
                    f'Perimeter: {mask_stats[7]:.0f}',
                    f'Perimeter/Area: {mask_stats[8]:.4f}'
                ]
                
                # Create side-by-side visualization
                fig, axes = plt.subplots(1, 2, figsize=(12, 6))
                fig.suptitle(f'Mask Statistics: {annotation.image_id} (Count: {annotation.nurdle_count})', 
                            fontsize=14, fontweight='bold')
                
                axes[0].imshow(rgb)
                axes[0].set_title('Normalized Image', fontsize=12, fontweight='bold')
                axes[0].axis('off')
                
                # Overlay mask on second panel
                mask_colored = np.zeros((*mask.shape, 3), dtype=np.uint8)
                mask_colored[mask > 0] = [0, 255, 0]  # Green for mask
                axes[1].imshow(rgb)
                axes[1].imshow(mask_colored, alpha=0.4)
                axes[1].set_title('Segmentation Mask', fontsize=12, fontweight='bold')
                axes[1].axis('off')
                
                # Add statistics text
                stats_text = '\n'.join(stats_labels)
                fig.text(0.5, -0.05, stats_text, ha='center', fontsize=10, family='monospace',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                
                plt.tight_layout()
                output_file = output_path / f'{annotation.image_id}_mask_stats.png'
                plt.savefig(output_file, dpi=150, bbox_inches='tight')
                plt.close()
                self.logger.info(f"Saved mask statistics visualization: {output_file}")
            except Exception as e:
                self.logger.error(f"Error visualizing mask statistics for {annotation.image_id}: {e}")

