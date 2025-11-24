"""
Feature extraction for nurdle detection pipeline.

This module provides combined HOG and LBP feature extraction from full images for SVM count classification.
"""

# Required imports
import logging
from typing import Dict, Any, List, Iterator, Tuple
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

    def extract_image_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract combined RGB HOG (per channel) and grayscale LBP features.
        Image is resized to config.image_size before feature extraction.
        """
        if image.shape[:2] != self.image_size:
            image = cv2.resize(image, self.image_size)

        # Ensure RGB ordering for per-channel HOG
        if len(image.shape) == 3:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # HOG per channel (captures color gradients)
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

        # LBP on grayscale for texture
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        lbp = local_binary_pattern(gray, 8, 1, method='uniform')
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 9))
        lbp_features = lbp_hist.astype(float)

        combined_features = np.concatenate([hog_features, lbp_features])
        return combined_features

    def extract_features(self, image_path: str) -> np.ndarray:
        """
        Load image from path and extract combined HOG+LBP features.
        """
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        return self.extract_image_features(image)

    def generate_training_data(self, annotations: List[Any], load_image_func) -> Iterator[Tuple[np.ndarray, int]]:
        """
        Yield (features, count) pairs for each image annotation.
        """
        for annotation in annotations:
            image = load_image_func(annotation.image_path)
            features = self.extract_image_features(image)
            count = annotation.nurdle_count
            yield (features, count)

    def visualize_feature_samples(self, test_annotations: List[Any], load_image_func, output_dir: str, num_samples: int = 3) -> None:
        """
        Save three images per test sample: original, HOG, and LBP.
        Args:
            test_annotations: List of ImageAnnotation objects (test set)
            load_image_func: Function to load image from path
            output_dir: Directory to save visualizations
            num_samples: Max number of samples (default 3)
        """
        import matplotlib.pyplot as plt
        from skimage import exposure

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        n_samples = min(num_samples, len(test_annotations), 3)
        sample_annotations = test_annotations[:n_samples]

        for annotation in sample_annotations:
            try:
                image = load_image_func(annotation.image_path)
                if image.shape[:2] != self.image_size:
                    image = cv2.resize(image, self.image_size)

                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

                # HOG per channel (visualize: average of channel HOG images)
                hog_images = []
                for c in range(3):
                    hog_feat, hog_img = hog(
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

                # LBP features and image
                lbp = local_binary_pattern(gray, 8, 1, method='uniform')

                fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                fig.suptitle(f'Feature Visualization: {annotation.image_id} (TEST sample)', fontsize=14, fontweight='bold')
                axes[0].imshow(image, cmap='gray')
                axes[0].set_title(f'Original\nCount: {annotation.nurdle_count}', fontsize=12, fontweight='bold')
                axes[0].axis('off')
                axes[1].imshow(hog_image, cmap='gray')
                axes[1].set_title('HOG Features', fontsize=12, fontweight='bold')
                axes[1].axis('off')
                axes[2].imshow(lbp, cmap='gray')
                axes[2].set_title('LBP Features', fontsize=12, fontweight='bold')
                axes[2].axis('off')
                plt.tight_layout()
                output_file = output_path / f'{annotation.image_id}_features.png'
                plt.savefig(output_file, dpi=150, bbox_inches='tight')
                plt.close()
                self.logger.info(f"Saved feature visualization: {output_file}")
            except Exception as e:
                self.logger.error(f"Error visualizing features for {annotation.image_id}: {e}")

