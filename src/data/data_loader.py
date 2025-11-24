"""
Data loading and management for single-stage SVM nurdle count prediction pipeline.

This module handles:
- Loading image annotations from JSON files
- Image normalization and preprocessing
- Train/test dataset splitting
- Memory-efficient batch processing
"""

import json
import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Any, Iterator
from dataclasses import dataclass
import logging


@dataclass
class ImageAnnotation:
    """Image annotation for count prediction (whole-image)."""
    def __init__(self, image_path: str, nurdle_count: int, image_id: str):
        self.image_path = image_path
        self.nurdle_count = nurdle_count
        self.image_id = image_id


class DataLoader:
    """
    Handles data loading, normalization, and splitting for the pipeline.
    
    Provides memory-efficient loading and processing of images with annotations.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize data loader with configuration."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Configuration parameters
        self.train_test_ratio = config.get('train_test_ratio', 0.8)
        self.target_resolution = config.get('target_resolution', 1080)
        self.batch_size = config.get('batch_size', 15)
        
        # Data storage
        self.annotations: List[ImageAnnotation] = []
        self.train_annotations: List[ImageAnnotation] = []
        self.test_annotations: List[ImageAnnotation] = []
        
    def load_annotations(self, input_dir: str) -> None:
        """
        Load image annotations from JSON files for count prediction.
        Args:
            input_dir: Directory containing images and JSON annotation files
        """
        self.logger.info(f"Loading annotations from {input_dir}")
        input_path = Path(input_dir)
        for json_file in input_path.glob("*.json"):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                image_id = json_file.stem
                image_path = str(json_file.with_suffix('.jpg'))
                if not Path(image_path).exists():
                    for ext in ['.png', '.jpeg', '.tiff']:
                        alt_path = str(json_file.with_suffix(ext))
                        if Path(alt_path).exists():
                            image_path = alt_path
                            break
                if not Path(image_path).exists():
                    self.logger.warning(f"Image file not found for {json_file}")
                    continue
                # Count nurdles from annotation file
                nurdle_count = 0
                if 'objects' in data:
                    nurdle_count = len(data['objects'])
                elif 'nurdles' in data:
                    nurdle_count = len(data['nurdles'])
                annotation = ImageAnnotation(
                    image_path=image_path,
                    nurdle_count=nurdle_count,
                    image_id=image_id
                )
                self.annotations.append(annotation)
            except Exception as e:
                self.logger.error(f"Error loading annotation {json_file}: {e}")
        self.logger.info(f"Loaded {len(self.annotations)} annotations")
    

    
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image to target resolution while maintaining aspect ratio.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Normalized image that fits within target resolution
        """
        h, w = image.shape[:2]
        
        # Calculate scale to fit within target resolution
        scale = min(self.target_resolution / w, self.target_resolution / h)
        
        if scale < 1.0:  # Only downscale, never upscale
            new_w = int(w * scale)
            new_h = int(h * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        return image
    
    def split_train_test(self) -> None:
        """Split annotations into training and test sets."""
        if not self.annotations:
            raise ValueError("No annotations loaded. Call load_annotations() first.")
            
        train_size = int(len(self.annotations) * self.train_test_ratio)
        
        # Shuffle annotations for random split
        np.random.seed(42)  # For reproducibility
        indices = np.random.permutation(len(self.annotations))
        
        train_indices = indices[:train_size]
        test_indices = indices[train_size:]
        
        self.train_annotations = [self.annotations[i] for i in train_indices]
        self.test_annotations = [self.annotations[i] for i in test_indices]
        
        self.logger.info(f"Split: {len(self.train_annotations)} training, {len(self.test_annotations)} test images")
    
    def get_image_batches(self, annotations: List[ImageAnnotation]) -> Iterator[List[ImageAnnotation]]:
        """
        Generate batches of annotations for memory-efficient processing.
        
        Args:
            annotations: List of image annotations to batch
            
        Yields:
            Batches of annotations
        """
        for i in range(0, len(annotations), self.batch_size):
            batch = annotations[i:i + self.batch_size]
            yield batch
    

    
    def load_image(self, image_path: str) -> np.ndarray:
        """
        Load and normalize a single image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Normalized image as numpy array
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        return self.normalize_image(image)
    
    @property
    def num_annotations(self) -> int:
        """Get total number of loaded annotations."""
        return len(self.annotations)
    
    @property
    def num_train(self) -> int:
        """Get number of training annotations."""
        return len(self.train_annotations)
    
    @property
    def num_test(self) -> int:
        """Get number of test annotations."""
        return len(self.test_annotations)
    
    def save_normalized_data(self, checkpoint_dir: Path) -> None:
        """
        Save normalized images and metadata to checkpoint directory for count prediction.
        Args:
            checkpoint_dir: Directory to save checkpoint files
        """
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Saving normalized data checkpoint to {checkpoint_dir}")
        images_dir = checkpoint_dir / 'images'
        images_dir.mkdir(exist_ok=True)
        metadata = {
            'target_resolution': self.target_resolution,
            'train_test_ratio': self.train_test_ratio,
            'total_images': len(self.annotations),
            'train_images': len(self.train_annotations),
            'test_images': len(self.test_annotations),
            'train_ids': [ann.image_id for ann in self.train_annotations],
            'test_ids': [ann.image_id for ann in self.test_annotations],
            'image_metadata': {}
        }
        for annotation in self.annotations:
            try:
                original_image = cv2.imread(annotation.image_path)
                if original_image is None:
                    self.logger.warning(f"Could not load {annotation.image_path}")
                    continue
                orig_h, orig_w = original_image.shape[:2]
                normalized_image = self.normalize_image(original_image)
                norm_h, norm_w = normalized_image.shape[:2]
                scale_x = norm_w / orig_w
                scale_y = norm_h / orig_h
                image_filename = f"{annotation.image_id}.png"
                image_path = images_dir / image_filename
                cv2.imwrite(str(image_path), normalized_image)
                metadata['image_metadata'][annotation.image_id] = {
                    'original_path': annotation.image_path,
                    'normalized_path': str(image_path),
                    'original_dimensions': {'width': orig_w, 'height': orig_h},
                    'normalized_dimensions': {'width': norm_w, 'height': norm_h},
                    'scale_factors': {'x': scale_x, 'y': scale_y},
                    'nurdle_count': annotation.nurdle_count
                }
            except Exception as e:
                self.logger.error(f"Error saving {annotation.image_path}: {e}")
        metadata_path = checkpoint_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        self.logger.info(f"Saved {len(metadata['image_metadata'])} normalized images")
    
    def load_normalized_data(self, checkpoint_dir: Path) -> None:
        """
        Load normalized images and metadata from checkpoint directory for count prediction.
        Args:
            checkpoint_dir: Directory containing checkpoint files
        Raises:
            FileNotFoundError: If checkpoint directory or metadata file not found
        """
        checkpoint_dir = Path(checkpoint_dir)
        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
        metadata_path = checkpoint_dir / 'metadata.json'
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        self.logger.info(f"Loading normalized data checkpoint from {checkpoint_dir}")
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        self.target_resolution = metadata['target_resolution']
        self.train_test_ratio = metadata['train_test_ratio']
        self.annotations = []
        self.train_annotations = []
        self.test_annotations = []
        for image_id, img_meta in metadata['image_metadata'].items():
            annotation = ImageAnnotation(
                image_path=img_meta['normalized_path'],
                nurdle_count=img_meta['nurdle_count'],
                image_id=image_id
            )
            self.annotations.append(annotation)
            if image_id in metadata['train_ids']:
                self.train_annotations.append(annotation)
            elif image_id in metadata['test_ids']:
                self.test_annotations.append(annotation)
        self.logger.info(f"Loaded {len(self.annotations)} annotations: "
                        f"{len(self.train_annotations)} train, {len(self.test_annotations)} test")
    
    def visualize_normalization_samples(self, checkpoint_dir: Path, output_dir: Path, num_samples: int = 3) -> None:
        """
        Generate visualization of normalized test images with count annotation only.
        IMPORTANT: Uses TEST SET samples only for visualization - NO data leakage!
        Args:
            checkpoint_dir: Directory containing normalized data
            output_dir: Directory to save visualizations
            num_samples: Maximum number of samples (limited by test set size, max 3)
        """
        import matplotlib.pyplot as plt

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if len(self.test_annotations) == 0:
            self.logger.warning("No test samples available for normalization visualization")
            return

        n_samples = min(num_samples, len(self.test_annotations), 3)
        sample_annotations = self.test_annotations[:n_samples]

        self.logger.info(f"Generating normalization visualizations using {n_samples} TEST set samples (no data leakage)")

        metadata_path = checkpoint_dir / 'metadata.json'
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        for annotation in sample_annotations:
            try:
                img_meta = metadata['image_metadata'].get(annotation.image_id)
                if not img_meta:
                    self.logger.warning(f"No metadata found for {annotation.image_id}")
                    continue
                original_path = Path(img_meta['original_path'])
                if not original_path.exists():
                    self.logger.warning(f"Original image not found: {original_path}")
                    continue
                original_img = cv2.imread(str(original_path))
                if original_img is None:
                    self.logger.warning(f"Failed to load original image: {original_path}")
                    continue
                original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
                normalized_path = checkpoint_dir / 'images' / f"{annotation.image_id}.png"
                if not normalized_path.exists():
                    normalized_path = checkpoint_dir / 'images' / f"{annotation.image_id}.jpg"
                if not normalized_path.exists():
                    self.logger.warning(f"Normalized image not found: {normalized_path}")
                    continue
                normalized_img = cv2.imread(str(normalized_path))
                if normalized_img is None:
                    self.logger.warning(f"Failed to load normalized image: {normalized_path}")
                    continue
                normalized_img = cv2.cvtColor(normalized_img, cv2.COLOR_BGR2RGB)
                fig, axes = plt.subplots(1, 2, figsize=(16, 8))
                fig.suptitle(f'Image Normalization: {annotation.image_id} (TEST sample - Visualization Only)', fontsize=14, fontweight='bold')
                axes[0].imshow(original_img)
                axes[0].set_title(f'Original ({img_meta["original_dimensions"]["width"]}x{img_meta["original_dimensions"]["height"]})\nCount: {annotation.nurdle_count}', fontsize=12, fontweight='bold')
                axes[0].axis('off')
                axes[1].imshow(normalized_img)
                axes[1].set_title(f'Normalized ({img_meta["normalized_dimensions"]["width"]}x{img_meta["normalized_dimensions"]["height"]})\nCount: {annotation.nurdle_count}', fontsize=12, fontweight='bold')
                axes[1].axis('off')
                plt.tight_layout()
                output_file = output_path / f'{annotation.image_id}_normalization.png'
                plt.savefig(output_file, dpi=150, bbox_inches='tight')
                plt.close()
                self.logger.info(f"Saved normalization visualization: {output_file}")
            except Exception as e:
                self.logger.error(f"Error visualizing {annotation.image_id}: {e}")
        self.logger.info(f"Generated {n_samples} normalization visualization files")
