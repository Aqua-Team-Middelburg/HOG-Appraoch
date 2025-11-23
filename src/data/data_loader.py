"""
Data loading and management for nurdle detection pipeline.

This module handles:
- Loading image annotations from JSON files
- Image normalization and preprocessing  
- Train/test dataset splitting
- Memory-efficient batch processing
"""

import os
import json
import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Any, Iterator
from dataclasses import dataclass
import logging
import random


@dataclass
class NurdleAnnotation:
    """Single nurdle annotation with center coordinates."""
    x: float
    y: float
    

@dataclass
class ImageAnnotation:
    """Image annotation containing all nurdles."""
    image_path: str
    nurdles: List[NurdleAnnotation]
    image_id: str
    
    @property
    def nurdle_count(self) -> int:
        """Get number of nurdles in this image."""
        return len(self.nurdles)
    
    @property
    def coordinates(self) -> np.ndarray:
        """Get nurdle coordinates as array shape (N, 2)."""
        if not self.nurdles:
            return np.empty((0, 2))
        return np.array([[n.x, n.y] for n in self.nurdles])


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
        Load image annotations from JSON files.
        
        Expects JSON files with nurdle coordinates alongside image files.
        
        Args:
            input_dir: Directory containing images and JSON annotation files
        """
        self.logger.info(f"Loading annotations from {input_dir}")
        input_path = Path(input_dir)
        
        for json_file in input_path.glob("*.json"):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                # Extract image info
                image_id = json_file.stem
                image_path = str(json_file.with_suffix('.jpg'))  # Assume JPG images
                
                # If JPG doesn't exist, try other extensions
                if not Path(image_path).exists():
                    for ext in ['.png', '.jpeg', '.tiff']:
                        alt_path = str(json_file.with_suffix(ext))
                        if Path(alt_path).exists():
                            image_path = alt_path
                            break
                
                if not Path(image_path).exists():
                    self.logger.warning(f"Image file not found for {json_file}")
                    continue
                
                # Extract nurdle coordinates - handle multiple formats
                nurdles = []
                
                # Handle new format with 'objects' array
                if 'objects' in data:
                    for obj_data in data['objects']:
                        if 'center_x' in obj_data and 'center_y' in obj_data:
                            nurdles.append(NurdleAnnotation(
                                x=float(obj_data['center_x']), 
                                y=float(obj_data['center_y'])
                            ))
                
                # Handle legacy format with 'nurdles' array  
                elif 'nurdles' in data:
                    for nurdle_data in data['nurdles']:
                        if 'center' in nurdle_data:
                            center = nurdle_data['center']
                            nurdles.append(NurdleAnnotation(x=center[0], y=center[1]))
                
                annotation = ImageAnnotation(
                    image_path=image_path,
                    nurdles=nurdles,
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
        Save normalized images and metadata to checkpoint directory.
        
        Saves:
        - Normalized PNG images
        - JSON metadata with dimensions, scale factors, and train/test splits
        
        Args:
            checkpoint_dir: Directory to save checkpoint files
        """
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Saving normalized data checkpoint to {checkpoint_dir}")
        

        
        # Create subdirectories
        images_dir = checkpoint_dir / 'images'
        images_dir.mkdir(exist_ok=True)
        
        # Prepare metadata
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
        
        # Save all images and collect metadata
        for annotation in self.annotations:
            try:
                # Load original image to get scale factors
                original_image = cv2.imread(annotation.image_path)
                if original_image is None:
                    self.logger.warning(f"Could not load {annotation.image_path}")
                    continue
                
                orig_h, orig_w = original_image.shape[:2]
                
                # Use resolution-based normalization
                normalized_image = self.normalize_image(original_image)
                
                norm_h, norm_w = normalized_image.shape[:2]
                
                # Calculate scale factors
                scale_x = norm_w / orig_w
                scale_y = norm_h / orig_h
                
                # Store original coordinates
                original_nurdles = [
                    {'x': n.x, 'y': n.y}
                    for n in annotation.nurdles
                ]
                
                # Transform nurdle coordinates to normalized space
                transformed_nurdles = [
                    {'x': n.x * scale_x, 'y': n.y * scale_y}
                    for n in annotation.nurdles
                ]
                
                # Save normalized image
                image_filename = f"{annotation.image_id}.png"
                image_path = images_dir / image_filename
                cv2.imwrite(str(image_path), normalized_image)
                
                # Store metadata
                metadata['image_metadata'][annotation.image_id] = {
                    'original_path': annotation.image_path,
                    'normalized_path': str(image_path),
                    'original_dimensions': {'width': orig_w, 'height': orig_h},
                    'normalized_dimensions': {'width': norm_w, 'height': norm_h},
                    'scale_factors': {'x': scale_x, 'y': scale_y},
                    'original_nurdles': original_nurdles,
                    'nurdles': transformed_nurdles,
                    'nurdle_count': len(transformed_nurdles)
                }
                
            except Exception as e:
                self.logger.error(f"Error saving {annotation.image_path}: {e}")
        
        # Save metadata JSON
        metadata_path = checkpoint_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"Saved {len(metadata['image_metadata'])} normalized images")
    
    def load_normalized_data(self, checkpoint_dir: Path) -> None:
        """
        Load normalized images and metadata from checkpoint directory.
        
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
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Restore configuration
        self.target_resolution = metadata['target_resolution']
        self.train_test_ratio = metadata['train_test_ratio']
        
        # Reconstruct annotations
        self.annotations = []
        self.train_annotations = []
        self.test_annotations = []
        
        for image_id, img_meta in metadata['image_metadata'].items():
            # Create nurdle annotations with normalized coordinates
            nurdles = [
                NurdleAnnotation(x=n['x'], y=n['y'])
                for n in img_meta['nurdles']
            ]
            
            # Create annotation pointing to normalized image
            annotation = ImageAnnotation(
                image_path=img_meta['normalized_path'],
                nurdles=nurdles,
                image_id=image_id
            )
            
            self.annotations.append(annotation)
            
            # Sort into train/test
            if image_id in metadata['train_ids']:
                self.train_annotations.append(annotation)
            elif image_id in metadata['test_ids']:
                self.test_annotations.append(annotation)
        
        self.logger.info(f"Loaded {len(self.annotations)} annotations: "
                        f"{len(self.train_annotations)} train, {len(self.test_annotations)} test")
    
    def visualize_normalization_samples(self, checkpoint_dir: Path, output_dir: Path, num_samples: int = 3) -> None:
        """
        Generate visualization of normalized test images with annotations.
        
        IMPORTANT: Uses TEST SET samples only for visualization - NO data leakage!
        Test images are never used in training, only for demonstration purposes.
        
        Args:
            checkpoint_dir: Directory containing normalized data
            output_dir: Directory to save visualizations
            num_samples: Maximum number of samples (limited by test set size, max 3)
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Use TEST set samples only (no data leakage)
        if len(self.test_annotations) == 0:
            self.logger.warning("No test samples available for normalization visualization")
            return
        
        # Limit to min(num_samples, test set size, 3)
        n_samples = min(num_samples, len(self.test_annotations), 3)
        sample_annotations = self.test_annotations[:n_samples]
        
        self.logger.info(f"Generating normalization visualizations using {n_samples} TEST set samples (no data leakage)")
        
        # Load metadata to get original paths
        metadata_path = checkpoint_dir / 'metadata.json'
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Create one file per test image
        for annotation in sample_annotations:
            try:
                # Get original path from metadata
                img_meta = metadata['image_metadata'].get(annotation.image_id)
                if not img_meta:
                    self.logger.warning(f"No metadata found for {annotation.image_id}")
                    continue
                
                original_path = Path(img_meta['original_path'])
                if not original_path.exists():
                    self.logger.warning(f"Original image not found: {original_path}")
                    continue
                
                # Load original image
                original_img = cv2.imread(str(original_path))
                if original_img is None:
                    self.logger.warning(f"Failed to load original image: {original_path}")
                    continue
                original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
                
                # Load normalized image (try both .png and .jpg extensions)
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
                
                # Create figure with original and normalized side-by-side
                fig, axes = plt.subplots(1, 2, figsize=(16, 8))
                fig.suptitle(f'Image Normalization: {annotation.image_id} (TEST sample - Visualization Only)', 
                            fontsize=14, fontweight='bold')
                
                # Plot original with original coordinates
                axes[0].imshow(original_img)
                for nurdle_data in img_meta['original_nurdles']:
                    circle = Circle((nurdle_data['x'], nurdle_data['y']), radius=15, 
                                   color='lime', fill=False, linewidth=2)
                    axes[0].add_patch(circle)
                axes[0].set_title(f'Original ({img_meta["original_dimensions"]["width"]}x{img_meta["original_dimensions"]["height"]})\n{len(img_meta["original_nurdles"])} nurdles', 
                                 fontsize=12, fontweight='bold')
                axes[0].axis('off')
                
                # Plot normalized with normalized coordinates
                axes[1].imshow(normalized_img)
                for nurdle_data in img_meta['nurdles']:
                    circle = Circle((nurdle_data['x'], nurdle_data['y']), radius=15, 
                                   color='lime', fill=False, linewidth=2)
                    axes[1].add_patch(circle)
                axes[1].set_title(f'Normalized ({img_meta["normalized_dimensions"]["width"]}x{img_meta["normalized_dimensions"]["height"]})\n{len(img_meta["nurdles"])} nurdles', 
                                 fontsize=12, fontweight='bold')
                axes[1].axis('off')
                
                plt.tight_layout()
                output_file = output_path / f'{annotation.image_id}_normalization.png'
                plt.savefig(output_file, dpi=150, bbox_inches='tight')
                plt.close()
                
                self.logger.info(f"Saved normalization visualization: {output_file}")
                
            except Exception as e:
                self.logger.error(f"Error visualizing {annotation.image_id}: {e}")
        
        self.logger.info(f"Generated {n_samples} normalization visualization files")