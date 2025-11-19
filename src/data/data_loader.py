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
    
    def get_normalization_scale(self, image: np.ndarray) -> tuple:
        """
        Get the scale factors that would be applied when normalizing an image.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Tuple of (scale_x, scale_y) - will be 1.0 if no scaling needed
        """
        h, w = image.shape[:2]
        scale = min(self.target_resolution / w, self.target_resolution / h)
        
        if scale < 1.0:
            return (scale, scale)
        return (1.0, 1.0)
    
    def transform_coordinates(self, annotation: ImageAnnotation, original_image: np.ndarray) -> ImageAnnotation:
        """
        Transform annotation coordinates to normalized image space.
        
        Args:
            annotation: Original annotation with coordinates in original image space
            original_image: The original image (to calculate scale factors)
            
        Returns:
            New ImageAnnotation with transformed coordinates
        """
        scale_x, scale_y = self.get_normalization_scale(original_image)
        
        transformed_nurdles = [
            NurdleAnnotation(x=n.x * scale_x, y=n.y * scale_y) 
            for n in annotation.nurdles
        ]
        
        return ImageAnnotation(
            image_path=annotation.image_path,
            nurdles=transformed_nurdles,
            image_id=annotation.image_id
        )
    
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