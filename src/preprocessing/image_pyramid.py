"""
Image pyramid module for multi-scale object detection.

This module implements image pyramids to enable detection at multiple scales,
improving detection accuracy for objects of varying sizes.
"""

import numpy as np
import cv2
import logging
from typing import List, Tuple, Generator, Optional
from dataclasses import dataclass

from ..utils.config import ConfigLoader

logger = logging.getLogger(__name__)


@dataclass
class PyramidLevel:
    """
    Represents a single level in the image pyramid.
    
    Attributes:
        scale: Scale factor relative to original image
        image: Resized image at this scale
        level: Pyramid level index (0 = original)
    """
    scale: float
    image: np.ndarray
    level: int


class ImagePyramid:
    """
    Generates multi-scale image pyramids for object detection.
    
    Creates progressively downsampled versions of an image to enable
    detection of objects at different scales. Supports both fixed-factor
    and adaptive scaling strategies.
    """
    
    def __init__(self, config: ConfigLoader):
        """
        Initialize image pyramid generator.
        
        Args:
            config: Configuration loader instance
        """
        self.config = config
        
        # Load pyramid configuration
        pyramid_config = config.get_section('multi_scale_detection').get('pyramid', {})
        
        self.enabled = pyramid_config.get('enabled', False)
        self.scale_factor = pyramid_config.get('scale_factor', 1.5)
        self.min_size = tuple(pyramid_config.get('min_size', [40, 40]))
        self.max_levels = pyramid_config.get('max_levels', 5)
        self.interpolation = pyramid_config.get('interpolation', 'linear')
        
        # Interpolation mapping
        self.interp_methods = {
            'nearest': cv2.INTER_NEAREST,
            'linear': cv2.INTER_LINEAR,
            'cubic': cv2.INTER_CUBIC,
            'area': cv2.INTER_AREA,
            'lanczos': cv2.INTER_LANCZOS4
        }
        
        logger.info(f"Image pyramid initialized: scale_factor={self.scale_factor}, "
                   f"min_size={self.min_size}, max_levels={self.max_levels}")
    
    def generate_pyramid(self, image: np.ndarray) -> List[PyramidLevel]:
        """
        Generate full image pyramid with all scale levels.
        
        Args:
            image: Original input image (H × W or H × W × C)
            
        Returns:
            List of PyramidLevel objects from coarse to fine
        """
        if not self.enabled:
            # Return single level with original image
            return [PyramidLevel(scale=1.0, image=image, level=0)]
        
        pyramid = []
        current_image = image.copy()
        level = 0
        scale = 1.0
        
        logger.debug(f"Generating pyramid for image {image.shape}")
        
        # Add original image
        pyramid.append(PyramidLevel(scale=scale, image=current_image, level=level))
        
        # Generate downsampled levels
        while level < self.max_levels:
            # Calculate new scale
            scale /= self.scale_factor
            new_size = (
                int(image.shape[1] * scale),  # width
                int(image.shape[0] * scale)   # height
            )
            
            # Check if we've reached minimum size
            if new_size[0] < self.min_size[0] or new_size[1] < self.min_size[1]:
                logger.debug(f"Stopping pyramid at level {level}: size {new_size} below minimum {self.min_size}")
                break
            
            # Resize image
            interp_method = self.interp_methods.get(self.interpolation, cv2.INTER_LINEAR)
            current_image = cv2.resize(image, new_size, interpolation=interp_method)
            
            level += 1
            pyramid.append(PyramidLevel(scale=scale, image=current_image, level=level))
            
            logger.debug(f"Pyramid level {level}: scale={scale:.3f}, size={current_image.shape[:2]}")
        
        logger.info(f"Generated {len(pyramid)} pyramid levels")
        return pyramid
    
    def generate_pyramid_generator(self, image: np.ndarray) -> Generator[PyramidLevel, None, None]:
        """
        Generate pyramid levels lazily using a generator.
        
        Memory-efficient alternative to generate_pyramid() for large images.
        
        Args:
            image: Original input image
            
        Yields:
            PyramidLevel objects from coarse to fine
        """
        if not self.enabled:
            yield PyramidLevel(scale=1.0, image=image, level=0)
            return
        
        current_image = image.copy()
        level = 0
        scale = 1.0
        
        # Yield original image
        yield PyramidLevel(scale=scale, image=current_image, level=level)
        
        # Generate downsampled levels
        while level < self.max_levels:
            scale /= self.scale_factor
            new_size = (
                int(image.shape[1] * scale),
                int(image.shape[0] * scale)
            )
            
            if new_size[0] < self.min_size[0] or new_size[1] < self.min_size[1]:
                break
            
            interp_method = self.interp_methods.get(self.interpolation, cv2.INTER_LINEAR)
            current_image = cv2.resize(image, new_size, interpolation=interp_method)
            
            level += 1
            yield PyramidLevel(scale=scale, image=current_image, level=level)
    
    def map_coordinates_to_original(self, 
                                   x: int, y: int, w: int, h: int,
                                   scale: float) -> Tuple[int, int, int, int]:
        """
        Map bounding box coordinates from pyramid level to original image scale.
        
        Args:
            x: X coordinate at pyramid scale
            y: Y coordinate at pyramid scale
            w: Width at pyramid scale
            h: Height at pyramid scale
            scale: Scale factor of pyramid level (< 1.0 for downsampled)
            
        Returns:
            Tuple of (x, y, w, h) in original image coordinates
        """
        # Scale coordinates back to original image
        orig_x = int(x / scale)
        orig_y = int(y / scale)
        orig_w = int(w / scale)
        orig_h = int(h / scale)
        
        return (orig_x, orig_y, orig_w, orig_h)
    
    def map_coordinates_from_original(self,
                                     x: int, y: int, w: int, h: int,
                                     scale: float) -> Tuple[int, int, int, int]:
        """
        Map bounding box coordinates from original image to pyramid level.
        
        Args:
            x: X coordinate in original image
            y: Y coordinate in original image
            w: Width in original image
            h: Height in original image
            scale: Scale factor of target pyramid level
            
        Returns:
            Tuple of (x, y, w, h) at pyramid scale
        """
        # Scale coordinates to pyramid level
        scaled_x = int(x * scale)
        scaled_y = int(y * scale)
        scaled_w = int(w * scale)
        scaled_h = int(h * scale)
        
        return (scaled_x, scaled_y, scaled_w, scaled_h)
    
    def get_scale_for_size(self, target_size: Tuple[int, int], 
                          image_shape: Tuple[int, int]) -> float:
        """
        Calculate scale factor to resize image to target size.
        
        Args:
            target_size: Desired (width, height)
            image_shape: Current (height, width) from image.shape
            
        Returns:
            Scale factor to achieve target size
        """
        current_h, current_w = image_shape[:2]
        target_w, target_h = target_size
        
        # Calculate scale based on smaller dimension to ensure target is not exceeded
        scale_w = target_w / current_w
        scale_h = target_h / current_h
        
        return min(scale_w, scale_h)
    
    def get_effective_scales(self, image: np.ndarray) -> List[float]:
        """
        Get list of scale factors that will be generated for an image.
        
        Useful for pre-computing without generating full pyramid.
        
        Args:
            image: Input image to analyze
            
        Returns:
            List of scale factors
        """
        if not self.enabled:
            return [1.0]
        
        scales = [1.0]
        level = 0
        scale = 1.0
        
        while level < self.max_levels:
            scale /= self.scale_factor
            new_size = (
                int(image.shape[1] * scale),
                int(image.shape[0] * scale)
            )
            
            if new_size[0] < self.min_size[0] or new_size[1] < self.min_size[1]:
                break
            
            scales.append(scale)
            level += 1
        
        return scales
    
    def visualize_pyramid(self, pyramid: List[PyramidLevel]) -> np.ndarray:
        """
        Create visualization of pyramid levels side by side.
        
        Args:
            pyramid: List of pyramid levels
            
        Returns:
            Visualization image with all levels
        """
        if not pyramid:
            return np.array([])
        
        # Calculate total width needed
        max_height = pyramid[0].image.shape[0]
        total_width = sum(level.image.shape[1] for level in pyramid)
        
        # Handle grayscale vs color
        if len(pyramid[0].image.shape) == 3:
            vis_image = np.zeros((max_height, total_width, 3), dtype=np.uint8)
        else:
            vis_image = np.zeros((max_height, total_width), dtype=np.uint8)
        
        # Place each level
        x_offset = 0
        for level in pyramid:
            h, w = level.image.shape[:2]
            
            # Place image at top-left of section
            if len(level.image.shape) == 3:
                vis_image[:h, x_offset:x_offset+w, :] = level.image
            else:
                vis_image[:h, x_offset:x_offset+w] = level.image
            
            x_offset += w
        
        return vis_image
    
    def get_config(self) -> dict:
        """
        Get current pyramid configuration.
        
        Returns:
            Configuration dictionary
        """
        return {
            'enabled': self.enabled,
            'scale_factor': self.scale_factor,
            'min_size': self.min_size,
            'max_levels': self.max_levels,
            'interpolation': self.interpolation
        }
