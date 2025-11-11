"""
Image normalization module for the HOG/LBP/SVM pipeline.

This module handles the normalization of raw images to consistent dimensions
while maintaining aspect ratios and updating corresponding JSON annotations.
"""

import os
import json
import shutil
from PIL import Image
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import logging
from collections import Counter

from ..utils.config import ConfigLoader

logger = logging.getLogger(__name__)


class ImageNormalizer:
    """
    Normalizes images to consistent dimensions while preserving aspect ratios.
    
    This class handles:
    - Resizing images to fit within maximum dimensions
    - Updating JSON annotation coordinates accordingly
    - Maintaining data integrity during the normalization process
    """
    
    def __init__(self, config: ConfigLoader):
        """
        Initialize the image normalizer.
        
        Args:
            config: Configuration loader instance
        """
        self.config = config
        self.target_max_dim = config.get('preprocessing.normalization.target_max_dimension', 1080)
        self.preserve_aspect_ratio = config.get('preprocessing.normalization.preserve_aspect_ratio', True)
        
        # Get paths from config
        self.input_dir = Path(config.get('paths.input_dir'))
        self.output_dir = Path(config.get('paths.output_dir'))
        self.normalized_dir = Path(config.get('paths.temp_dir')) / 'normalized_images'
        
        # Supported image formats
        self.supported_formats = config.get('data.supported_formats', ['.jpg', '.jpeg', '.png', '.bmp'])
        
        # Create output directory
        self.normalized_dir.mkdir(parents=True, exist_ok=True)
        
        # Statistics tracking
        self.stats = {
            'total_images': 0,
            'normalized_images': 0,
            'skipped_images': 0,
            'resolution_counts': Counter(),
            'window_counts': [],
            'scale_factors': []
        }
    
    def analyze_input_data(self) -> Dict[str, Any]:
        """
        Analyze the input data to understand image dimensions and annotation counts.
        
        Returns:
            Dictionary containing analysis results
        """
        logger.info("Analyzing input data...")
        
        image_files = []
        for ext in self.supported_formats:
            image_files.extend(list(self.input_dir.glob(f"*{ext}")))
            image_files.extend(list(self.input_dir.glob(f"*{ext.upper()}")))
        
        logger.info(f"Found {len(image_files)} image files")
        
        resolution_counts = Counter()
        window_counts = []
        
        for img_path in image_files:
            try:
                with Image.open(img_path) as img:
                    resolution = (img.width, img.height)
                    resolution_counts[resolution] += 1
                
                # Check corresponding JSON file
                json_path = img_path.with_suffix('.json')
                if json_path.exists():
                    with open(json_path, 'r') as f:
                        json_data = json.load(f)
                    window_counts.append(len(json_data.get('objects', [])))
                else:
                    logger.warning(f"No JSON annotation found for {img_path.name}")
            
            except Exception as e:
                logger.error(f"Error analyzing {img_path.name}: {e}")
        
        analysis = {
            'total_images': len(image_files),
            'resolution_counts': dict(resolution_counts),
            'window_count_stats': {
                'min': min(window_counts) if window_counts else 0,
                'max': max(window_counts) if window_counts else 0,
                'avg': sum(window_counts) / len(window_counts) if window_counts else 0
            }
        }
        
        # Update internal stats
        self.stats['total_images'] = len(image_files)
        self.stats['resolution_counts'] = resolution_counts
        self.stats['window_counts'] = window_counts
        
        logger.info(f"Analysis complete: {analysis}")
        return analysis
    
    def calculate_target_dimensions(self, width: int, height: int) -> Tuple[int, int, float]:
        """
        Calculate target dimensions to fit within maximum dimension while maintaining aspect ratio.
        
        Args:
            width: Original image width
            height: Original image height
            
        Returns:
            Tuple of (new_width, new_height, scale_factor)
        """
        max_dimension = max(width, height)
        
        # If already at or below target, don't resize
        if max_dimension <= self.target_max_dim:
            return width, height, 1.0
        
        # Calculate scale factor to fit within target dimension
        scale_factor = self.target_max_dim / max_dimension
        
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        
        return new_width, new_height, scale_factor
    
    def scale_coordinates(self, json_data: Dict[str, Any], scale_factor: float) -> Dict[str, Any]:
        """
        Scale all coordinates in the JSON annotation data by the given scale factor.
        
        Args:
            json_data: Original JSON annotation data
            scale_factor: Scale factor to apply to coordinates
            
        Returns:
            Updated JSON data with scaled coordinates
        """
        # Create a copy to avoid modifying original
        scaled_data = json.deepcopy(json_data) if hasattr(json, 'deepcopy') else json.loads(json.dumps(json_data))
        
        # Update image dimensions
        if 'image' in scaled_data:
            scaled_data['image']['width'] = int(scaled_data['image']['width'] * scale_factor)
            scaled_data['image']['height'] = int(scaled_data['image']['height'] * scale_factor)
        
        # Update object coordinates
        for obj in scaled_data.get('objects', []):
            # Scale center coordinates
            if 'center_x' in obj:
                obj['center_x'] = int(obj['center_x'] * scale_factor)
            if 'center_y' in obj:
                obj['center_y'] = int(obj['center_y'] * scale_factor)
            
            # Scale radius (if present)
            if 'radius' in obj:
                obj['radius'] = max(1, int(obj['radius'] * scale_factor))
            
            # Scale bounding box coordinates
            if 'bbox' in obj:
                obj['bbox']['x'] = int(obj['bbox']['x'] * scale_factor)
                obj['bbox']['y'] = int(obj['bbox']['y'] * scale_factor)
                obj['bbox']['width'] = int(obj['bbox']['width'] * scale_factor)
                obj['bbox']['height'] = int(obj['bbox']['height'] * scale_factor)
        
        return scaled_data
    
    def normalize_single_image(self, image_path: Path) -> bool:
        """
        Normalize a single image and its corresponding JSON annotation.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            True if normalization successful, False otherwise
        """
        try:
            logger.debug(f"Normalizing {image_path.name}")
            
            # Load image
            with Image.open(image_path) as img:
                original_width, original_height = img.size
                
                # Calculate target dimensions
                new_width, new_height, scale_factor = self.calculate_target_dimensions(
                    original_width, original_height
                )
                
                # Skip if no scaling needed
                if scale_factor == 1.0:
                    logger.debug(f"Skipping {image_path.name} - already at target size")
                    # Copy to normalized directory anyway for consistency
                    shutil.copy2(image_path, self.normalized_dir / image_path.name)
                    json_path = image_path.with_suffix('.json')
                    if json_path.exists():
                        shutil.copy2(json_path, self.normalized_dir / json_path.name)
                    self.stats['skipped_images'] += 1
                    return True
                
                # Resize image
                resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                
                # Save resized image
                output_image_path = self.normalized_dir / image_path.name
                resized_img.save(output_image_path, quality=95, optimize=True)
                
                # Process corresponding JSON file
                json_path = image_path.with_suffix('.json')
                if json_path.exists():
                    with open(json_path, 'r') as f:
                        json_data = json.load(f)
                    
                    # Scale coordinates
                    scaled_json_data = self.scale_coordinates(json_data, scale_factor)
                    
                    # Save scaled JSON
                    output_json_path = self.normalized_dir / json_path.name
                    with open(output_json_path, 'w') as f:
                        json.dump(scaled_json_data, f, indent=2)
                
                # Track statistics
                self.stats['normalized_images'] += 1
                self.stats['scale_factors'].append(scale_factor)
                
                logger.debug(f"Normalized {image_path.name}: {original_width}x{original_height} -> {new_width}x{new_height} (scale: {scale_factor:.3f})")
                return True
                
        except Exception as e:
            logger.error(f"Error normalizing {image_path.name}: {e}")
            return False
    
    def normalize_all_images(self) -> Dict[str, Any]:
        """
        Normalize all images in the input directory.
        
        Returns:
            Dictionary containing normalization results and statistics
        """
        logger.info("Starting image normalization process...")
        
        # First analyze the data
        analysis = self.analyze_input_data()
        
        # Get all image files
        image_files = []
        for ext in self.supported_formats:
            image_files.extend(list(self.input_dir.glob(f"*{ext}")))
            image_files.extend(list(self.input_dir.glob(f"*{ext.upper()}")))
        
        # Process each image
        successful_normalizations = 0
        failed_normalizations = 0
        
        for i, image_path in enumerate(image_files, 1):
            logger.info(f"Processing image {i}/{len(image_files)}: {image_path.name}")
            
            if self.normalize_single_image(image_path):
                successful_normalizations += 1
            else:
                failed_normalizations += 1
        
        # Calculate final statistics
        results = {
            'input_analysis': analysis,
            'total_processed': len(image_files),
            'successful_normalizations': successful_normalizations,
            'failed_normalizations': failed_normalizations,
            'skipped_images': self.stats['skipped_images'],
            'scale_factor_stats': {
                'min': min(self.stats['scale_factors']) if self.stats['scale_factors'] else 1.0,
                'max': max(self.stats['scale_factors']) if self.stats['scale_factors'] else 1.0,
                'avg': sum(self.stats['scale_factors']) / len(self.stats['scale_factors']) if self.stats['scale_factors'] else 1.0
            },
            'output_directory': str(self.normalized_dir)
        }
        
        logger.info(f"Normalization complete: {results}")
        return results
    
    def validate_normalization(self) -> Dict[str, Any]:
        """
        Validate that normalization was successful by checking output files.
        
        Returns:
            Dictionary containing validation results
        """
        logger.info("Validating normalization results...")
        
        normalized_images = list(self.normalized_dir.glob("*.jpg"))
        normalized_images.extend(list(self.normalized_dir.glob("*.jpeg")))
        normalized_images.extend(list(self.normalized_dir.glob("*.png")))
        
        validation_results = {
            'normalized_image_count': len(normalized_images),
            'dimension_check': True,
            'json_integrity_check': True,
            'issues': []
        }
        
        for img_path in normalized_images:
            try:
                # Check image dimensions
                with Image.open(img_path) as img:
                    max_dim = max(img.width, img.height)
                    if max_dim > self.target_max_dim * 1.1:  # Allow 10% tolerance
                        validation_results['dimension_check'] = False
                        validation_results['issues'].append(f"Image {img_path.name} exceeds target dimension: {max_dim}")
                
                # Check JSON integrity
                json_path = img_path.with_suffix('.json')
                if json_path.exists():
                    with open(json_path, 'r') as f:
                        json_data = json.load(f)
                    
                    # Basic validation
                    if 'objects' not in json_data:
                        validation_results['json_integrity_check'] = False
                        validation_results['issues'].append(f"JSON {json_path.name} missing 'objects' key")
                        
            except Exception as e:
                validation_results['issues'].append(f"Error validating {img_path.name}: {e}")
        
        logger.info(f"Validation complete: {len(validation_results['issues'])} issues found")
        return validation_results