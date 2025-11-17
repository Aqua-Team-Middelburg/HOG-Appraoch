"""
HOG (Histogram of Oriented Gradients) feature extractor.

This module provides HOG feature extraction with consistent parameters
matching the original training configuration.
"""

import cv2
import numpy as np
from skimage import feature
import logging
from typing import Dict, Any, Tuple, Optional

from ..utils.config import ConfigLoader

logger = logging.getLogger(__name__)


class HOGExtractor:
    """
    Extracts HOG (Histogram of Oriented Gradients) features from image windows.
    
    This class ensures consistent HOG parameter usage across training and inference,
    matching the original preprocessing pipeline specifications.
    """
    
    def __init__(self, config: ConfigLoader):
        """
        Initialize HOG extractor with configuration.
        
        Args:
            config: Configuration loader instance
        """
        self.config = config
        
        # Load HOG parameters from config
        hog_config = config.get_section('features').get('hog', {})
        
        self.orientations = hog_config.get('orientations', 9)
        self.pixels_per_cell = tuple(hog_config.get('pixels_per_cell', [8, 8]))
        self.cells_per_block = tuple(hog_config.get('cells_per_block', [2, 2]))
        self.block_norm = hog_config.get('block_norm', 'L2-Hys')
        self.transform_sqrt = hog_config.get('transform_sqrt', False)
        
        # Color feature configuration
        self.use_color = hog_config.get('use_color', False)
        self.color_space = hog_config.get('color_space', 'RGB')
        
        # Calculate expected feature vector size
        self.feature_vector_size = self._calculate_feature_size()
        
        color_mode = f"{self.color_space} multi-channel" if self.use_color else "grayscale"
        logger.info(f"HOG extractor initialized: {self.orientations} orientations, "
                   f"{self.pixels_per_cell} pixels/cell, {self.cells_per_block} cells/block, "
                   f"{color_mode}, expected vector size: {self.feature_vector_size}")

    
    def _calculate_feature_size(self) -> int:
        """
        Calculate the expected HOG feature vector size.
        
        For multi-channel mode (RGB), the feature vector is 3× larger.
        For a 40x40 window with 8x8 cells and 2x2 blocks:
        - Cells: (40/8, 40/8) = (5, 5) = 25 cells
        - Blocks: (5-2+1, 5-2+1) = (4, 4) = 16 blocks  
        - Features per block: orientations * cells_per_block = 9 * 2 * 2 = 36
        - Single channel: 16 * 36 = 576 (but typically reduced to 81 based on actual extraction)
        - Multi-channel (RGB): 81 * 3 = 243
        
        Returns:
            Expected feature vector size
        """
        # This calculation assumes 40x40 windows - should be made dynamic
        window_size = tuple(self.config.get('preprocessing.target_size', [40, 40]))
        
        # Calculate number of cells
        n_cells_x = window_size[0] // self.pixels_per_cell[0]
        n_cells_y = window_size[1] // self.pixels_per_cell[1]
        
        # Calculate number of blocks
        n_blocks_x = n_cells_x - self.cells_per_block[0] + 1
        n_blocks_y = n_cells_y - self.cells_per_block[1] + 1
        
        # Total features per channel
        features_per_block = self.orientations * self.cells_per_block[0] * self.cells_per_block[1]
        single_channel_features = n_blocks_x * n_blocks_y * features_per_block
        
        # Multiply by number of channels if using color
        if self.use_color:
            total_features = single_channel_features * 3  # RGB = 3 channels
        else:
            total_features = single_channel_features
        
        return max(total_features, 1)  # Ensure at least 1 feature
    
    def _extract_channel_hog(self, channel: np.ndarray) -> np.ndarray:
        """
        Extract HOG features from a single channel (grayscale image).
        
        Args:
            channel: Single channel image as 2D numpy array
            
        Returns:
            HOG feature vector for this channel
        """
        # Ensure correct data type
        if channel.dtype != np.uint8:
            channel = (channel * 255).astype(np.uint8) if channel.max() <= 1.0 else channel.astype(np.uint8)
        
        try:
            # Extract HOG features
            features = feature.hog(
                channel,
                orientations=self.orientations,
                pixels_per_cell=self.pixels_per_cell,
                cells_per_block=self.cells_per_block,
                block_norm=self.block_norm,
                transform_sqrt=self.transform_sqrt,
                visualize=False,
                feature_vector=True
            )
            return features
        except Exception as e:
            logger.error(f"Error extracting HOG from channel: {e}")
            # Return zeros based on expected single-channel size
            single_channel_size = self.feature_vector_size // (3 if self.use_color else 1)
            return np.zeros(single_channel_size)

    
    def extract_features(self, window: np.ndarray) -> np.ndarray:
        """
        Extract HOG features from a single window.
        
        Supports both grayscale and multi-channel (RGB) extraction based on configuration.
        
        Args:
            window: Input window as numpy array (should be preprocessed)
            
        Returns:
            HOG feature vector as 1D numpy array
            
        Raises:
            ValueError: If window dimensions are invalid
        """
        # Validate input
        if window is None or window.size == 0:
            raise ValueError("Input window is empty or None")
        
        # Multi-channel mode (extract from RGB channels separately)
        if self.use_color and len(window.shape) == 3 and window.shape[2] >= 3:
            try:
                # Convert color space if needed
                if self.color_space == 'HSV':
                    color_image = cv2.cvtColor(window, cv2.COLOR_RGB2HSV)
                elif self.color_space == 'LAB':
                    color_image = cv2.cvtColor(window, cv2.COLOR_RGB2LAB)
                else:  # RGB (default)
                    color_image = window.copy()
                
                # Extract HOG from each channel
                channel_features = []
                for channel_idx in range(3):
                    channel = color_image[:, :, channel_idx]
                    channel_hog = self._extract_channel_hog(channel)
                    channel_features.append(channel_hog)
                
                # Concatenate features from all channels
                features = np.concatenate(channel_features)
                
                # Validate output
                if len(features) == 0:
                    logger.warning("Multi-channel HOG extraction returned empty feature vector")
                    return np.zeros(self.feature_vector_size)
                
                return features
                
            except Exception as e:
                logger.error(f"Error extracting multi-channel HOG features: {e}")
                # Return zero vector as fallback
                return np.zeros(self.feature_vector_size)
        
        # Grayscale mode (original behavior)
        else:
            # Convert to grayscale if needed
            if len(window.shape) == 3:
                if window.shape[2] == 3:  # RGB
                    gray = cv2.cvtColor(window, cv2.COLOR_RGB2GRAY)
                elif window.shape[2] == 4:  # RGBA
                    gray = cv2.cvtColor(window, cv2.COLOR_RGBA2GRAY)
                else:
                    gray = window[:, :, 0]  # Take first channel
            else:
                gray = window.copy()
            
            # Extract HOG from grayscale
            return self._extract_channel_hog(gray)
    
    def extract_features_batch(self, windows: np.ndarray) -> np.ndarray:
        """
        Extract HOG features from multiple windows efficiently.
        
        Args:
            windows: Array of windows with shape (n_windows, height, width, channels)
            
        Returns:
            Feature matrix with shape (n_windows, n_features)
        """
        if len(windows) == 0:
            return np.empty((0, self.feature_vector_size))
        
        features_list = []
        
        for i, window in enumerate(windows):
            try:
                features = self.extract_features(window)
                features_list.append(features)
            except Exception as e:
                logger.warning(f"Error processing window {i}: {e}")
                # Add zero vector for failed extraction
                features_list.append(np.zeros(self.feature_vector_size))
        
        return np.array(features_list)
    
    def extract_features_with_visualization(self, window: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract HOG features and visualization from a window.
        
        Args:
            window: Input window as numpy array
            
        Returns:
            Tuple of (features, hog_image) for visualization
        """
        # Convert to grayscale if needed
        if len(window.shape) == 3:
            gray = cv2.cvtColor(window, cv2.COLOR_RGB2GRAY)
        else:
            gray = window.copy()
        
        # Ensure correct data type
        if gray.dtype != np.uint8:
            gray = (gray * 255).astype(np.uint8) if gray.max() <= 1.0 else gray.astype(np.uint8)
        
        try:
            # Extract HOG with visualization
            features, hog_image = feature.hog(
                gray,
                orientations=self.orientations,
                pixels_per_cell=self.pixels_per_cell,
                cells_per_block=self.cells_per_block,
                block_norm=self.block_norm,
                transform_sqrt=self.transform_sqrt,
                visualize=True,
                feature_vector=True
            )
            
            return features, hog_image
            
        except Exception as e:
            logger.error(f"Error extracting HOG features with visualization: {e}")
            # Return zero arrays as fallback
            zero_features = np.zeros(self.feature_vector_size)
            zero_image = np.zeros_like(gray)
            return zero_features, zero_image
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get current HOG configuration for saving/loading.
        
        Returns:
            Dictionary containing HOG parameters
        """
        return {
            'orientations': self.orientations,
            'pixels_per_cell': list(self.pixels_per_cell),
            'cells_per_block': list(self.cells_per_block),
            'block_norm': self.block_norm,
            'transform_sqrt': self.transform_sqrt,
            'use_color': self.use_color,
            'color_space': self.color_space,
            'feature_vector_size': self.feature_vector_size
        }
    
    def validate_parameters(self) -> bool:
        """
        Validate HOG parameters for consistency.
        
        Returns:
            True if parameters are valid, False otherwise
        """
        try:
            # Check orientations
            if not isinstance(self.orientations, int) or self.orientations < 1:
                logger.error(f"Invalid orientations: {self.orientations}")
                return False
            
            # Check pixels_per_cell
            if (not isinstance(self.pixels_per_cell, (list, tuple)) or 
                len(self.pixels_per_cell) != 2 or
                any(not isinstance(x, int) or x < 1 for x in self.pixels_per_cell)):
                logger.error(f"Invalid pixels_per_cell: {self.pixels_per_cell}")
                return False
            
            # Check cells_per_block
            if (not isinstance(self.cells_per_block, (list, tuple)) or 
                len(self.cells_per_block) != 2 or
                any(not isinstance(x, int) or x < 1 for x in self.cells_per_block)):
                logger.error(f"Invalid cells_per_block: {self.cells_per_block}")
                return False
            
            # Check block_norm
            valid_norms = ['L1', 'L1-sqrt', 'L2', 'L2-Hys']
            if self.block_norm not in valid_norms:
                logger.error(f"Invalid block_norm: {self.block_norm}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating HOG parameters: {e}")
            return False