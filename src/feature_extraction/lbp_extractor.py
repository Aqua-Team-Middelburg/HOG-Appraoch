"""
LBP (Local Binary Pattern) feature extractor.

This module provides LBP feature extraction with consistent parameters
matching the original training configuration.
"""

import cv2
import numpy as np
from skimage import feature
import logging
from typing import Dict, Any, Tuple, Optional

from ..utils.config import ConfigLoader

logger = logging.getLogger(__name__)


class LBPExtractor:
    """
    Extracts LBP (Local Binary Pattern) features from image windows.
    
    This class ensures consistent LBP parameter usage across training and inference,
    creating histogram-based feature vectors from local binary patterns.
    """
    
    def __init__(self, config: ConfigLoader):
        """
        Initialize LBP extractor with configuration.
        
        Args:
            config: Configuration loader instance
        """
        self.config = config
        
        # Load LBP parameters from config
        lbp_config = config.get_section('features').get('lbp', {})
        
        self.n_points = lbp_config.get('n_points', 8)
        self.radius = lbp_config.get('radius', 1)
        self.method = lbp_config.get('method', 'uniform')
        
        # Calculate expected feature vector size based on method
        self.feature_vector_size = self._calculate_feature_size()
        
        logger.info(f"LBP extractor initialized: {self.n_points} points, "
                   f"radius {self.radius}, method '{self.method}', "
                   f"expected vector size: {self.feature_vector_size}")
    
    def _calculate_feature_size(self) -> int:
        """
        Calculate the expected LBP feature vector size.
        
        This depends on the LBP method:
        - 'uniform': n_points + 2 bins (uniform patterns + non-uniform)
        - 'default'/'nri_uniform': 2^n_points bins
        
        Returns:
            Expected feature vector size
        """
        if self.method == 'uniform':
            return self.n_points + 2
        elif self.method in ['default', 'nri_uniform']:
            return 2 ** self.n_points
        else:
            logger.warning(f"Unknown LBP method '{self.method}', using default calculation")
            return self.n_points + 2
    
    def extract_features(self, window: np.ndarray) -> np.ndarray:
        """
        Extract LBP features from a single window.
        
        Args:
            window: Input window as numpy array (should be preprocessed)
            
        Returns:
            LBP histogram feature vector as 1D numpy array
            
        Raises:
            ValueError: If window dimensions are invalid
        """
        # Validate input
        if window is None or window.size == 0:
            raise ValueError("Input window is empty or None")
        
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
        
        # Ensure correct data type
        if gray.dtype != np.uint8:
            gray = (gray * 255).astype(np.uint8) if gray.max() <= 1.0 else gray.astype(np.uint8)
        
        try:
            # Compute LBP
            lbp = feature.local_binary_pattern(
                gray,
                self.n_points,
                self.radius,
                method=self.method
            )
            
            # Create histogram
            if self.method == 'uniform':
                n_bins = self.n_points + 2
                hist_range = (0, n_bins)
            else:
                n_bins = 2 ** self.n_points
                hist_range = (0, n_bins)
            
            # Compute normalized histogram
            hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=hist_range, density=True)
            
            # Handle edge case where histogram is all zeros
            if np.sum(hist) == 0:
                logger.warning("LBP histogram is all zeros, using uniform distribution")
                hist = np.ones(n_bins) / n_bins
            
            return hist
            
        except Exception as e:
            logger.error(f"Error extracting LBP features: {e}")
            # Return uniform distribution as fallback
            return np.ones(self.feature_vector_size) / self.feature_vector_size
    
    def extract_features_batch(self, windows: np.ndarray) -> np.ndarray:
        """
        Extract LBP features from multiple windows efficiently.
        
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
                # Add uniform distribution for failed extraction
                uniform_hist = np.ones(self.feature_vector_size) / self.feature_vector_size
                features_list.append(uniform_hist)
        
        return np.array(features_list)
    
    def extract_lbp_image(self, window: np.ndarray) -> np.ndarray:
        """
        Extract the LBP image itself for visualization.
        
        Args:
            window: Input window as numpy array
            
        Returns:
            LBP image as 2D numpy array
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
            # Compute LBP
            lbp = feature.local_binary_pattern(
                gray,
                self.n_points,
                self.radius,
                method=self.method
            )
            
            return lbp
            
        except Exception as e:
            logger.error(f"Error extracting LBP image: {e}")
            return np.zeros_like(gray)
    
    def extract_features_with_visualization(self, window: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract LBP features and visualization from a window.
        
        Args:
            window: Input window as numpy array
            
        Returns:
            Tuple of (features, lbp_image) for visualization
        """
        try:
            features = self.extract_features(window)
            lbp_image = self.extract_lbp_image(window)
            return features, lbp_image
        except Exception as e:
            logger.error(f"Error extracting LBP features with visualization: {e}")
            # Return fallback values
            uniform_hist = np.ones(self.feature_vector_size) / self.feature_vector_size
            zero_image = np.zeros((window.shape[0], window.shape[1]))
            return uniform_hist, zero_image
    
    def analyze_lbp_distribution(self, window: np.ndarray) -> Dict[str, Any]:
        """
        Analyze the LBP distribution for debugging and visualization.
        
        Args:
            window: Input window as numpy array
            
        Returns:
            Dictionary containing LBP analysis results
        """
        try:
            features = self.extract_features(window)
            lbp_image = self.extract_lbp_image(window)
            
            # Calculate statistics
            analysis = {
                'histogram': features,
                'histogram_sum': np.sum(features),
                'histogram_max': np.max(features),
                'histogram_min': np.min(features),
                'histogram_mean': np.mean(features),
                'histogram_std': np.std(features),
                'lbp_image_stats': {
                    'min': float(np.min(lbp_image)),
                    'max': float(np.max(lbp_image)),
                    'mean': float(np.mean(lbp_image)),
                    'std': float(np.std(lbp_image))
                },
                'unique_patterns': len(np.unique(lbp_image)),
                'feature_vector_size': len(features)
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing LBP distribution: {e}")
            return {'error': str(e)}
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get current LBP configuration for saving/loading.
        
        Returns:
            Dictionary containing LBP parameters
        """
        return {
            'n_points': self.n_points,
            'radius': self.radius,
            'method': self.method,
            'feature_vector_size': self.feature_vector_size
        }
    
    def validate_parameters(self) -> bool:
        """
        Validate LBP parameters for consistency.
        
        Returns:
            True if parameters are valid, False otherwise
        """
        try:
            # Check n_points
            if not isinstance(self.n_points, int) or self.n_points < 3:
                logger.error(f"Invalid n_points: {self.n_points} (must be int >= 3)")
                return False
            
            # Check radius
            if not isinstance(self.radius, (int, float)) or self.radius <= 0:
                logger.error(f"Invalid radius: {self.radius} (must be positive number)")
                return False
            
            # Check method
            valid_methods = ['uniform', 'default', 'nri_uniform', 'var']
            if self.method not in valid_methods:
                logger.error(f"Invalid method: {self.method} (must be one of {valid_methods})")
                return False
            
            # Check if n_points is reasonable for the method
            if self.method in ['default', 'nri_uniform'] and self.n_points > 24:
                logger.warning(f"Large n_points ({self.n_points}) may cause memory issues with method '{self.method}'")
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating LBP parameters: {e}")
            return False
    
    def get_pattern_interpretations(self) -> Dict[str, str]:
        """
        Get interpretations of LBP patterns for uniform method.
        
        Returns:
            Dictionary mapping pattern indices to their meanings
        """
        if self.method != 'uniform':
            return {}
        
        interpretations = {}
        
        # For 8-point LBP uniform patterns
        if self.n_points == 8:
            patterns = {
                0: "Flat region (all neighbors similar)",
                1: "Edge (1 transition)",
                2: "Corner (1 transition)", 
                3: "Line end (2 transitions)",
                4: "Corner (2 transitions)",
                5: "Line (2 transitions)",
                6: "Corner (2 transitions)",
                7: "Line (2 transitions)",
                8: "Corner (2 transitions)",
                9: "Non-uniform patterns"
            }
            interpretations.update(patterns)
        else:
            # General case
            for i in range(self.n_points + 2):
                if i < self.n_points:
                    interpretations[i] = f"Uniform pattern {i}"
                else:
                    interpretations[i] = "Non-uniform patterns"
        
        return interpretations