"""
Combined feature extractor that integrates HOG and LBP features.

This module provides a unified interface for extracting and combining
different types of features from image windows.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
import logging
from typing import Dict, Any, List, Optional, Tuple

from .hog_extractor import HOGExtractor
from .lbp_extractor import LBPExtractor
from ..utils.config import ConfigLoader

logger = logging.getLogger(__name__)


class CombinedFeatureExtractor:
    """
    Combines HOG and LBP feature extractors for comprehensive feature extraction.
    
    This class provides a unified interface for extracting different types of features
    and combining them into a single feature vector for training and inference.
    """
    
    def __init__(self, config: ConfigLoader):
        """
        Initialize combined feature extractor.
        
        Args:
            config: Configuration loader instance
        """
        self.config = config
        
        # Initialize individual extractors
        self.hog_extractor = HOGExtractor(config)
        self.lbp_extractor = LBPExtractor(config)
        
        # Feature combination configuration
        feature_config = config.get_section('features')
        self.feature_types = feature_config.get('combination', {}).get('feature_types', ['hog', 'lbp', 'combined'])
        self.normalize_before_combine = feature_config.get('combination', {}).get('normalize_before_combine', False)
        
        # Calculate feature sizes
        self.hog_size = self.hog_extractor.feature_vector_size
        self.lbp_size = self.lbp_extractor.feature_vector_size
        self.combined_size = self.hog_size + self.lbp_size
        
        # Optional normalizers for individual features
        self.hog_normalizer = None
        self.lbp_normalizer = None
        
        color_mode = "color (RGB)" if self.hog_extractor.use_color else "grayscale"
        logger.info(f"Combined extractor initialized - HOG: {self.hog_size} ({color_mode}), "
                   f"LBP: {self.lbp_size}, Combined: {self.combined_size}")
    
    def extract_hog_features(self, windows: np.ndarray) -> np.ndarray:
        """
        Extract HOG features from windows.
        
        Args:
            windows: Array of windows
            
        Returns:
            HOG feature matrix
        """
        if isinstance(windows, list):
            return self.hog_extractor.extract_features_batch(np.array(windows))
        elif len(windows.shape) == 3:  # Single window
            return self.hog_extractor.extract_features(windows).reshape(1, -1)
        else:  # Batch of windows
            return self.hog_extractor.extract_features_batch(windows)
    
    def extract_lbp_features(self, windows: np.ndarray) -> np.ndarray:
        """
        Extract LBP features from windows.
        
        Args:
            windows: Array of windows
            
        Returns:
            LBP feature matrix
        """
        if isinstance(windows, list):
            return self.lbp_extractor.extract_features_batch(np.array(windows))
        elif len(windows.shape) == 3:  # Single window
            return self.lbp_extractor.extract_features(windows).reshape(1, -1)
        else:  # Batch of windows
            return self.lbp_extractor.extract_features_batch(windows)
    
    def extract_combined_features(self, windows: np.ndarray) -> np.ndarray:
        """
        Extract and combine HOG and LBP features.
        
        Args:
            windows: Array of windows
            
        Returns:
            Combined feature matrix
        """
        # Extract individual features
        hog_features = self.extract_hog_features(windows)
        lbp_features = self.extract_lbp_features(windows)
        
        # Validate dimensions
        if hog_features.shape[0] != lbp_features.shape[0]:
            logger.error(f"Feature dimension mismatch: HOG has {hog_features.shape[0]} samples, "
                        f"LBP has {lbp_features.shape[0]} samples")
            # Use minimum number of samples
            min_samples = min(hog_features.shape[0], lbp_features.shape[0])
            hog_features = hog_features[:min_samples]
            lbp_features = lbp_features[:min_samples]
        
        # Normalize if requested
        if self.normalize_before_combine:
            if self.hog_normalizer is not None:
                hog_features = self.hog_normalizer.transform(hog_features)
            if self.lbp_normalizer is not None:
                lbp_features = self.lbp_normalizer.transform(lbp_features)
        
        # Combine features
        combined_features = np.concatenate([hog_features, lbp_features], axis=1)
        
        # Validate combined dimensions
        expected_size = self.combined_size
        actual_size = combined_features.shape[1]
        if actual_size != expected_size:
            logger.warning(f"Combined feature size mismatch: expected {expected_size}, got {actual_size}")
            # Update combined_size if this is the first extraction
            self.combined_size = actual_size
        
        return combined_features
    
    def extract_features(self, windows: np.ndarray, feature_type: str = 'combined') -> np.ndarray:
        """
        Extract features of specified type from windows.
        
        Args:
            windows: Array of windows
            feature_type: Type of features to extract ('hog', 'lbp', 'combined')
            
        Returns:
            Feature matrix of requested type
            
        Raises:
            ValueError: If feature_type is not supported
        """
        if feature_type == 'hog':
            return self.extract_hog_features(windows)
        elif feature_type == 'lbp':
            return self.extract_lbp_features(windows)
        elif feature_type == 'combined':
            return self.extract_combined_features(windows)
        else:
            raise ValueError(f"Unsupported feature type: {feature_type}")
    
    def extract_features_for_training(self, positive_windows: np.ndarray, 
                                    negative_windows: np.ndarray) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Extract all feature types for training data.
        
        Args:
            positive_windows: Array of positive training windows
            negative_windows: Array of negative training windows
            
        Returns:
            Dictionary containing feature matrices for each feature type
        """
        logger.info(f"Extracting features for training: {len(positive_windows)} positive, {len(negative_windows)} negative")
        
        # Combine windows and create labels
        all_windows = np.concatenate([positive_windows, negative_windows], axis=0)
        labels = np.concatenate([
            np.ones(len(positive_windows)),
            np.zeros(len(negative_windows))
        ])
        
        results = {}
        
        # Extract each feature type
        for feature_type in self.feature_types:
            logger.info(f"Extracting {feature_type} features...")
            
            try:
                features = self.extract_features(all_windows, feature_type)
                
                results[feature_type] = {
                    'features': features,
                    'labels': labels,
                    'positive_features': features[:len(positive_windows)],
                    'negative_features': features[len(positive_windows):],
                    'feature_size': features.shape[1] if len(features.shape) > 1 else len(features)
                }
                
                logger.info(f"{feature_type} features extracted: shape {features.shape}")
                
            except Exception as e:
                logger.error(f"Error extracting {feature_type} features: {e}")
                # Create empty results for failed extraction
                results[feature_type] = {
                    'features': np.empty((0, 1)),
                    'labels': np.empty(0),
                    'positive_features': np.empty((0, 1)),
                    'negative_features': np.empty((0, 1)),
                    'feature_size': 0,
                    'error': str(e)
                }
        
        return results
    
    def fit_feature_normalizers(self, training_features: Dict[str, np.ndarray]) -> None:
        """
        Fit feature normalizers on training data.
        
        Args:
            training_features: Dictionary containing training feature matrices
        """
        if not self.normalize_before_combine:
            return
        
        logger.info("Fitting feature normalizers...")
        
        if 'hog' in training_features:
            self.hog_normalizer = StandardScaler()
            self.hog_normalizer.fit(training_features['hog'])
            logger.info("HOG normalizer fitted")
        
        if 'lbp' in training_features:
            self.lbp_normalizer = StandardScaler()
            self.lbp_normalizer.fit(training_features['lbp'])
            logger.info("LBP normalizer fitted")
    
    def save_feature_extractors(self, output_dir: str) -> None:
        """
        Save feature extractor configurations and normalizers.
        
        Args:
            output_dir: Directory to save extractor configurations
        """
        import joblib
        from pathlib import Path
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save configurations
        config_data = {
            'hog_config': self.hog_extractor.get_config(),
            'lbp_config': self.lbp_extractor.get_config(),
            'feature_types': self.feature_types,
            'normalize_before_combine': self.normalize_before_combine,
            'feature_sizes': {
                'hog': self.hog_size,
                'lbp': self.lbp_size,
                'combined': self.combined_size
            }
        }
        
        import json
        with open(output_path / 'feature_config.json', 'w') as f:
            json.dump(config_data, f, indent=2)
        
        # Save normalizers if fitted
        if self.hog_normalizer is not None:
            joblib.dump(self.hog_normalizer, output_path / 'hog_normalizer.pkl')
        
        if self.lbp_normalizer is not None:
            joblib.dump(self.lbp_normalizer, output_path / 'lbp_normalizer.pkl')
        
        logger.info(f"Feature extractors saved to {output_path}")
    
    def load_feature_extractors(self, input_dir: str) -> bool:
        """
        Load previously saved feature extractor configurations.
        
        Args:
            input_dir: Directory containing saved extractor configurations
            
        Returns:
            True if loading successful, False otherwise
        """
        try:
            import joblib
            from pathlib import Path
            
            input_path = Path(input_dir)
            
            # Load normalizers if they exist
            hog_normalizer_path = input_path / 'hog_normalizer.pkl'
            if hog_normalizer_path.exists():
                self.hog_normalizer = joblib.load(hog_normalizer_path)
                logger.info("HOG normalizer loaded")
            
            lbp_normalizer_path = input_path / 'lbp_normalizer.pkl'
            if lbp_normalizer_path.exists():
                self.lbp_normalizer = joblib.load(lbp_normalizer_path)
                logger.info("LBP normalizer loaded")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading feature extractors: {e}")
            return False
    
    def analyze_feature_distribution(self, features: np.ndarray, feature_type: str) -> Dict[str, Any]:
        """
        Analyze the distribution of extracted features for debugging.
        
        Args:
            features: Feature matrix to analyze
            feature_type: Type of features being analyzed
            
        Returns:
            Dictionary containing distribution statistics
        """
        if len(features) == 0:
            return {'error': 'No features provided'}
        
        try:
            analysis = {
                'feature_type': feature_type,
                'shape': features.shape,
                'mean': np.mean(features, axis=0),
                'std': np.std(features, axis=0),
                'min': np.min(features, axis=0),
                'max': np.max(features, axis=0),
                'global_stats': {
                    'overall_mean': float(np.mean(features)),
                    'overall_std': float(np.std(features)),
                    'overall_min': float(np.min(features)),
                    'overall_max': float(np.max(features))
                }
            }
            
            # Check for potential issues
            if np.any(np.isnan(features)):
                analysis['has_nan'] = True
                analysis['nan_count'] = int(np.sum(np.isnan(features)))
            
            if np.any(np.isinf(features)):
                analysis['has_inf'] = True
                analysis['inf_count'] = int(np.sum(np.isinf(features)))
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing feature distribution: {e}")
            return {'error': str(e)}
    
    def get_feature_size(self, feature_type: str) -> int:
        """
        Get the size of feature vector for specified type.
        
        Args:
            feature_type: Type of features
            
        Returns:
            Feature vector size
        """
        if feature_type == 'hog':
            return self.hog_size
        elif feature_type == 'lbp':
            return self.lbp_size
        elif feature_type == 'combined':
            return self.combined_size
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")
    
    def validate_extractors(self) -> Dict[str, bool]:
        """
        Validate all feature extractors.
        
        Returns:
            Dictionary indicating validation status for each extractor
        """
        results = {
            'hog': self.hog_extractor.validate_parameters(),
            'lbp': self.lbp_extractor.validate_parameters()
        }
        
        results['combined'] = results['hog'] and results['lbp']
        
        return results