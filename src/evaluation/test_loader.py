"""
Test set loading module for model evaluation.

This module handles loading test data from various sources including
separate test directories or splits from training data.
"""

import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, Any

from ..utils.config import ConfigLoader

logger = logging.getLogger(__name__)


class TestSetLoader:
    """
    Loads and processes test set data with proper image-window mapping.
    
    Supports loading from:
    - Separate test set directory
    - Split from training data
    """
    
    def __init__(self, config: ConfigLoader):
        """
        Initialize test set loader.
        
        Args:
            config: Configuration loader instance
        """
        self.config = config
        
    def load_test_set_features_and_metadata(self) -> Dict[str, Any]:
        """
        Load test set features and create proper window-to-image mapping.
        
        Returns:
            Dictionary containing test features, labels, and metadata
        """
        logger.info("Loading test set with proper image-window mapping...")
        
        # Try to load from separate test set first
        test_set_dir = Path(self.config.get('paths.test_set_dir', 'test_set'))
        
        if test_set_dir.exists():
            return self._load_separate_test_set(test_set_dir)
        else:
            return self._create_test_split_from_training()
    
    def _load_separate_test_set(self, test_dir: Path) -> Dict[str, Any]:
        """
        Load features from separate test set directory.
        
        Args:
            test_dir: Path to test set directory
            
        Returns:
            Dictionary with test data and metadata
        """
        logger.info(f"Loading separate test set from {test_dir}")
        
        # Look for test set features
        test_features_dir = test_dir / 'features'
        if not test_features_dir.exists():
            logger.warning(f"Test features directory not found: {test_features_dir}")
            return self._create_test_split_from_training()
        
        test_data = {}
        window_metadata = {}
        
        for feature_type in ['hog', 'lbp', 'combined']:
            feature_file = test_features_dir / f'{feature_type}_features.npy'
            labels_file = test_features_dir / f'{feature_type}_labels.npy'
            metadata_file = test_features_dir / f'{feature_type}_window_metadata.json'
            
            if feature_file.exists() and labels_file.exists():
                features = np.load(feature_file)
                labels = np.load(labels_file)
                
                test_data[feature_type] = {
                    'features': features,
                    'labels': labels
                }
                
                # Load window metadata if available
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        window_metadata[feature_type] = json.load(f)
                
                logger.info(f"Loaded test {feature_type}: {features.shape}")
        
        return {
            'test_data': test_data,
            'window_metadata': window_metadata,
            'source': 'separate_test_set'
        }
    
    def _create_test_split_from_training(self) -> Dict[str, Any]:
        """
        Create test split from training data with enhanced metadata.
        
        Returns:
            Dictionary with test data and metadata
        """
        logger.info("Creating test split from training data...")
        
        features_dir = Path(self.config.get('paths.extracted_features_dir', 'temp/extracted_features'))
        test_split_ratio = self.config.get_section('evaluation').get('test_split_ratio', 0.2)
        
        test_data = {}
        window_metadata = {}
        
        for feature_type in ['hog', 'lbp', 'combined']:
            feature_file = features_dir / f'{feature_type}_features.npy'
            labels_file = features_dir / f'{feature_type}_labels.npy'
            
            if feature_file.exists() and labels_file.exists():
                features = np.load(feature_file)
                labels = np.load(labels_file)
                
                # Create stratified test split
                from sklearn.model_selection import train_test_split
                
                _, test_features, _, test_labels = train_test_split(
                    features, labels, 
                    test_size=test_split_ratio, 
                    stratify=labels,
                    random_state=42
                )
                
                test_data[feature_type] = {
                    'features': test_features,
                    'labels': test_labels
                }
                
                # Create simulated window metadata for image grouping
                window_metadata[feature_type] = self._create_simulated_metadata(len(test_features))
                
                logger.info(f"Created test {feature_type}: {test_features.shape}")
        
        return {
            'test_data': test_data,
            'window_metadata': window_metadata,
            'source': 'training_split'
        }
    
    def _create_simulated_metadata(self, n_windows: int) -> Dict[str, Any]:
        """
        Create simulated window-to-image mapping.
        
        Args:
            n_windows: Total number of windows
            
        Returns:
            Metadata dictionary with window-to-image mapping
        """
        windows_per_image = self.config.get_section('evaluation').get('simulated_windows_per_image', 20)
        n_images = max(1, n_windows // windows_per_image)
        
        metadata = {
            'window_to_image': {},
            'image_info': {},
            'windows_per_image': windows_per_image
        }
        
        for i in range(n_windows):
            image_id = f"test_image_{i // windows_per_image}"
            metadata['window_to_image'][str(i)] = {
                'image_id': image_id,
                'window_index': i % windows_per_image
            }
        
        for i in range(n_images):
            image_id = f"test_image_{i}"
            metadata['image_info'][image_id] = {
                'total_windows': min(windows_per_image, n_windows - i * windows_per_image)
            }
        
        return metadata
