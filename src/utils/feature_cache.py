"""
Feature caching utilities for the pipeline.

Saves and loads pre-extracted features to avoid redundant computation across stages.
Features are cached as numpy arrays with associated metadata (image paths, counts).
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
import logging


logger = logging.getLogger(__name__)


class FeatureCache:
    """Handles saving and loading cached feature vectors."""
    
    CACHE_FEATURES_FILE = "features.npy"
    CACHE_COUNTS_FILE = "counts.npy"
    CACHE_PATHS_FILE = "image_paths.json"
    CACHE_METADATA_FILE = "cache_metadata.json"
    
    @staticmethod
    def save_features(
        cache_dir: Path,
        train_features: np.ndarray,
        train_counts: np.ndarray,
        train_paths: List[str],
        test_features: np.ndarray,
        test_counts: np.ndarray,
        test_paths: List[str],
        config: Dict[str, Any]
    ) -> None:
        """
        Save extracted features to cache directory.
        
        Args:
            cache_dir: Directory to save cache files
            train_features: (n_train, n_features) array of training features
            train_counts: (n_train,) array of training counts
            train_paths: List of training image paths
            test_features: (n_test, n_features) array of test features
            test_counts: (n_test,) array of test counts
            test_paths: List of test image paths
            config: Feature config dict (for reproducibility checking)
        """
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Save feature arrays
        np.save(cache_dir / "train_features.npy", train_features)
        np.save(cache_dir / "train_counts.npy", train_counts)
        np.save(cache_dir / "test_features.npy", test_features)
        np.save(cache_dir / "test_counts.npy", test_counts)
        
        # Save metadata as JSON
        metadata = {
            "train_paths": train_paths,
            "test_paths": test_paths,
            "train_size": len(train_features),
            "test_size": len(test_features),
            "feature_dim": int(train_features.shape[1]) if len(train_features) > 0 else 0,
            "feature_config": config,
        }
        
        with open(cache_dir / "cache_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Feature cache saved to {cache_dir}")
        logger.info(f"  Train: {len(train_features)} samples, {train_features.shape[1] if len(train_features) > 0 else 0} features")
        logger.info(f"  Test:  {len(test_features)} samples, {test_features.shape[1] if len(test_features) > 0 else 0} features")
    
    @staticmethod
    def load_features(cache_dir: Path) -> Tuple[
        np.ndarray, np.ndarray, List[str],
        np.ndarray, np.ndarray, List[str],
        Dict[str, Any]
    ]:
        """
        Load cached features from directory.
        
        Returns:
            (train_features, train_counts, train_paths, 
             test_features, test_counts, test_paths, config)
        """
        cache_dir = Path(cache_dir)
        
        if not cache_dir.exists():
            raise FileNotFoundError(f"Feature cache directory not found: {cache_dir}")
        
        # Load feature arrays
        train_features = np.load(cache_dir / "train_features.npy")
        train_counts = np.load(cache_dir / "train_counts.npy")
        test_features = np.load(cache_dir / "test_features.npy")
        test_counts = np.load(cache_dir / "test_counts.npy")
        
        # Load metadata
        with open(cache_dir / "cache_metadata.json", 'r') as f:
            metadata = json.load(f)
        
        train_paths = metadata.get("train_paths", [])
        test_paths = metadata.get("test_paths", [])
        config = metadata.get("feature_config", {})
        
        logger.info(f"Feature cache loaded from {cache_dir}")
        logger.info(f"  Train: {len(train_features)} samples, {train_features.shape[1]} features")
        logger.info(f"  Test:  {len(test_features)} samples, {test_features.shape[1]} features")
        
        return train_features, train_counts, train_paths, \
               test_features, test_counts, test_paths, config
    
    @staticmethod
    def cache_exists(cache_dir: Path) -> bool:
        """Check if feature cache exists and is complete."""
        cache_dir = Path(cache_dir)
        required_files = [
            "train_features.npy",
            "train_counts.npy",
            "test_features.npy",
            "test_counts.npy",
            "cache_metadata.json"
        ]
        return all((cache_dir / f).exists() for f in required_files)
