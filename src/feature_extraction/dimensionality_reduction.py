"""
Dimensionality reduction module for feature compression.

This module implements PCA-based feature reduction to decrease memory usage
and training time while retaining >95% of information content.
"""

import numpy as np
import logging
from typing import Dict, Any, Optional, Union
from sklearn.decomposition import PCA
import joblib
from pathlib import Path

from ..utils.config import ConfigLoader

logger = logging.getLogger(__name__)


class FeatureReducer:
    """
    Reduces feature dimensionality using Principal Component Analysis (PCA).
    
    Compresses high-dimensional feature vectors while preserving variance,
    enabling faster training and lower memory usage with minimal accuracy loss.
    """
    
    def __init__(self, config: ConfigLoader):
        """
        Initialize feature reducer.
        
        Args:
            config: Configuration loader instance
        """
        self.config = config
        
        # Load configuration
        reduction_config = config.get_section('features').get('dimensionality_reduction', {})
        self.enabled = reduction_config.get('enabled', False)
        self.method = reduction_config.get('method', 'pca')
        
        # PCA configuration
        pca_config = reduction_config.get('pca', {})
        self.n_components = pca_config.get('n_components', 'auto')
        self.variance_threshold = pca_config.get('variance_threshold', 0.95)
        self.whiten = pca_config.get('whiten', False)
        
        # Target dimensions (manual specification)
        target_dims = reduction_config.get('target_dimensions', {})
        self.target_dim_hog = target_dims.get('hog', None)
        self.target_dim_lbp = target_dims.get('lbp', None)
        self.target_dim_combined = target_dims.get('combined', None)
        
        # Fitted components
        self.pca = None
        self.original_dim = None
        self.reduced_dim = None
        self.explained_variance_ratio = None
        
        if self.enabled:
            logger.info(f"Feature reduction initialized: method={self.method}, "
                       f"variance_threshold={self.variance_threshold}")
    
    def fit(self, features: np.ndarray, feature_type: str = 'combined') -> 'FeatureReducer':
        """
        Fit PCA reducer on training features.
        
        Args:
            features: Training feature matrix (N × D)
            feature_type: Type of features ('hog', 'lbp', 'combined')
            
        Returns:
            Self for method chaining
        """
        if not self.enabled:
            logger.info("Dimensionality reduction disabled - skipping fit")
            return self
        
        logger.info(f"Fitting PCA on {features.shape[0]} samples with {features.shape[1]} dimensions")
        
        self.original_dim = features.shape[1]
        
        # Determine n_components
        n_components = self._get_n_components(feature_type)
        
        # Fit PCA
        if isinstance(n_components, float):
            # Variance-based selection
            self.pca = PCA(n_components=n_components, whiten=self.whiten, random_state=42)
        elif n_components == 'auto':
            # Auto-select based on variance threshold
            self.pca = PCA(n_components=self.variance_threshold, whiten=self.whiten, random_state=42)
        else:
            # Fixed number of components
            self.pca = PCA(n_components=n_components, whiten=self.whiten, random_state=42)
        
        self.pca.fit(features)
        
        self.reduced_dim = self.pca.n_components_
        self.explained_variance_ratio = self.pca.explained_variance_ratio_
        
        total_variance = np.sum(self.explained_variance_ratio)
        reduction_pct = (1 - self.reduced_dim / self.original_dim) * 100
        
        logger.info(f"PCA fitted: {self.original_dim}D → {self.reduced_dim}D "
                   f"({reduction_pct:.1f}% reduction)")
        logger.info(f"Explained variance: {total_variance*100:.2f}%")
        logger.info(f"Top 5 components variance: "
                   f"{np.sum(self.explained_variance_ratio[:5])*100:.2f}%")
        
        return self
    
    def transform(self, features: np.ndarray) -> np.ndarray:
        """
        Transform features using fitted PCA.
        
        Args:
            features: Feature matrix to transform (N × D_original)
            
        Returns:
            Reduced feature matrix (N × D_reduced)
        """
        if not self.enabled or self.pca is None:
            logger.debug("PCA not fitted or disabled - returning original features")
            return features
        
        if features.shape[1] != self.original_dim:
            logger.warning(f"Feature dimension mismatch: expected {self.original_dim}, "
                          f"got {features.shape[1]}. Skipping reduction.")
            return features
        
        reduced_features = self.pca.transform(features)
        
        logger.debug(f"Transformed {features.shape[0]} samples: "
                    f"{features.shape[1]}D → {reduced_features.shape[1]}D")
        
        return reduced_features
    
    def fit_transform(self, features: np.ndarray, feature_type: str = 'combined') -> np.ndarray:
        """
        Fit PCA and transform features in one step.
        
        Args:
            features: Training feature matrix (N × D)
            feature_type: Type of features for target dimension selection
            
        Returns:
            Reduced feature matrix (N × D_reduced)
        """
        self.fit(features, feature_type)
        return self.transform(features)
    
    def inverse_transform(self, reduced_features: np.ndarray) -> np.ndarray:
        """
        Reconstruct original features from reduced representation.
        
        Useful for visualization and analysis of information loss.
        
        Args:
            reduced_features: PCA-transformed features (N × D_reduced)
            
        Returns:
            Reconstructed features (N × D_original)
        """
        if not self.enabled or self.pca is None:
            return reduced_features
        
        return self.pca.inverse_transform(reduced_features)
    
    def _get_n_components(self, feature_type: str) -> Union[int, float, str]:
        """
        Determine number of components based on feature type and config.
        
        Args:
            feature_type: Type of features ('hog', 'lbp', 'combined')
            
        Returns:
            Number of components (int), variance ratio (float), or 'auto'
        """
        # Check if manual target specified
        if feature_type == 'hog' and self.target_dim_hog is not None:
            return self.target_dim_hog
        elif feature_type == 'lbp' and self.target_dim_lbp is not None:
            return self.target_dim_lbp
        elif feature_type == 'combined' and self.target_dim_combined is not None:
            return self.target_dim_combined
        
        # Otherwise use n_components from config
        return self.n_components
    
    def get_variance_explained(self) -> Optional[np.ndarray]:
        """
        Get explained variance ratio for each component.
        
        Returns:
            Array of variance ratios, or None if not fitted
        """
        if self.pca is None:
            return None
        return self.explained_variance_ratio
    
    def get_cumulative_variance(self) -> Optional[np.ndarray]:
        """
        Get cumulative explained variance.
        
        Returns:
            Array of cumulative variance, or None if not fitted
        """
        if self.explained_variance_ratio is None:
            return None
        return np.cumsum(self.explained_variance_ratio)
    
    def get_components_for_variance(self, target_variance: float = 0.95) -> Optional[int]:
        """
        Calculate how many components needed to reach target variance.
        
        Args:
            target_variance: Target explained variance (0.0-1.0)
            
        Returns:
            Number of components needed, or None if not fitted
        """
        cumvar = self.get_cumulative_variance()
        if cumvar is None:
            return None
        
        n_components = np.argmax(cumvar >= target_variance) + 1
        return int(n_components)
    
    def save(self, filepath: Union[str, Path]) -> None:
        """
        Save fitted PCA reducer to disk.
        
        Args:
            filepath: Path to save reducer
        """
        if self.pca is None:
            logger.warning("PCA not fitted - nothing to save")
            return
        
        package = {
            'pca': self.pca,
            'original_dim': self.original_dim,
            'reduced_dim': self.reduced_dim,
            'explained_variance_ratio': self.explained_variance_ratio,
            'config': {
                'n_components': self.n_components,
                'variance_threshold': self.variance_threshold,
                'whiten': self.whiten
            }
        }
        
        joblib.dump(package, filepath)
        logger.info(f"Feature reducer saved to {filepath}")
    
    def load(self, filepath: Union[str, Path]) -> 'FeatureReducer':
        """
        Load fitted PCA reducer from disk.
        
        Args:
            filepath: Path to saved reducer
            
        Returns:
            Self for method chaining
        """
        package = joblib.load(filepath)
        
        self.pca = package['pca']
        self.original_dim = package['original_dim']
        self.reduced_dim = package['reduced_dim']
        self.explained_variance_ratio = package['explained_variance_ratio']
        
        logger.info(f"Feature reducer loaded: {self.original_dim}D → {self.reduced_dim}D")
        
        return self
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of reducer state and performance.
        
        Returns:
            Summary dictionary
        """
        if self.pca is None:
            return {
                'fitted': False,
                'enabled': self.enabled
            }
        
        return {
            'fitted': True,
            'enabled': self.enabled,
            'method': self.method,
            'original_dim': self.original_dim,
            'reduced_dim': self.reduced_dim,
            'reduction_percentage': (1 - self.reduced_dim / self.original_dim) * 100,
            'total_variance_explained': float(np.sum(self.explained_variance_ratio)),
            'n_components_95pct': self.get_components_for_variance(0.95),
            'n_components_99pct': self.get_components_for_variance(0.99),
            'whiten': self.whiten
        }
