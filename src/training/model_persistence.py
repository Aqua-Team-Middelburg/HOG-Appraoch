"""
Model persistence module for saving and loading trained models.

This module handles all I/O operations for SVM models, bbox regressors,
and associated metadata, providing a clean separation between training
logic and model persistence.
"""

import joblib
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class ModelPersistence:
    """
    Handles saving and loading of trained models and metadata.
    
    Supports:
    - SVM model packages (model, scaler, feature_reducer)
    - Bounding box regressor packages
    - Metadata serialization
    - Model loading and registry
    """
    
    def __init__(self, models_dir: Path):
        """
        Initialize model persistence handler.
        
        Args:
            models_dir: Directory for saving/loading models
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Registries for loaded components
        self.scalers = {}
        self.feature_reducers = {}
        self.trained_models = {}
    
    def save_model(self, model_package: Dict[str, Any], feature_type: str) -> Path:
        """
        Save trained model package to disk.
        
        Args:
            model_package: Complete model package containing:
                - model: Trained classifier
                - scaler: Feature scaler
                - feature_reducer: Optional dimensionality reducer
                - metadata: Training info, metrics, etc.
            feature_type: Type of model being saved (hog, lbp, combined, stacking)
            
        Returns:
            Path to saved model file
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Model filename
        model_filename = f"svm_{feature_type}_model_{timestamp}.pkl"
        model_path = self.models_dir / model_filename
        
        # Save model package
        joblib.dump(model_package, model_path)
        
        # Save metadata separately for human readability
        metadata = {k: v for k, v in model_package.items() 
                   if k not in ['model', 'scaler', 'feature_reducer']}
        
        metadata_filename = f"svm_{feature_type}_metadata_{timestamp}.json"
        metadata_path = self.models_dir / metadata_filename
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Model saved: {model_filename}")
        return model_path
    
    def save_bbox_regressor(self, regressor_package: Dict[str, Any], feature_type: str) -> Path:
        """
        Save trained bbox regressor to disk.
        
        Args:
            regressor_package: Complete regressor package containing:
                - regressor: Trained BoundingBoxRegressor
                - metadata: Training info, metrics, etc.
            feature_type: Type of features used for regression
            
        Returns:
            Path to saved regressor file
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Regressor filename
        regressor_filename = f"bbox_regressor_{feature_type}_{timestamp}.pkl"
        regressor_path = self.models_dir / regressor_filename
        
        # Save regressor package
        joblib.dump(regressor_package, regressor_path)
        
        # Save metadata separately
        metadata = {k: v for k, v in regressor_package.items() 
                   if k not in ['regressor']}
        
        metadata_filename = f"bbox_regressor_{feature_type}_metadata_{timestamp}.json"
        metadata_path = self.models_dir / metadata_filename
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Bbox regressor saved: {regressor_filename}")
        return regressor_path
    
    def load_model(self, model_path: Path) -> Dict[str, Any]:
        """
        Load previously trained model from disk.
        
        Args:
            model_path: Path to saved model file
            
        Returns:
            Loaded model package with all components
        """
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load model package
        model_package = joblib.load(model_path)
        feature_type = model_package['feature_type']
        
        # Register components for reuse
        self.scalers[feature_type] = model_package['scaler']
        if 'feature_reducer' in model_package and model_package['feature_reducer'] is not None:
            self.feature_reducers[feature_type] = model_package['feature_reducer']
        
        self.trained_models[feature_type] = model_package
        
        logger.info(f"Model loaded: {feature_type} from {model_path.name}")
        return model_package
    
    def load_bbox_regressor(self, regressor_path: Path) -> Dict[str, Any]:
        """
        Load previously trained bbox regressor from disk.
        
        Args:
            regressor_path: Path to saved regressor file
            
        Returns:
            Loaded regressor package
        """
        regressor_path = Path(regressor_path)
        
        if not regressor_path.exists():
            raise FileNotFoundError(f"Regressor file not found: {regressor_path}")
        
        regressor_package = joblib.load(regressor_path)
        
        logger.info(f"Bbox regressor loaded from {regressor_path.name}")
        return regressor_package
    
    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get summary of all loaded/trained models.
        
        Returns:
            Summary dictionary with model metadata
        """
        summary = {
            'total_models': len(self.trained_models),
            'models': {}
        }
        
        for feature_type, model_package in self.trained_models.items():
            if 'error' not in model_package:
                summary['models'][feature_type] = {
                    'cv_f1_mean': model_package.get('cv_mean', 0),
                    'cv_f1_std': model_package.get('cv_std', 0),
                    'training_samples': model_package.get('training_samples', 0),
                    'feature_dimension': model_package.get('feature_dimension', 0),
                    'timestamp': model_package.get('timestamp', 'unknown')
                }
            else:
                summary['models'][feature_type] = {
                    'status': 'failed',
                    'error': model_package['error']
                }
        
        return summary
    
    def list_available_models(self, pattern: str = "svm_*_model_*.pkl") -> list[Path]:
        """
        List all available model files in the models directory.
        
        Args:
            pattern: Glob pattern for model files
            
        Returns:
            List of model file paths
        """
        return sorted(self.models_dir.glob(pattern))
    
    def list_available_regressors(self, pattern: str = "bbox_regressor_*.pkl") -> list[Path]:
        """
        List all available bbox regressor files.
        
        Args:
            pattern: Glob pattern for regressor files
            
        Returns:
            List of regressor file paths
        """
        return sorted(self.models_dir.glob(pattern))
    
    def get_scaler(self, feature_type: str) -> Optional[Any]:
        """
        Get the scaler for a specific feature type.
        
        Args:
            feature_type: Type of features (hog, lbp, combined, stacking)
            
        Returns:
            Scaler object or None if not found
        """
        return self.scalers.get(feature_type)
    
    def get_feature_reducer(self, feature_type: str) -> Optional[Any]:
        """
        Get the feature reducer for a specific feature type.
        
        Args:
            feature_type: Type of features
            
        Returns:
            Feature reducer object or None if not found
        """
        return self.feature_reducers.get(feature_type)
