"""
SVM trainer module for all feature types (HOG, LBP, Combined, Stacking).

This module implements comprehensive SVM training with hyperparameter tuning,
cross-validation, and model persistence for the nurdle detection pipeline.

Refactored for maintainability - delegates to specialized trainers.
"""

import numpy as np
import joblib
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional, Union
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

from ..utils.config import ConfigLoader
from .bbox_regressor import BoundingBoxRegressor
from ..feature_extraction.dimensionality_reduction import FeatureReducer
from .hard_negative_miner import HardNegativeMiner
from .ensemble_trainer import EnsembleTrainer
from .incremental_trainer import IncrementalTrainer
from .hyperparameter_optimizer import HyperparameterOptimizer
from .model_persistence import ModelPersistence

logger = logging.getLogger(__name__)


class SVMTrainer:
    """
    Comprehensive SVM trainer supporting multiple feature types and ensemble methods.
    
    Supports:
    - HOG-only SVM
    - LBP-only SVM  
    - Combined HOG+LBP SVM
    - Stacking ensemble of all models
    """
    
    def __init__(self, config: ConfigLoader):
        """
        Initialize SVM trainer.
        
        Args:
            config: Configuration loader instance
        """
        self.config = config
        
        # Training configuration
        training_config = config.get_section('training')
        self.svm_config = training_config.get('svm', {})
        self.cv_config = training_config.get('cross_validation', {})
        self.tuning_config = training_config.get('hyperparameter_tuning', {})
        
        # Paths
        self.models_dir = Path(config.get('paths.output_dir')) / config.get('paths.models_dir')
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Random state for reproducibility
        self.random_state = training_config.get('random_state', 42)
        
        # Initialize model persistence handler
        self.persistence = ModelPersistence(self.models_dir)
        
        # Delegate to persistence for scalers and reducers
        self.scalers = self.persistence.scalers
        self.feature_reducers = self.persistence.feature_reducers
        self.trained_models = self.persistence.trained_models
        
        # Feature reduction configuration
        self.use_dimensionality_reduction = config.get_section('features').get(
            'dimensionality_reduction', {}
        ).get('enabled', False)
        
        # Initialize specialized trainers
        self.hard_negative_miner = HardNegativeMiner(
            training_config.get('hard_negative_mining', {}),
            self.random_state
        )
        
        self.ensemble_trainer = EnsembleTrainer(
            training_config.get('ensemble', {}),
            self.random_state
        )
        
        performance_config = config.get_section('system').get('performance', {})
        self.incremental_trainer = IncrementalTrainer(
            performance_config.get('incremental_training', {}),
            self.random_state
        )
        
        self.hyperparameter_optimizer = HyperparameterOptimizer(
            self.tuning_config,
            self.random_state
        )
        
        logger.info("SVM trainer initialized")
    
    def create_svm_model(self, **kwargs) -> Union[LinearSVC, SVC]:
        """
        Create SVM model with configured parameters.
        
        Args:
            **kwargs: Override parameters
            
        Returns:
            Configured SVM model
        """
        # Merge default config with overrides
        params = {**self.svm_config, **kwargs}
        
        # Use LinearSVC for linear kernel (more efficient)
        if params.get('kernel', 'linear') == 'linear':
            model = LinearSVC(
                C=params.get('C', 1.0),
                max_iter=params.get('max_iter', 1000),
                random_state=self.random_state,
                class_weight=params.get('class_weight', None)
            )
        else:
            model = SVC(
                kernel=params.get('kernel', 'rbf'),
                C=params.get('C', 1.0),
                gamma=params.get('gamma', 'scale'),
                max_iter=params.get('max_iter', 1000),
                random_state=self.random_state,
                class_weight=params.get('class_weight', None),
                probability=True  # Enable probability estimates for stacking
            )
        
        return model
    
    def optimize_hyperparameters(self, 
                                features: np.ndarray, 
                                labels: np.ndarray,
                                feature_type: str) -> Dict[str, Any]:
        """
        Optimize SVM hyperparameters using delegated optimizer.
        
        Args:
            features: Training feature matrix
            labels: Training labels
            feature_type: Type of features being optimized
            
        Returns:
            Dictionary containing best parameters and optimization results
        """
        return self.hyperparameter_optimizer.optimize(
            features, labels, feature_type,
            self.create_svm_model, self.svm_config
        )
        
        # Get parameter grid from config
        param_grid = self.tuning_config.get('param_grid', {
            'C': [0.1, 1.0, 10.0],
            'kernel': ['linear', 'rbf']
        })
        
        # Create base model
        model = self.create_svm_model()
        
        # Grid search
        cv_folds = self.tuning_config.get('cv_folds', 3)
        grid_search = GridSearchCV(
            model,
            param_grid,
            cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state),
            scoring='f1',
            n_jobs=-1
        )
        
        grid_search.fit(features, labels)
        
        results = {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'method': 'gridsearch',
            'n_combinations': len(grid_search.cv_results_['params'])
        }
        
        logger.info(f"GridSearch optimization completed: {results['best_score']:.4f} F1 score")
        return results
    
    def train_incremental_model(self,
                               image_list: List[Path],
                               feature_extractor,
                               feature_type: str,
                               dataset_builder=None) -> Dict[str, Any]:
        """
        Train model incrementally using generator for memory efficiency.
        
        This method processes images one by one, extracting windows and features
        on-the-fly without storing everything in memory.
        
        Args:
            image_list: List of image paths to process
            feature_extractor: Feature extractor instance
            feature_type: Type of features being trained
            dataset_builder: TrainingDatasetBuilder instance
            
        Returns:
            Dictionary containing trained model and training results
        """
        if not self.incremental_trainer.enabled:
            logger.info("Incremental training not enabled, falling back to standard training")
            return {}
        
        if dataset_builder is None:
            logger.error("Dataset builder is required for incremental training")
            return {}
        
        logger.info(f"Training {feature_type} model incrementally from {len(image_list)} images")
        
        # Get batch size from config
        batch_size = self.incremental_trainer.partial_fit_batch_size
        
        # Create generator
        feature_generator = dataset_builder.generate_training_batches(
            image_list, feature_extractor, batch_size
        )
        
        # Train using generator
        return self.incremental_trainer.train_from_generator(
            feature_generator, feature_type, total_samples=None
        )
    
    def train_single_model(self,
                          features: np.ndarray,
                          labels: np.ndarray,
                          feature_type: str) -> Dict[str, Any]:
        """
        Train a single SVM model for specified feature type.
        
        Supports optional hard negative mining for improved performance.
        
        Args:
            features: Training feature matrix
            labels: Training labels  
            feature_type: Type of features ('hog', 'lbp', 'combined')
            
        Returns:
            Dictionary containing trained model and training results
        """
        logger.info(f"Training {feature_type} SVM model")
        
        # Check if incremental training is enabled
        if self.incremental_trainer.enabled:
            logger.info("Using incremental training mode")
            return self.incremental_trainer.train_incremental(features, labels, feature_type)
        
        # Separate positive and negative samples for hard negative mining
        positive_mask = labels == 1
        negative_mask = labels == 0
        
        positive_features = features[positive_mask]
        negative_features = features[negative_mask]
        positive_labels = labels[positive_mask]
        negative_labels = labels[negative_mask]
        
        logger.info(f"Initial dataset: {len(positive_features)} positives, {len(negative_features)} negatives")
        
        # Apply hard negative mining using delegated miner
        final_features, final_labels = self.hard_negative_miner.mine_hard_negatives(
            positive_features, negative_features,
            positive_labels, negative_labels,
            self.create_svm_model  # Pass model factory
        )
        
        # Scale features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(final_features)
        self.scalers[feature_type] = scaler
        
        # Apply dimensionality reduction if enabled
        feature_reducer = None
        if self.use_dimensionality_reduction:
            logger.info(f"Applying PCA dimensionality reduction to {feature_type} features")
            feature_reducer = FeatureReducer(self.config)
            scaled_features = feature_reducer.fit_transform(scaled_features, feature_type)
            self.feature_reducers[feature_type] = feature_reducer
            
            # Log reduction summary
            summary = feature_reducer.get_summary()
            logger.info(f"PCA: {summary['original_dim']}D → {summary['reduced_dim']}D "
                       f"({summary['reduction_percentage']:.1f}% reduction, "
                       f"{summary['total_variance_explained']*100:.1f}% variance retained)")
        
        # Optimize hyperparameters
        optimization_results = self.optimize_hyperparameters(scaled_features, final_labels, feature_type)
        
        # Train final model with best parameters
        best_params = optimization_results['best_params']
        model = self.create_svm_model(**best_params)
        model.fit(scaled_features, final_labels)
        
        # Cross-validation evaluation
        cv_folds = self.cv_config.get('folds', 5)
        cv_scores = cross_val_score(
            model, scaled_features, final_labels,
            cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state),
            scoring='f1'
        )
        
        # Additional metrics
        predictions = model.predict(scaled_features)
        
        try:
            # For probability-capable models (only SVC with probability=True has predict_proba)
            if hasattr(model, 'decision_function'):
                decision_scores = model.decision_function(scaled_features)
                auc_score = roc_auc_score(final_labels, decision_scores)
            else:
                auc_score = None
        except Exception:
            auc_score = None
        
        # Create model package
        model_package = {
            'model': model,
            'scaler': scaler,
            'feature_reducer': feature_reducer,  # May be None if not used
            'feature_type': feature_type,
            'best_params': best_params,
            'optimization_results': optimization_results,
            'cv_scores': cv_scores,
            'cv_mean': np.mean(cv_scores),
            'cv_std': np.std(cv_scores),
            'auc_score': auc_score,
            'training_samples': len(final_features),
            'original_samples': len(features),
            'feature_dimension': features.shape[1],
            'reduced_dimension': scaled_features.shape[1] if feature_reducer else features.shape[1],
            'dimensionality_reduction_enabled': feature_reducer is not None,
            'class_distribution': np.bincount(final_labels.astype(int)),
            'hard_negative_mining_enabled': self.hard_negative_miner.enabled,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save model
        self._save_model(model_package, feature_type)
        
        # Store in trained models
        self.trained_models[feature_type] = model_package
        
        logger.info(f"{feature_type} model trained - CV F1: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
        
        return model_package
    
    def train_bbox_regressor(self,
                            positive_features: np.ndarray,
                            positive_windows: List[Tuple[int, int, int, int]],
                            ground_truth_boxes: List[Tuple[int, int, int, int]],
                            feature_type: str) -> Dict[str, Any]:
        """
        Train bounding box regressor to refine detection windows.
        
        Trains a regression model to predict offset and scale adjustments
        that transform sliding window detections into tight-fitting bounding boxes.
        
        Args:
            positive_features: Feature vectors for positive windows (N × D)
            positive_windows: Window coordinates as (x, y, w, h) tuples (N,)
            ground_truth_boxes: Ground truth bbox coordinates as (x, y, w, h) tuples (N,)
            feature_type: Type of features ('hog', 'lbp', 'combined')
            
        Returns:
            Dictionary containing trained regressor and training metrics
        """
        logger.info(f"Training bbox regressor for {feature_type} features")
        
        # Get bbox regression config
        bbox_config = self.config.get_section('training').get('bounding_box_regression', {})
        
        if not bbox_config.get('enabled', False):
            logger.info(f"Bbox regression disabled for {feature_type}")
            return {}
        
        # Validate inputs
        n_samples = len(positive_features)
        if n_samples != len(positive_windows) or n_samples != len(ground_truth_boxes):
            raise ValueError(
                f"Mismatched sample counts: features={n_samples}, "
                f"windows={len(positive_windows)}, gt_boxes={len(ground_truth_boxes)}"
            )
        
        if n_samples < 10:
            logger.warning(f"Only {n_samples} positive samples - bbox regressor may underfit")
        
        # Create regressor with config
        regressor = BoundingBoxRegressor(self.config)
        
        # Train regressor
        logger.info(f"Training on {n_samples} positive samples with GT boxes")
        metrics = regressor.train(positive_features, positive_windows, ground_truth_boxes)
        
        # Log training results
        logger.info(f"Bbox regressor trained - MSE: {metrics['mse']:.6f}, MAE: {metrics['mae']:.6f}")
        logger.info(f"Delta ranges - x: [{metrics['delta_x_min']:.3f}, {metrics['delta_x_max']:.3f}], "
                   f"y: [{metrics['delta_y_min']:.3f}, {metrics['delta_y_max']:.3f}], "
                   f"w: [{metrics['delta_w_min']:.3f}, {metrics['delta_w_max']:.3f}], "
                   f"h: [{metrics['delta_h_min']:.3f}, {metrics['delta_h_max']:.3f}]")
        
        # Create regressor package
        regressor_package = {
            'regressor': regressor,
            'feature_type': feature_type,
            'regressor_type': bbox_config.get('regressor_type', 'svr'),
            'training_samples': n_samples,
            'feature_dimension': positive_features.shape[1],
            'metrics': metrics,
            'config': bbox_config,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save regressor
        self._save_bbox_regressor(regressor_package, feature_type)
        
        logger.info(f"{feature_type} bbox regressor training complete")
        
        return regressor_package
    
    def train_stacking_ensemble(self,
                               feature_data: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, Any]:
        """
        Train stacking ensemble using base models.
        
        Args:
            feature_data: Dictionary containing features for each type
            
        Returns:
            Dictionary containing stacking model and results
        """
        # Delegate to ensemble trainer
        stacking_package = self.ensemble_trainer.train_stacking_ensemble(
            feature_data,
            self.trained_models,
            self.scalers,
            self.cv_config.get('folds', 5)
        )
        
        if stacking_package:
            # Save stacking model
            self._save_model(stacking_package, 'stacking')
            
            # Store in trained models
            self.trained_models['stacking'] = stacking_package
        
        return stacking_package
    
    def train_all_models(self, feature_data: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, Dict[str, Any]]:
        """
        Train all model types (HOG, LBP, Combined, Stacking).
        
        Args:
            feature_data: Dictionary containing features and labels for each type
            
        Returns:
            Dictionary containing all trained models
        """
        logger.info("Starting training for all model types")
        
        results = {}
        
        # Train individual models
        for feature_type in ['hog', 'lbp', 'combined']:
            if feature_type in feature_data:
                try:
                    model_result = self.train_single_model(
                        feature_data[feature_type]['features'],
                        feature_data[feature_type]['labels'],
                        feature_type
                    )
                    results[feature_type] = model_result
                    logger.info(f"✅ {feature_type} model training completed")
                except Exception as e:
                    logger.error(f"❌ {feature_type} model training failed: {e}")
                    results[feature_type] = {'error': str(e)}
        
        # Train stacking ensemble
        if len(results) >= 2:  # Need at least 2 base models
            try:
                stacking_result = self.train_stacking_ensemble(feature_data)
                if stacking_result:
                    results['stacking'] = stacking_result
                    logger.info("✅ Stacking ensemble training completed")
            except Exception as e:
                logger.error(f"❌ Stacking ensemble training failed: {e}")
                results['stacking'] = {'error': str(e)}
        
        logger.info(f"Training completed: {len(results)} models trained")
        return results
    
    def _save_model(self, model_package: Dict[str, Any], feature_type: str) -> None:
        """
        Save trained model package to disk.
        
        Args:
            model_package: Complete model package
            feature_type: Type of model being saved
        """
        self.persistence.save_model(model_package, feature_type)
    
    def _save_bbox_regressor(self, regressor_package: Dict[str, Any], feature_type: str) -> None:
        """
        Save trained bbox regressor to disk.
        
        Args:
            regressor_package: Complete regressor package
            feature_type: Type of features used for regression
        """
        self.persistence.save_bbox_regressor(regressor_package, feature_type)
    
    def load_model(self, model_path: str) -> Dict[str, Any]:
        """
        Load previously trained model.
        
        Args:
            model_path: Path to saved model file
            
        Returns:
            Loaded model package
        """
        return self.persistence.load_model(Path(model_path))
    
    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get summary of all trained models.
        
        Returns:
            Summary dictionary
        """
        return self.persistence.get_model_summary()