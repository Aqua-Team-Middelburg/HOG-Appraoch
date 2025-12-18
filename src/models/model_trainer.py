"""
Model training for ensemble SVR count prediction.

Supports training multiple feature-specific SVR models and an ensemble meta-learner.
"""

import numpy as np
import pickle
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import logging
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge


class ModelTrainer:
    """
    Trains ensemble of feature-specific SVR regressors for count prediction.
    
    Can train separate models for:
    - HOG features
    - LBP features
    - Mask statistics features
    
    Then combines them via a meta-learner (Ridge regression).
    """
    def __init__(self, config: Dict[str, Any], silent: bool = False):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.silent = silent
        
        # Ensemble models
        self.ensemble_enabled = config.get('ensemble', {}).get('enabled', True)
        self.svr_hog = None
        self.svr_lbp = None
        self.svr_mask_stats = None
        self.ensemble_meta_learner = None
        
        # Track which models are enabled
        ensemble_cfg = config.get('ensemble', {})
        self.use_hog = ensemble_cfg.get('use_hog', False)
        self.use_lbp = ensemble_cfg.get('use_lbp', True)
        self.use_mask_stats = ensemble_cfg.get('use_mask_stats', True)
    
    def _get_svr_config(self) -> Tuple[float, float, float, float, str, str]:
        """Extract SVR hyperparameters from config."""
        bounds = self.config.get('svr_bounds', {})
        c_min = bounds.get('c_min', 0.1)
        c_max = bounds.get('c_max', 10.0)
        eps_min = bounds.get('eps_min', 0.01)
        eps_max = bounds.get('eps_max', 0.5)
        kernel_val = bounds.get('kernel', 'rbf')
        gamma_val = self.config.get('svr_gamma', bounds.get('gamma', 'scale'))
        
        return c_min, c_max, eps_min, eps_max, kernel_val, gamma_val
    
    def _resolve_gamma(self, val, bounds):
        """Normalize gamma to a value accepted by sklearn."""
        if isinstance(val, (list, tuple)):
            val = val[0] if len(val) > 0 else 'scale'
        if isinstance(val, (int, float, np.floating)):
            if bounds.get('gamma_min') is not None:
                val = max(bounds['gamma_min'], val)
            if bounds.get('gamma_max') is not None:
                val = min(bounds['gamma_max'], val)
        if val is None:
            val = 'scale'
        return val
    
    def _build_svr_pipeline(self) -> Any:
        """Build SVR pipeline with configured hyperparameters."""
        c_min, c_max, eps_min, eps_max, kernel_val, gamma_val = self._get_svr_config()
        bounds = self.config.get('svr_bounds', {})
        
        return make_pipeline(
            StandardScaler(),
            SVR(
                C=max(c_min, min(self.config.get('svr_c', 1.0), c_max)),
                kernel=kernel_val,
                gamma=self._resolve_gamma(gamma_val, bounds),
                epsilon=max(eps_min, min(self.config.get('svr_epsilon', 0.1), eps_max))
            )
        )
    
    def _compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Compute MAE, RMSE, MAPE metrics."""
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1))) * 100
        return {
            'mae': float(mae),
            'rmse': float(rmse),
            'mape': float(mape)
        }
    
    def train_ensemble_models(
        self,
        feature_vectors: Dict[str, np.ndarray],
        target_counts: np.ndarray
    ) -> Dict[str, Any]:
        """
        Train feature-specific SVR models independently.
        
        Args:
            feature_vectors: Dict mapping feature type to feature arrays (already built)
                           Keys: 'hog', 'lbp', 'mask_stats'
                           Values: numpy arrays of shape (n_samples, n_features)
            target_counts: Target counts array of shape (n_samples,)
                           
        Returns:
            Dictionary with metrics for each model
        """
        results = {}
        
        # Train HOG model if enabled
        if self.use_hog and 'hog' in feature_vectors:
            X_hog = feature_vectors['hog']
            
            if X_hog.shape[1] > 0:  # Only train if features exist
                y_log = np.log1p(target_counts)
                self.svr_hog = self._build_svr_pipeline()
                self.svr_hog.fit(X_hog, y_log)
                
                y_log_pred = self.svr_hog.predict(X_hog)
                max_log = np.log1p(np.max(target_counts)) + 2.0
                y_log_pred = np.clip(y_log_pred, a_min=None, a_max=max_log)
                y_pred = np.expm1(y_log_pred)
                y_pred = np.clip(y_pred, 0, None)
                
                results['hog'] = self._compute_metrics(target_counts, y_pred)
                results['hog']['n_samples'] = len(X_hog)
                if not self.silent:
                    self.logger.info(f"Trained HOG SVR model: MAE={results['hog']['mae']:.4f}, MAPE={results['hog']['mape']:.2f}%")
            else:
                results['hog'] = {'mae': 0, 'rmse': 0, 'mape': 0, 'n_samples': 0}
                self.logger.warning("HOG features are empty, skipping HOG model training")
        
        # Train LBP model if enabled
        if self.use_lbp and 'lbp' in feature_vectors:
            X_lbp = feature_vectors['lbp']
            
            if X_lbp.shape[1] > 0:  # Only train if features exist
                y_log = np.log1p(target_counts)
                self.svr_lbp = self._build_svr_pipeline()
                self.svr_lbp.fit(X_lbp, y_log)
                
                y_log_pred = self.svr_lbp.predict(X_lbp)
                max_log = np.log1p(np.max(target_counts)) + 2.0
                y_log_pred = np.clip(y_log_pred, a_min=None, a_max=max_log)
                y_pred = np.expm1(y_log_pred)
                y_pred = np.clip(y_pred, 0, None)
                
                results['lbp'] = self._compute_metrics(target_counts, y_pred)
                results['lbp']['n_samples'] = len(X_lbp)
                if not self.silent:
                    self.logger.info(f"Trained LBP SVR model: MAE={results['lbp']['mae']:.4f}, MAPE={results['lbp']['mape']:.2f}%")
            else:
                results['lbp'] = {'mae': 0, 'rmse': 0, 'mape': 0, 'n_samples': 0}
                self.logger.warning("LBP features are empty, skipping LBP model training")
        
        # Train mask stats model if enabled
        if self.use_mask_stats and 'mask_stats' in feature_vectors:
            X_mask = feature_vectors['mask_stats']
            
            if X_mask.shape[1] > 0:  # Only train if features exist
                y_log = np.log1p(target_counts)
                self.svr_mask_stats = self._build_svr_pipeline()
                self.svr_mask_stats.fit(X_mask, y_log)
                
                y_log_pred = self.svr_mask_stats.predict(X_mask)
                max_log = np.log1p(np.max(target_counts)) + 2.0
                y_log_pred = np.clip(y_log_pred, a_min=None, a_max=max_log)
                y_pred = np.expm1(y_log_pred)
                y_pred = np.clip(y_pred, 0, None)
                
                results['mask_stats'] = self._compute_metrics(target_counts, y_pred)
                results['mask_stats']['n_samples'] = len(X_mask)
                if not self.silent:
                    self.logger.info(f"Trained Mask Stats SVR model: MAE={results['mask_stats']['mae']:.4f}, MAPE={results['mask_stats']['mape']:.2f}%")
            else:
                results['mask_stats'] = {'mae': 0, 'rmse': 0, 'mape': 0, 'n_samples': 0}
                self.logger.warning("Mask stats features are empty, skipping mask stats model training")
        
        return results
    
    def train_ensemble_meta_learner(
        self,
        base_model_predictions: Dict[str, np.ndarray],
        y_true: np.ndarray
    ) -> Dict[str, Any]:
        """
        Train Ridge regression meta-learner on base model predictions.
        
        Args:
            base_model_predictions: Dict mapping model names to prediction arrays
            y_true: Target counts
            
        Returns:
            Dictionary with meta-learner training metrics
        """
        # Stack enabled model predictions
        enabled_models = []
        prediction_stack = []
        
        if self.use_hog and 'hog' in base_model_predictions:
            enabled_models.append('hog')
            prediction_stack.append(base_model_predictions['hog'])
        if self.use_lbp and 'lbp' in base_model_predictions:
            enabled_models.append('lbp')
            prediction_stack.append(base_model_predictions['lbp'])
        if self.use_mask_stats and 'mask_stats' in base_model_predictions:
            enabled_models.append('mask_stats')
            prediction_stack.append(base_model_predictions['mask_stats'])
        
        if not prediction_stack:
            raise ValueError("No enabled models for ensemble meta-learner")
        
        X_meta = np.column_stack(prediction_stack)
        
        # Train Ridge meta-learner
        alpha = self.config.get('ensemble', {}).get('meta_learner_alpha', 1.0)
        self.ensemble_meta_learner = Ridge(alpha=alpha)
        self.ensemble_meta_learner.fit(X_meta, y_true)
        
        # Compute metrics
        y_pred = self.ensemble_meta_learner.predict(X_meta)
        metrics = self._compute_metrics(y_true, y_pred)
        metrics['n_samples'] = len(y_true)
        metrics['enabled_models'] = enabled_models
        metrics['model_weights'] = dict(zip(enabled_models, [float(w) for w in self.ensemble_meta_learner.coef_]))
        metrics['intercept'] = float(self.ensemble_meta_learner.intercept_)
        
        if not self.silent:
            self.logger.info(f"Trained ensemble meta-learner with {len(enabled_models)} models")
            self.logger.info(f"  Model weights: {metrics['model_weights']}")
            self.logger.info(f"  Ensemble MAE={metrics['mae']:.4f}, MAPE={metrics['mape']:.2f}%")
        
        return metrics
    
    def predict_ensemble(self, feature_dict: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Get predictions from all enabled base models.
        
        Args:
            feature_dict: Dict mapping feature type to feature arrays
                         Keys: 'hog', 'lbp', 'mask_stats'
                         
        Returns:
            Dict mapping model names to their predictions
        """
        base_predictions = {}
        
        # Get HOG prediction if enabled
        if self.use_hog and 'hog' in feature_dict and self.svr_hog is not None:
            X_hog = feature_dict['hog'].reshape(1, -1) if feature_dict['hog'].ndim == 1 else feature_dict['hog']
            y_log_pred = self.svr_hog.predict(X_hog)[0]
            max_log = np.log1p(100) + 2.0  # Reasonable upper bound
            y_log_pred = np.clip(y_log_pred, a_min=None, a_max=max_log)
            y_pred = np.expm1(y_log_pred)
            base_predictions['hog'] = float(np.clip(y_pred, 0, None))
        
        # Get LBP prediction if enabled
        if self.use_lbp and 'lbp' in feature_dict and self.svr_lbp is not None:
            X_lbp = feature_dict['lbp'].reshape(1, -1) if feature_dict['lbp'].ndim == 1 else feature_dict['lbp']
            y_log_pred = self.svr_lbp.predict(X_lbp)[0]
            max_log = np.log1p(100) + 2.0  # Reasonable upper bound
            y_log_pred = np.clip(y_log_pred, a_min=None, a_max=max_log)
            y_pred = np.expm1(y_log_pred)
            base_predictions['lbp'] = float(np.clip(y_pred, 0, None))
        
        # Get Mask Stats prediction if enabled
        if self.use_mask_stats and 'mask_stats' in feature_dict and self.svr_mask_stats is not None:
            X_mask = feature_dict['mask_stats'].reshape(1, -1) if feature_dict['mask_stats'].ndim == 1 else feature_dict['mask_stats']
            y_log_pred = self.svr_mask_stats.predict(X_mask)[0]
            max_log = np.log1p(100) + 2.0  # Reasonable upper bound
            y_log_pred = np.clip(y_log_pred, a_min=None, a_max=max_log)
            y_pred = np.expm1(y_log_pred)
            base_predictions['mask_stats'] = float(np.clip(y_pred, 0, None))
        
        return base_predictions
    
    def predict_ensemble_meta(self, base_predictions: Dict[str, float]) -> float:
        """
        Combine base model predictions using the meta-learner.
        
        Args:
            base_predictions: Dict mapping model names to predictions
            
        Returns:
            Ensemble prediction
        """
        if self.ensemble_meta_learner is None:
            raise ValueError("Ensemble meta-learner not trained")
        
        # Stack predictions in order
        pred_stack = []
        if self.use_hog and 'hog' in base_predictions:
            pred_stack.append(base_predictions['hog'])
        if self.use_lbp and 'lbp' in base_predictions:
            pred_stack.append(base_predictions['lbp'])
        if self.use_mask_stats and 'mask_stats' in base_predictions:
            pred_stack.append(base_predictions['mask_stats'])
        
        X_meta = np.array(pred_stack).reshape(1, -1)
        ensemble_pred = self.ensemble_meta_learner.predict(X_meta)[0]
        
        return float(ensemble_pred)

    def save_model(self, output_dir: str) -> None:
        """Save legacy single model."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        svr_path = output_path / "svr_regressor.pkl"
        with open(svr_path, 'wb') as f:
            pickle.dump(self.svr_regressor, f)
        if not self.silent:
            self.logger.info(f"Saved SVR regressor to {svr_path}")
    
    def save_ensemble_models(self, output_dir: str) -> None:
        """Save all ensemble models."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if self.use_hog and self.svr_hog is not None:
            hog_path = output_path / "svr_hog.pkl"
            with open(hog_path, 'wb') as f:
                pickle.dump(self.svr_hog, f)
            if not self.silent:
                self.logger.info(f"Saved HOG SVR model to {hog_path}")
        
        if self.use_lbp and self.svr_lbp is not None:
            lbp_path = output_path / "svr_lbp.pkl"
            with open(lbp_path, 'wb') as f:
                pickle.dump(self.svr_lbp, f)
            if not self.silent:
                self.logger.info(f"Saved LBP SVR model to {lbp_path}")
        
        if self.use_mask_stats and self.svr_mask_stats is not None:
            mask_path = output_path / "svr_mask_stats.pkl"
            with open(mask_path, 'wb') as f:
                pickle.dump(self.svr_mask_stats, f)
            if not self.silent:
                self.logger.info(f"Saved Mask Stats SVR model to {mask_path}")
        
        meta_path = output_path / "ensemble_meta_learner.pkl"
        with open(meta_path, 'wb') as f:
            pickle.dump(self.ensemble_meta_learner, f)
        if not self.silent:
            self.logger.info(f"Saved ensemble meta-learner to {meta_path}")

    def load_ensemble_models(self, model_dir: str) -> None:
        """Load all ensemble models."""
        model_path = Path(model_dir)
        
        if self.use_hog:
            hog_path = model_path / "svr_hog.pkl"
            if hog_path.exists():
                with open(hog_path, 'rb') as f:
                    self.svr_hog = pickle.load(f)
                if not self.silent:
                    self.logger.info(f"Loaded HOG SVR model from {hog_path}")
        
        if self.use_lbp:
            lbp_path = model_path / "svr_lbp.pkl"
            if lbp_path.exists():
                with open(lbp_path, 'rb') as f:
                    self.svr_lbp = pickle.load(f)
                if not self.silent:
                    self.logger.info(f"Loaded LBP SVR model from {lbp_path}")
        
        if self.use_mask_stats:
            mask_path = model_path / "svr_mask_stats.pkl"
            if mask_path.exists():
                with open(mask_path, 'rb') as f:
                    self.svr_mask_stats = pickle.load(f)
                if not self.silent:
                    self.logger.info(f"Loaded Mask Stats SVR model from {mask_path}")
        
        meta_path = model_path / "ensemble_meta_learner.pkl"
        if meta_path.exists():
            with open(meta_path, 'rb') as f:
                self.ensemble_meta_learner = pickle.load(f)
            if not self.silent:
                self.logger.info(f"Loaded ensemble meta-learner from {meta_path}")
