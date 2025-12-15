"""
Model training for single-stage SVM multi-class count prediction.
"""

import numpy as np
import pickle
from pathlib import Path
from typing import List, Tuple, Dict, Any
import logging
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

class ModelTrainer:
    """
    Trains a single SVM classifier for count prediction using full-image features.
    """
    def __init__(self, config: Dict[str, Any], silent: bool = False):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.svr_regressor = None
        self.silent = silent

    def train_count_regressor(self, training_data: List[Tuple[np.ndarray, int]]) -> Dict[str, Any]:
        """
        Train SVR regressor for log(count+1) prediction.
        Args:
            training_data: List of (features, count) pairs
        Returns:
            Dictionary with training metrics (on true counts)
        """
        X = np.array([f for f, c in training_data])
        y_true = np.array([c for f, c in training_data])
        y_log = np.log1p(y_true)
        bounds = self.config.get('svr_bounds', {})
        c_min = bounds.get('c_min', 0.1)
        c_max = bounds.get('c_max', 10.0)
        eps_min = bounds.get('eps_min', 0.01)
        eps_max = bounds.get('eps_max', 0.5)
        kernel_val = bounds.get('kernel', 'rbf')
        gamma_val = self.config.get('svr_gamma', bounds.get('gamma', 'scale'))

        def _resolve_gamma(val):
            """
            Normalize gamma to a value accepted by sklearn:
            - If a list/tuple is provided (from config), take the first choice.
            - If numeric, clamp to optional bounds.
            - Default to 'scale' when missing.
            """
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
        # Suppress all info-level logging during training/tuning
        self.svr_regressor = make_pipeline(
            StandardScaler(),
            SVR(
                C=max(c_min, min(self.config.get('svr_c', 1.0), c_max)),
                kernel=kernel_val,
                gamma=_resolve_gamma(gamma_val),
                epsilon=max(eps_min, min(self.config.get('svr_epsilon', 0.1), eps_max))
            )
        )
        self.svr_regressor.fit(X, y_log)
        y_log_pred = self.svr_regressor.predict(X)
        # Prevent overflow in expm1 by capping predictions slightly above the max observed log target
        max_log = np.log1p(np.max(y_true)) + 2.0
        y_log_pred = np.clip(y_log_pred, a_min=None, a_max=max_log)
        y_pred = np.expm1(y_log_pred)
        y_pred = np.clip(y_pred, 0, None)
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1))) * 100
        # Suppress all info-level logging during training/tuning
        return {
            'mae': float(mae),
            'rmse': float(rmse),
            'mape': float(mape),
            'n_samples': len(X)
        }

    def predict_count(self, features: np.ndarray) -> float:
        """
        Predict nurdle count for a single image's features (regression, log transform).
        """
        if self.svr_regressor is None:
            raise ValueError("SVR regressor not trained")
        features_2d = features.reshape(1, -1)
        log_pred = float(self.svr_regressor.predict(features_2d)[0])
        log_pred = np.clip(log_pred, a_min=-5.0, a_max=np.log1p(np.max([0, 1000])) + 2.0)
        count_pred = np.expm1(log_pred)
        count_pred = np.clip(count_pred, 0, None)
        return float(count_pred)

    def save_model(self, output_dir: str) -> None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        svr_path = output_path / "svr_regressor.pkl"
        with open(svr_path, 'wb') as f:
            pickle.dump(self.svr_regressor, f)
        self.logger.info(f"Saved SVR regressor to {svr_path}")

    def load_model(self, model_dir: str) -> None:
        model_path = Path(model_dir)
        svr_path = model_path / "svr_regressor.pkl"
        if svr_path.exists():
            with open(svr_path, 'rb') as f:
                self.svr_regressor = pickle.load(f)
            self.logger.info(f"Loaded SVR regressor from {svr_path}")
    
    # All SVR, window, and tuning logic removed. Only single-stage SVM classifier logic remains.
