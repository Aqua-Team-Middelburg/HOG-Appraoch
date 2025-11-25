"""
Model training for single-stage SVM multi-class count prediction.
"""

import numpy as np
import pickle
from pathlib import Path
from typing import List, Tuple, Dict, Any
import logging
from sklearn.svm import SVR



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
        # Use float32 to keep the feature matrix light during tuning
        X = np.asarray([f for f, c in training_data], dtype=np.float32)
        y_true = np.asarray([c for f, c in training_data], dtype=np.float32)
        y_log = np.log1p(y_true)
        # Suppress all info-level logging during training/tuning
        self.svr_regressor = SVR(
            C=self.config.get('svr_c', 1.0),
            kernel=self.config.get('svr_kernel', 'rbf'),
            gamma=self.config.get('svr_gamma', 'scale'),
            epsilon=self.config.get('svr_epsilon', 0.1)
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
        count_pred = np.expm1(log_pred)
        return max(0.0, count_pred)

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
