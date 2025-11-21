"""
Model training for two-stage nurdle detection pipeline.

This module implements:
- SVM Classifier: Binary classification (is window positive?)
- SVR Regressor: Coordinate offset prediction (where is nurdle in window?)
"""

import numpy as np
import pickle
from pathlib import Path
from typing import List, Tuple, Dict, Any
import logging
from sklearn.svm import SVC, SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import KFold

from ..features import TrainingWindow


class ModelTrainer:
    """
    Trains SVM classifier and SVR regressor for two-stage detection.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize model trainer with configuration."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Models
        self.svm_classifier = None      # Binary classifier (all windows)
        self.svr_regressor = None       # Coordinate regressor (positive windows only)
        
    def train_models(self, window_batches) -> Dict[str, Any]:
        """
        Train both SVM classifier and SVR regressor with metrics collection.
        
        Args:
            window_batches: Iterator of training window batches
            
        Returns:
            Dictionary with training metrics for SVM, SVR, and combined pipeline
        """
        self.logger.info("Training models...")
        
        # Collect all windows
        all_windows = []
        for batch in window_batches:
            all_windows.extend(batch)
        
        if not all_windows:
            raise ValueError("No training windows generated")
        
        self.logger.info(f"Total training windows: {len(all_windows)}")
        
        # Separate positive and negative windows
        positive_windows = [w for w in all_windows if w.is_nurdle]
        negative_windows = [w for w in all_windows if not w.is_nurdle]
        
        self.logger.info(f"Positive windows: {len(positive_windows)}")
        self.logger.info(f"Negative windows: {len(negative_windows)}")
        
        # Train SVM classifier on all windows
        svm_metrics = self.train_svm_classifier(all_windows)
        
        # Train SVR regressor on positive windows only
        svr_metrics = self.train_svr_regressor(positive_windows)
        
        self.logger.info("Model training completed")
        
        # Combine metrics
        training_metrics = {
            'svm': svm_metrics,
            'svr': svr_metrics,
            'total_windows': len(all_windows),
            'positive_windows': len(positive_windows),
            'negative_windows': len(negative_windows)
        }
        
        return training_metrics
    
    def train_svm_classifier(self, all_windows: List[TrainingWindow]) -> Dict[str, float]:
        """
        Train SVM classifier on ALL windows (positive + negative).
        
        Args:
            all_windows: List of all training windows
            
        Returns:
            Dictionary with SVM training metrics
        """
        X = np.array([w.features for w in all_windows])
        y = np.array([w.is_nurdle for w in all_windows])
        
        self.svm_classifier = SVC(
            C=self.config.get('svm_c', 1.0),
            kernel=self.config.get('svm_kernel', 'rbf'),
            gamma=self.config.get('svm_gamma', 'scale'),
            probability=True,  # Enable confidence scores via predict_proba
            random_state=42
        )
        
        self.svm_classifier.fit(X, y)
        
        # Calculate training accuracy and predictions
        train_accuracy = self.svm_classifier.score(X, y)
        y_pred = self.svm_classifier.predict(X)
        
        # Calculate precision, recall, f1
        from sklearn.metrics import precision_score, recall_score, f1_score
        precision = precision_score(y, y_pred, zero_division=0)
        recall = recall_score(y, y_pred, zero_division=0)
        f1 = f1_score(y, y_pred, zero_division=0)
        
        self.logger.info(f"SVM Classifier trained on {len(X)} windows")
        self.logger.info(f"SVM training accuracy: {train_accuracy:.3f}")
        self.logger.info(f"SVM precision: {precision:.3f}, recall: {recall:.3f}, f1: {f1:.3f}")
        
        return {
            'accuracy': float(train_accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'n_samples': len(X)
        }
    
    def train_svr_regressor(self, positive_windows: List[TrainingWindow]) -> Dict[str, float]:
        """
        Train SVR regressor on POSITIVE windows only.
        
        Predicts coordinate offsets (offset_x, offset_y) from window center.
        
        Args:
            positive_windows: List of positive training windows with offset labels
            
        Returns:
            Dictionary with SVR training metrics (includes CV error for hyperparameter tuning)
        """
        if not positive_windows:
            raise ValueError("No positive windows for SVR training")
        
        X = np.array([w.features for w in positive_windows])
        y = np.array([[w.offset_x, w.offset_y] for w in positive_windows])
        
        # Create base SVR
        base_svr = SVR(
            C=self.config.get('svr_c', 1.0),
            kernel=self.config.get('svr_kernel', 'rbf'),
            gamma=self.config.get('svr_gamma', 'scale'),
            epsilon=self.config.get('svr_epsilon', 0.1)
        )
        
        # Use MultiOutputRegressor to predict both x and y offsets
        self.svr_regressor = MultiOutputRegressor(base_svr)
        
        # Perform cross-validation to get generalization error for hyperparameter tuning
        from sklearn.model_selection import KFold, cross_val_score
        from sklearn.metrics import make_scorer
        
        def euclidean_error(y_true, y_pred):
            """Calculate mean euclidean distance error."""
            errors = np.sqrt(np.sum((y_true - y_pred)**2, axis=1))
            return np.mean(errors)
        
        scorer = make_scorer(euclidean_error, greater_is_better=False)
        
        # 5-fold cross-validation (or fewer folds if not enough samples)
        n_splits = min(5, len(X))
        if n_splits >= 2:
            # Use shuffled CV with different random state each time
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=None)
            cv_scores = cross_val_score(
                self.svr_regressor, X, y, 
                cv=cv, 
                scoring=scorer,
                n_jobs=-1
            )
            cv_mean_error = float(-np.mean(cv_scores))  # Negate because scorer returns negative
            cv_std_error = float(np.std(-cv_scores))
        else:
            # Not enough samples for CV, use training error
            cv_mean_error = 0.0
            cv_std_error = 0.0
        
        # Train final model on all data
        self.svr_regressor.fit(X, y)
        
        # Calculate training error
        y_pred = self.svr_regressor.predict(X)
        errors = np.sqrt(np.sum((y - y_pred)**2, axis=1))
        mean_error = float(np.mean(errors))
        std_error = float(np.std(errors))
        max_error = float(np.max(errors))
        
        self.logger.info(f"SVR Regressor trained on {len(X)} positive windows")
        self.logger.info(f"SVR training error: {mean_error:.2f} pixels (std: {std_error:.2f}, max: {max_error:.2f})")
        if n_splits >= 2:
            self.logger.info(f"SVR CV error ({n_splits}-fold): {cv_mean_error:.2f} ± {cv_std_error:.2f} pixels")
        
        return {
            'mean_error': cv_mean_error if n_splits >= 2 else mean_error,  # Use CV error for tuning
            'training_error': mean_error,
            'cv_error': cv_mean_error if n_splits >= 2 else None,
            'std_error': std_error,
            'max_error': max_error,
            'n_samples': len(X)
        }
    
    def predict(self, features: np.ndarray) -> Tuple[bool, float, float]:
        """
        Two-stage prediction: SVM classification then SVR coordinate refinement.
        
        Args:
            features: Feature vector for a single window
            
        Returns:
            Tuple of (is_nurdle, offset_x, offset_y)
        """
        if self.svm_classifier is None or self.svr_regressor is None:
            raise ValueError("Models not trained yet")
        
        features_2d = features.reshape(1, -1)
        
        # Stage 1: SVM Classification
        svm_pred = self.svm_classifier.predict(features_2d)[0]
        
        # Early return if classified as negative
        if not svm_pred:
            return False, 0.0, 0.0
        
        # Stage 2: SVR Coordinate Refinement
        offsets = self.svr_regressor.predict(features_2d)[0]
        offset_x = float(offsets[0])
        offset_y = float(offsets[1])
        
        return True, offset_x, offset_y
    
    def save_models(self, output_dir: str) -> None:
        """Save trained models to disk."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if self.svm_classifier:
            svm_path = output_path / "svm_classifier.pkl"
            with open(svm_path, 'wb') as f:
                pickle.dump(self.svm_classifier, f)
            self.logger.info(f"Saved SVM classifier to {svm_path}")
        
        if self.svr_regressor:
            svr_path = output_path / "svr_regressor.pkl"
            with open(svr_path, 'wb') as f:
                pickle.dump(self.svr_regressor, f)
            self.logger.info(f"Saved SVR regressor to {svr_path}")
    
    def load_models(self, model_dir: str) -> None:
        """Load trained models from disk."""
        model_path = Path(model_dir)
        
        svm_path = model_path / "svm_classifier.pkl"
        if svm_path.exists():
            with open(svm_path, 'rb') as f:
                self.svm_classifier = pickle.load(f)
            self.logger.info(f"Loaded SVM classifier from {svm_path}")
        
        svr_path = model_path / "svr_regressor.pkl"
        if svr_path.exists():
            with open(svr_path, 'rb') as f:
                self.svr_regressor = pickle.load(f)
            self.logger.info(f"Loaded SVR regressor from {svr_path}")
    
    def save_training_metrics(self, metrics: Dict[str, Any], output_dir: str) -> None:
        """
        Save training metrics and generate visualization curves.
        
        Args:
            metrics: Training metrics dictionary
            output_dir: Directory to save metrics
        """
        import json
        import matplotlib.pyplot as plt
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save SVM metrics
        svm_dir = output_path / 'svm'
        svm_dir.mkdir(exist_ok=True)
        
        with open(svm_dir / 'metrics.json', 'w') as f:
            json.dump(metrics['svm'], f, indent=2)
        
        # Create SVM metrics visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        svm_metrics = metrics['svm']
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        metric_values = [svm_metrics['accuracy'], svm_metrics['precision'], 
                        svm_metrics['recall'], svm_metrics['f1_score']]
        
        bars = ax.bar(metric_names, metric_values, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'], alpha=0.7, edgecolor='black')
        ax.set_ylim(0, 1.1)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title(f'SVM Classifier Training Metrics (n={svm_metrics["n_samples"]})', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(svm_dir / 'training_metrics.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Save SVR metrics
        svr_dir = output_path / 'svr'
        svr_dir.mkdir(exist_ok=True)
        
        with open(svr_dir / 'metrics.json', 'w') as f:
            json.dump(metrics['svr'], f, indent=2)
        
        # Create SVR error distribution visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        svr_metrics = metrics['svr']
        
        error_types = ['Mean Error', 'Std Error', 'Max Error']
        error_values = [svr_metrics['mean_error'], svr_metrics['std_error'], svr_metrics['max_error']]
        
        bars = ax.bar(error_types, error_values, color=['steelblue', 'orange', 'crimson'], alpha=0.7, edgecolor='black')
        ax.set_ylabel('Error (pixels)', fontsize=12)
        ax.set_title(f'SVR Regressor Training Error (n={svr_metrics["n_samples"]})', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, error_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(svr_dir / 'training_error.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Saved training metrics to {output_path}")
    
    @property
    def is_trained(self) -> bool:
        """Check if both models are trained."""
        return self.svm_classifier is not None and self.svr_regressor is not None
    
    def load_tuning_results(self, tuning_dir: Path) -> Dict[str, Any]:
        """
        Load optimized hyperparameters from tuning results.
        
        Args:
            tuning_dir: Directory containing tuning results (output/04_tuning)
            
        Returns:
            Dictionary of best parameters, empty dict if not found
        """
        import json
        
        tuning_path = Path(tuning_dir)
        best_params = {}
        
        if not tuning_path.exists():
            self.logger.warning(f"Tuning directory not found: {tuning_path}")
            return best_params
        
        # Try to load best parameters from tuning
        svm_params_file = tuning_path / 'svm_optimization' / 'best_params.json'
        svr_params_file = tuning_path / 'svr_optimization' / 'best_params.json'
        
        if svm_params_file.exists() and svr_params_file.exists():
            with open(svm_params_file, 'r') as f:
                svm_results = json.load(f)
                best_params.update(svm_results['best_params'])
            with open(svr_params_file, 'r') as f:
                svr_results = json.load(f)
                best_params.update(svr_results['best_params'])
            self.logger.info(f"Loaded optimized hyperparameters: {best_params}")
        else:
            self.logger.warning("Tuning results not found, no parameters loaded")
        
        return best_params