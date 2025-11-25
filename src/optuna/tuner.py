"""
Hyperparameter Optimization using Optuna - Phase 8 Implementation
================================================================

This module provides hyperparameter optimization for the nurdle detection pipeline,
focused on SVM classifier parameters for multi-class count prediction.
"""

import optuna
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any, Callable, Tuple, List
import joblib
import json
from datetime import datetime
from tqdm import tqdm
from contextlib import contextmanager
import math
from optuna.trial import TrialState

# Visualization imports (with fallback)
try:
    from optuna.visualization import plot_optimization_history, plot_param_importances, plot_slice
    try:
        import plotly.graph_objects as go
        PLOTLY_AVAILABLE = True
    except ImportError:
        PLOTLY_AVAILABLE = False
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    PLOTLY_AVAILABLE = False


class OptunaTuner:
    """
    Hyperparameter optimization using Optuna with comprehensive visualization.
    
    Optimizes SVM classifier parameters (C, kernel, gamma) for multi-class count prediction.
    """
    
    def __init__(self, config: Dict[str, Any], logger=None):
        """Initialize Optuna tuner with configuration."""
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Optuna configuration
        # n_trials configured per optimization type (svm_optimization, svr_optimization)
        # direction and metric are hardcoded per optimization (maximize f1, minimize error)
        
        # Storage for best parameters
        self.best_params = None
        self.best_value = None
        self.study = None
        
        # Results storage
        self.optimization_history = []
        self.trial_results = []

    def _set_log_levels(self, overrides: Dict[str, int]) -> Dict[str, Tuple[int, bool]]:
        """
        Temporarily override logger levels.

        Returns mapping of previous (level, propagate) values so they can be restored.
        """
        previous = {}
        for name, level in overrides.items():
            logger = logging.getLogger(name)
            previous[name] = (logger.level, logger.propagate)
            logger.setLevel(level)
            logger.propagate = False
        return previous

    def _restore_log_levels(self, previous: Dict[str, Tuple[int, bool]]):
        """Restore logger levels and propagation flags."""
        for name, (level, propagate) in previous.items():
            logger = logging.getLogger(name)
            logger.setLevel(level)
            logger.propagate = propagate

    @contextmanager
    def _suppress_optuna_logging(self):
        """Keep Optuna noise off the console so tqdm progress stays clean."""
        noisy_loggers = {
            "optuna": logging.WARNING,
            "optuna.study": logging.WARNING,
            "optuna.trial": logging.WARNING,
        }
        previous_levels = self._set_log_levels(noisy_loggers)
        previous_verbosity = optuna.logging.get_verbosity()
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        try:
            yield
        finally:
            optuna.logging.set_verbosity(previous_verbosity)
            self._restore_log_levels(previous_levels)

    @contextmanager
    def _suppress_plot_logging(self):
        """Hide kaleido/choreographer noise when saving plots."""
        noisy_loggers = {
            "kaleido": logging.ERROR,
            "kaleido.kaleido": logging.ERROR,
            "kaleido._kaleido_tab": logging.ERROR,
            "choreographer": logging.ERROR,
            "choreographer.browser_async": logging.ERROR,
            "choreographer.browsers.chromium": logging.ERROR,
            "choreographer.utils._tmpfile": logging.ERROR,
        }
        previous_levels = self._set_log_levels(noisy_loggers)
        try:
            yield
        finally:
            self._restore_log_levels(previous_levels)
        
    def optimize(self, pipeline_trainer_func: Callable, output_dir: str = "output/optuna") -> Dict[str, Any]:
        """
        Run SVR hyperparameter optimization for count prediction (minimize MAE).
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        study = optuna.create_study(
            direction='minimize',
            study_name='svr_count_optimization',
            pruner=optuna.pruners.MedianPruner()
        )
        interrupted = False
        trial_results = {}
        # Ranges from config (overridable)
        opt_cfg = {}
        if isinstance(self.config, dict):
            opt_cfg = self.config.get('svr_optimization', {}) or self.config.get('optimization', {}).get('svr_optimization', {}) or self.config.get('optimization', {}).get('svm_optimization', {})
        c_range = opt_cfg.get('svr_c_range', [0.001, 100.0])
        eps_range = opt_cfg.get('svr_epsilon_range', [0.01, 1.0])
        gamma_range = opt_cfg.get('svr_gamma_range', [0.001, 1.0])

        def objective(trial):
            params = {
                'svr_c': trial.suggest_float('svr_c', c_range[0], c_range[1], log=True),
                'svr_kernel': trial.suggest_categorical('svr_kernel', ['linear', 'rbf']),
                'svr_epsilon': trial.suggest_float('svr_epsilon', eps_range[0], eps_range[1])
            }
            if params['svr_kernel'] == 'rbf':
                params['svr_gamma'] = trial.suggest_float('svr_gamma', gamma_range[0], gamma_range[1], log=True)
            try:
                metrics = pipeline_trainer_func(params)
                mae = float(metrics.get('mae', np.inf))
                if not np.isfinite(mae) or mae <= 0 or mae > 1e6:
                    raise optuna.exceptions.TrialPruned("Non-finite or implausible MAE")
                trial_results[trial.number] = {'params': params, 'metrics': metrics}
                return mae
            except Exception as e:
                trial_results[trial.number] = {'params': params, 'metrics': {'mae': 1e6, 'error': str(e)}}
                return 1e6
        n_trials = self.config.get('svr_optimization', {}).get('n_trials', 10)
        try:
            with self._suppress_optuna_logging():
                with tqdm(total=n_trials, desc="SVR Optimization", unit="trial") as pbar:
                    def callback(study, trial):
                        pbar.update(1)
                        pbar.set_postfix({"best_mae": f"{study.best_value:.2f}" if study.best_value is not None else "n/a"})
                    study.optimize(objective, n_trials=n_trials, callbacks=[callback], show_progress_bar=False)
        except KeyboardInterrupt:
            interrupted = True
            self.logger.warning("SVR optimization interrupted; keeping best completed trials so far.")
        # Guard against the case where no trial completed successfully
        completed_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
        if completed_trials:
            best_params = study.best_params
            best_value = float(study.best_value)
        else:
            best_params = {}
            best_value = None
        best_trial_idx = None
        best_trial_metrics = None
        # Find best trial index and metrics
        for t in completed_trials:
            if best_value is not None and t.value == best_value:
                best_trial_idx = t.number
                best_trial_metrics = trial_results.get(t.number, {}).get('metrics', None)
                break
        trials_summary = []
        for t in study.trials:
            value = float(t.value) if t.value is not None else None
            log_value = math.log10(value) if value and value > 0 else None
            trials_summary.append({
                'number': t.number,
                'state': str(t.state),
                'value': value,
                'value_log10': log_value,
                'params': trial_results.get(t.number, {}).get('params', t.params),
                'metrics': trial_results.get(t.number, {}).get('metrics'),
                'datetime_start': t.datetime_start.isoformat() if t.datetime_start else None,
                'datetime_complete': t.datetime_complete.isoformat() if t.datetime_complete else None,
                'duration_seconds': t.duration.total_seconds() if t.duration else None,
            })
        results_json = {
            'best_params': best_params,
            'best_value': best_value,
            'best_value_log10': math.log10(best_value) if best_value and best_value > 0 else None,
            'n_trials': len(study.trials),
            'trial_results': trial_results,
            'best_trial_idx': best_trial_idx,
            'best_trial_metrics': best_trial_metrics,
            'trials': trials_summary,
            'interrupted': interrupted,
        }
        with open(output_path / 'tuning_results.json', 'w') as f:
            json.dump(results_json, f, indent=2)
        # Remove best_params.json if it exists
        best_params_path = output_path / 'best_params.json'
        if best_params_path.exists():
            best_params_path.unlink()
        self._save_optimization_results(study, output_path)
        self.best_params = best_params
        self.best_value = best_value
        self.study = study
        return results_json
    
    # Removed optimize_svm_only, optimize_svr_only, optimize_sequential (legacy, not used in single-stage pipeline)
    
    def optimize_svr_only(self,
                         pipeline_trainer_func: Callable,
                         output_dir: str,
                         fixed_svm_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize SVR regressor parameters on coordinate error (minimize).
        
        Uses fixed SVM parameters from previous optimization.
        
        Args:
            pipeline_trainer_func: Function that trains pipeline and returns metrics
            output_dir: Directory to save optimization results
            fixed_svm_params: Best SVM parameters from step 1
            
        Returns:
            Best SVR parameters (combined with SVM params)
        """
        self.logger.info("=" * 60)
        self.logger.info("OPTIMIZING SVR REGRESSOR (Step 2/2)")
        self.logger.info("=" * 60)
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create study for minimization
        study = optuna.create_study(
            direction='minimize',
            study_name='svr_optimization',
            pruner=optuna.pruners.MedianPruner()
        )
        
        def objective(trial):
            # Use fixed SVM parameters
            params = fixed_svm_params.copy()
            
            # Sample SVR parameters
            params.update({
                'svr_c': trial.suggest_float('svr_c', 0.001, 100.0, log=True),
                'svr_kernel': trial.suggest_categorical('svr_kernel', ['linear', 'rbf']),
                'svr_epsilon': trial.suggest_float('svr_epsilon', 0.01, 1.0),
            })
            
            if params['svr_kernel'] == 'rbf':
                params['svr_gamma'] = trial.suggest_float('svr_gamma', 0.001, 1.0, log=True)
            
            try:
                metrics = pipeline_trainer_func(params)
                return metrics['avg_coordinate_error']
            except Exception as e:
                self.logger.error(f"Trial failed: {e}")
                return 1000.0  # Large error for failed trials
        
        # Run optimization with progress bar
        n_trials = self.config.get('svr_optimization', {}).get('n_trials', 10)
        
        with self._suppress_optuna_logging():
            with tqdm(total=n_trials, desc="SVR Optimization", unit="trial") as pbar:
                def callback(study, trial):
                    pbar.update(1)
                    pbar.set_postfix({"best_error": f"{study.best_value:.2f}px"})
            
                study.optimize(objective, n_trials=n_trials, callbacks=[callback], show_progress_bar=False)
        
        best_svr_params = study.best_params
        self.logger.info(f"Best SVR Coordinate Error: {study.best_value:.2f} pixels")
        self.logger.info(f"Best SVR parameters: {best_svr_params}")
        
        # Combine with SVM params
        best_combined_params = {**fixed_svm_params, **best_svr_params}
        
        # Save results
        self._save_optimization_results(study, output_path / "svr_optimization")
        
        return best_combined_params
    
    def optimize_sequential(self,
                           pipeline_trainer_func: Callable,
                           output_dir: str) -> Dict[str, Any]:
        """
        Run sequential optimization: SVM first, then SVR.
        
        This is the main entry point for optimization.
        
        Args:
            pipeline_trainer_func: Function that trains pipeline and returns metrics
            output_dir: Directory to save optimization results
            
        Returns:
            Best parameters for both SVM and SVR
        """
        self.logger.info("=" * 60)
        self.logger.info("STARTING SEQUENTIAL HYPERPARAMETER OPTIMIZATION")
        self.logger.info("=" * 60)
        
        # Step 1: Optimize SVM
        self.logger.warning("Sequential optimization is deprecated in single-stage pipeline.")
        return {}
    
    def _save_optimization_results(self, study, output_path: Path):
        """Save optimization results and visualizations for a single study."""
        try:
            output_path.mkdir(parents=True, exist_ok=True)
            if len(study.trials) == 0:
                self.logger.warning("No trials completed; skipping optimization visualizations.")
                return
            if not any(t.state == TrialState.COMPLETE for t in study.trials):
                self.logger.warning("No completed trials to visualize; skipping plots.")
                return
            
            # Save study
            study_path = output_path / "study.pkl"
            joblib.dump(study, study_path)
            
            # Do not save best_params.json (all info is in tuning_results.json)
            
            # Generate visualizations (PNG only)
            if VISUALIZATION_AVAILABLE and PLOTLY_AVAILABLE:
                try:
                    with self._suppress_plot_logging():
                        # Optimization history
                        fig_history = plot_optimization_history(study)
                        fig_history.update_yaxes(type="log", title_text="Objective (MAE, log scale)")
                        fig_history.write_image(str(output_path / "optimization_history.png"))
                        self.logger.info("Saved optimization history plot")
                        
                        # Parameter importances
                        if len(study.trials) > 1:
                            fig_importance = plot_param_importances(study)
                            fig_importance.write_image(str(output_path / "param_importances.png"))
                            self.logger.info("Saved parameter importances plot")
                        
                        # Slice plot (parameter relationships)
                        if len(study.trials) > 1:
                            fig_slice = plot_slice(study)
                            fig_slice.update_yaxes(type="log", title_text="Objective (MAE, log scale)")
                            fig_slice.write_image(str(output_path / "param_slice.png"))
                            self.logger.info("Saved parameter slice plot")
                    
                except Exception as viz_error:
                    self.logger.warning(f"Could not generate visualizations with plotly: {viz_error}")
                    # Fallback to matplotlib-based visualizations to ensure files exist
                    try:
                        from optuna.visualization.matplotlib import (
                            plot_optimization_history as mpl_plot_history,
                            plot_param_importances as mpl_plot_importances,
                            plot_slice as mpl_plot_slice,
                        )
                        import matplotlib.pyplot as plt
                        import numpy as np

                        def _save_mpl(obj, path: Path, ylog: bool = False):
                            fig = None
                            axes = None
                            if hasattr(obj, "get_figure"):
                                fig = obj.get_figure()
                                axes = obj
                            elif hasattr(obj, "figure"):
                                fig = obj.figure
                                axes = obj
                            elif isinstance(obj, (list, tuple, np.ndarray)) and len(obj) > 0:
                                first = obj[0]
                                if hasattr(first, "get_figure"):
                                    fig = first.get_figure()
                                    axes = first
                                elif hasattr(first, "figure"):
                                    fig = first.figure
                                    axes = first
                            if fig is None:
                                raise RuntimeError("Could not resolve matplotlib figure from object")
                            if ylog and axes is not None:
                                axes.set_yscale("log")
                                axes.set_ylabel("Objective (MAE, log scale)")
                            fig.savefig(path, bbox_inches="tight", dpi=150)
                            plt.close(fig)

                        _save_mpl(mpl_plot_history(study), output_path / "optimization_history.png", ylog=True)

                        if len(study.trials) > 1:
                            _save_mpl(mpl_plot_importances(study), output_path / "param_importances.png")

                        if len(study.trials) > 1:
                            _save_mpl(mpl_plot_slice(study), output_path / "param_slice.png", ylog=True)
                        self.logger.info("Saved optimization plots using matplotlib fallback")
                    except Exception as fallback_error:
                        self.logger.warning(f"Could not generate matplotlib visualizations: {fallback_error}")
            else:
                self.logger.warning("Optuna visualization libraries not available, skipping plots")
            
            self.logger.info(f"Saved optimization results to {output_path}")
        except Exception as e:
            self.logger.error(f"Error saving optimization results: {e}")
    
    # Removed create_training_callback (legacy, not used in single-stage pipeline)
    
    def load_best_parameters(self, output_dir: str) -> Dict[str, Any]:
        """Load previously optimized parameters."""
        best_params_path = Path(output_dir) / "best_parameters.json"
        
        if best_params_path.exists():
            with open(best_params_path, 'r') as f:
                data = json.load(f)
                self.best_params = data['best_params']
                self.best_value = data['best_value']
                self.logger.info(f"Loaded best parameters from {best_params_path}")
                return self.best_params
        else:
            raise FileNotFoundError(f"No optimization results found at {best_params_path}")
