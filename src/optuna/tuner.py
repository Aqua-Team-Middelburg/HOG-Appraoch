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
from typing import Dict, Any, Callable, Tuple
import joblib
import json
from tqdm import tqdm
from contextlib import contextmanager
import math

# Visualization imports (matplotlib-only to avoid lingering kaleido/browser threads)
try:
    from optuna.visualization import plot_optimization_history, plot_param_importances, plot_slice
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
PLOTLY_AVAILABLE = False  # force disable plotly/kaleido


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
        # n_trials configured per optimization type (svr)
        # direction and metric are hardcoded per optimization (minimize error)
        
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
        trial_results = {}
        # Bounds from config
        opt_cfg = self.config.get('optimization', {}) if isinstance(self.config, dict) else {}
        svr_cfg = opt_cfg.get('svr', {})
        c_min = svr_cfg.get('c_min', 0.1)
        c_max = svr_cfg.get('c_max', 10.0)
        eps_min = svr_cfg.get('eps_min', 0.01)
        eps_max = svr_cfg.get('eps_max', 0.5)
        kernel_choices = svr_cfg.get('kernel', ['rbf'])
        if isinstance(kernel_choices, str):
            kernel_choices = [kernel_choices]
        gamma_min = svr_cfg.get('gamma_min')
        gamma_max = svr_cfg.get('gamma_max')
        gamma_spec = svr_cfg.get('gamma', ['scale'])
        if isinstance(gamma_spec, str):
            gamma_choices = [gamma_spec]
        elif isinstance(gamma_spec, (list, tuple)):
            gamma_choices = list(gamma_spec)
        else:
            gamma_choices = None
        n_trials = svr_cfg.get('n_trials', 10)

        def objective(trial):
            params = {
                'svr_c': trial.suggest_float('svr_c', c_min, c_max, log=True),
                'svr_kernel': trial.suggest_categorical('svr_kernel', kernel_choices),
                'svr_epsilon': trial.suggest_float('svr_epsilon', eps_min, eps_max)
            }
            if params['svr_kernel'] == 'rbf':
                if gamma_min is not None and gamma_max is not None:
                    params['svr_gamma'] = trial.suggest_float('svr_gamma', gamma_min, gamma_max, log=True)
                elif gamma_choices:
                    params['svr_gamma'] = trial.suggest_categorical('svr_gamma', gamma_choices)
                else:
                    params['svr_gamma'] = 'scale'
            try:
                metrics = pipeline_trainer_func(params)
                if isinstance(metrics, (int, float, np.number)):
                    mae = float(metrics)
                    metrics = {'mae': mae}
                elif isinstance(metrics, dict):
                    mae = float(metrics.get('mae', np.inf))
                else:
                    raise ValueError("Pipeline trainer must return a dict with 'mae' or a numeric MAE.")
                if not np.isfinite(mae) or mae <= 0 or mae > 1e6:
                    raise optuna.exceptions.TrialPruned("Non-finite or implausible MAE")
                trial_results[trial.number] = {'params': params, 'metrics': metrics}
                return mae
            except Exception as e:
                trial_results[trial.number] = {'params': params, 'metrics': {'mae': 1e6, 'error': str(e)}}
                return 1e6
        with self._suppress_optuna_logging():
            with tqdm(total=n_trials, desc="SVR Optimization", unit="trial") as pbar:
                def callback(study, trial):
                    pbar.update(1)
                    pbar.set_postfix({"best_mae": f"{study.best_value:.2f}"})
                study.optimize(objective, n_trials=n_trials, callbacks=[callback], show_progress_bar=False)
        best_params = study.best_params
        best_trial_idx = None
        best_trial_metrics = None
        # Find best trial index and metrics
        for t in study.trials:
            if t.value == study.best_value:
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
            'best_value': float(study.best_value),
            'best_value_log10': math.log10(study.best_value) if study.best_value and study.best_value > 0 else None,
            'n_trials': len(study.trials),
            'trial_results': trial_results,
            'best_trial_idx': best_trial_idx,
            'best_trial_metrics': best_trial_metrics,
            'trials': trials_summary,
        }
        with open(output_path / 'tuning_results.json', 'w') as f:
            json.dump(results_json, f, indent=2)
        # Remove best_params.json if it exists
        best_params_path = output_path / 'best_params.json'
        if best_params_path.exists():
            best_params_path.unlink()
        self._save_optimization_results(study, output_path)
        self.best_params = best_params
        self.best_value = study.best_value
        self.study = study
        return results_json
    
    def _save_optimization_results(self, study, output_path: Path):
        """Save optimization results and visualizations for a single study."""
        try:
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save study
            study_path = output_path / "study.pkl"
            joblib.dump(study, study_path)
            
            # Do not save best_params.json (all info is in tuning_results.json)
            
            # Generate visualizations (PNG, matplotlib only to avoid kaleido threads)
            if VISUALIZATION_AVAILABLE:
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
