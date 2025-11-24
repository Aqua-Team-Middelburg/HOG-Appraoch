import argparse
import logging
import sys
import shutil
import cv2
import json
import zipfile
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import numpy as np
import os
import threading

# Import pipeline components
try:
    from src.utils.config import load_config
    from src.data import DataLoader
    from src.features import FeatureExtractor
    from src.models import ModelTrainer
    from src.evaluation import ModelEvaluator
    from tqdm import tqdm
except ImportError as e:
    print(f"Error importing pipeline components: {e}")
    print("Make sure all src modules are properly installed")
    sys.exit(1)


class NurdlePipeline:
    """
    Modular pipeline orchestrator with checkpoint-based stage execution.
    
    Each stage produces intermediate outputs and can be run independently.
    """
    
    # Define available stages
    AVAILABLE_STAGES = ['normalization', 'features', 'tuning', 'training', 'evaluation', 'save']
    
    def __init__(self, config_path: str):
        """Initialize pipeline with configuration."""
        self.config_path = config_path
        self.config = load_config(config_path)  # Load config on init
        self.logger = self._setup_logging()
        
        # Define checkpoint and output directories
        self.temp_dir = Path('temp')
        self.checkpoint_normalized = self.temp_dir / 'normalized'
        # Stage output directories
        self.output_dirs = {
            'normalization': Path('output/01_normalization'),
            'features': Path('output/02_features'),
            'tuning': Path('output/03_tuning'),
            'training': Path('output/04_training'),
            'evaluation': Path('output/05_evaluation')
        }
        
        # Models directory
        self.models_dir = Path('output/models')
        
        # Initialize components (will be created as needed per stage)
        self.data_loader = None
        self.feature_extractor = None
        self.model_trainer = None
        self.evaluator = None
        
        self.logger.info("Modular pipeline initialized successfully")
    
    # Remove _load_checkpoints (legacy, not used in single-stage pipeline)
    
    def _setup_logging(self):
        """Setup pipeline-specific logging."""
        logs_dir = Path('output/logs')
        logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Clean up old log files, keeping only the last 3
        self._cleanup_old_logs(logs_dir, keep_last=3)
        
        logger = logging.getLogger('NurdlePipeline')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            # Console handler
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.INFO)
            
            # File handler
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            file_handler = logging.FileHandler(logs_dir / f'pipeline_{timestamp}.log')
            file_handler.setLevel(logging.DEBUG)
            
            # Formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(formatter)
            file_handler.setFormatter(formatter)
            
            logger.addHandler(console_handler)
            logger.addHandler(file_handler)
        
        return logger
    
    def _cleanup_old_logs(self, logs_dir: Path, keep_last: int = 3):
        """Remove old log files, keeping only the most recent ones."""
        log_files = sorted(logs_dir.glob('pipeline_*.log'), key=lambda f: f.stat().st_mtime, reverse=True)
        
        # Remove older log files beyond keep_last
        for old_log in log_files[keep_last:]:
            try:
                old_log.unlink()
            except Exception:
                pass  # Ignore errors during cleanup
    
    def _create_stage_output_dir(self, stage: str):
        """Create and clear stage output directory."""
        output_dir = self.output_dirs[stage]
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.logger.debug(f"Created output directory: {output_dir}")
        return output_dir
    
    def _check_checkpoint_exists(self, checkpoint_path: Path, stage_name: str, prerequisite: str) -> bool:
        """Check if required checkpoint exists."""
        if not checkpoint_path.exists():
            error_msg = f"ERROR: {stage_name} - Required checkpoint not found: {checkpoint_path}. Run --steps {prerequisite} first."
            self.logger.error(error_msg)
            return False
        return True
    
    def run(self, steps: List[str]) -> Dict[str, Any]:
        """
        Run specified pipeline stages with checkpoint-based execution.

        Available stages:
        - normalization: Load and normalize images, save checkpoints
        - features: Extract and visualize features
        - training: Train SVM count classifier and save metrics
        - evaluation: Evaluate count prediction accuracy
        - save: Zip output folder

        Args:
            steps: List of stage names to execute

        Returns:
            Dictionary with results from executed stages
        """
        self.logger.info("=" * 80)
        self.logger.info(f"STARTING MODULAR PIPELINE - STEPS: {', '.join(steps)}")
        self.logger.info("=" * 80)
        
        results = {}
        
        last_stage = None
        try:
            for stage in steps:
                if stage not in self.AVAILABLE_STAGES:
                    raise ValueError(f"Unknown stage: {stage}. Available: {self.AVAILABLE_STAGES}")
                self.logger.info(f"\n{'='*80}")
                self.logger.info(f"STAGE: {stage.upper()}")
                self.logger.info(f"{'='*80}")
                # Execute stage
                if stage == 'normalization':
                    results[stage] = self._run_normalization_stage()
                elif stage == 'features':
                    results[stage] = self._run_features_stage()
                elif stage == 'tuning':
                    results[stage] = self._run_tuning_stage()
                elif stage == 'training':
                    results[stage] = self._run_training_stage()
                elif stage == 'evaluation':
                    results[stage] = self._run_evaluation_stage()
                elif stage == 'save':
                    results[stage] = self._run_save_stage()
                self.logger.info(f"Stage '{stage}' completed successfully")
                last_stage = stage
        except Exception as e:
            stage_name = last_stage if last_stage is not None else 'unknown'
            self.logger.error(f"Pipeline failed at stage '{stage_name}': {e}", exc_info=True)
            raise
        
        self.logger.info("=" * 80)
        self.logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        self.logger.info("=" * 80)
        
        return results
    
    def _run_normalization_stage(self) -> Dict[str, Any]:
        """
        Stage 1: Load images, normalize, and save checkpoints.
        
        Outputs:
        - temp/normalized/ - Checkpoint with normalized images and metadata
        - output/01_normalization/ - Sample visualizations
        """
        output_dir = self._create_stage_output_dir('normalization')
        
        # Initialize data loader
        input_dir = self.config.data.get('input_dir', 'input')
        self.data_loader = DataLoader(self.config.data)
        
        # Load and process data
        self.logger.info(f"Loading data from {input_dir}...")
        self.data_loader.load_annotations(input_dir)
        
        if self.data_loader.num_annotations == 0:
            raise ValueError(f"No annotations loaded from {input_dir}")
        
        # Split train/test FIRST (before normalization)
        self.data_loader.split_train_test()
        self.logger.info(f"Loaded {self.data_loader.num_annotations} images: "
                        f"{self.data_loader.num_train} train, {self.data_loader.num_test} test")
        
        # THEN normalize and save (normalizes all images, saves split info)
        self.logger.info("Normalizing and saving all images...")
        self.data_loader.save_normalized_data(self.checkpoint_normalized)
        
        # Generate sample visualizations
        self.logger.info("Generating normalization visualizations...")
        self.data_loader.visualize_normalization_samples(self.checkpoint_normalized, output_dir, num_samples=3)
        
        return {
            'total_images': self.data_loader.num_annotations,
            'train_images': self.data_loader.num_train,
            'test_images': self.data_loader.num_test,
            'checkpoint': str(self.checkpoint_normalized)
        }
    
    def _run_features_stage(self) -> Dict[str, Any]:
        """
        Stage 2: Extract and visualize features.

        Prerequisites:
        - temp/normalized/ must exist

        Outputs:
        - output/03_features/ - Feature visualizations (HOG overlays, LBP patterns)

        Note: Features are NOT saved, training uses generator.
        """
        if not self._check_checkpoint_exists(self.checkpoint_normalized, 'Features', 'normalization'):
            raise FileNotFoundError(f"ERROR: Features - Normalized data not found. Run --steps normalization first.")
        output_dir = self._create_stage_output_dir('features')
        self.logger.info("Extracting features for all images...")
        self.data_loader = DataLoader(self.config.data)
        self.data_loader.load_normalized_data(self.checkpoint_normalized)
        feature_config = {
            'hog_cell_size': self.config.features.get('hog_cell_size', [4, 4]),
            'image_size': self.config.features.get('image_size', [1080, 1080])
        }
        self.feature_extractor = FeatureExtractor(feature_config)
        # Visualize feature samples for up to 3 test images
        self.feature_extractor.visualize_feature_samples(
            self.data_loader.test_annotations,
            self.data_loader.load_image,
            str(output_dir),
            num_samples=3
        )
        return {'visualization_dir': str(output_dir)}
    
    def _run_tuning_stage(self) -> Dict[str, Any]:
        """
        Stage 3: Hyperparameter tuning for SVR regressor.

        Prerequisites:
        - temp/normalized/ must exist

        Outputs:
        - output/03_tuning/ - Tuning results and best parameters
        """
        from src.optuna import OptunaTuner
        if not self._check_checkpoint_exists(self.checkpoint_normalized, 'Tuning', 'normalization'):
            raise FileNotFoundError(f"ERROR: Tuning - Normalized data not found. Run --steps normalization first.")
        output_dir = self._create_stage_output_dir('tuning')
        self.logger.info("Loading normalized data for tuning...")
        self.data_loader = DataLoader(self.config.data)
        self.data_loader.load_normalized_data(self.checkpoint_normalized)
        feature_config = {
            'hog_cell_size': self.config.features.get('hog_cell_size', [4, 4]),
            'image_size': self.config.features.get('image_size', [1080, 1080])
        }
        self.feature_extractor = FeatureExtractor(feature_config)
        training_data = list(self.feature_extractor.generate_training_data(
            self.data_loader.train_annotations, self.data_loader.load_image))
        config_dict = self.config if isinstance(self.config, dict) else vars(self.config)
        tuner = OptunaTuner(config_dict)
        def trainer_func(params):
            trainer = ModelTrainer({**config_dict, **params})
            return trainer.train_count_regressor(training_data)
        tuning_results = tuner.optimize(trainer_func, str(output_dir))
        best_params = tuning_results.get('best_params', {})
        return {
            'best_params': best_params,
            'best_mae': tuning_results.get('best_value'),
            'n_trials': tuning_results.get('n_trials'),
            'output_dir': str(output_dir),
            'tuning_results_path': str(output_dir / 'tuning_results.json'),
        }
    
    def _run_training_stage(self) -> Dict[str, Any]:
        """
        Stage 3: Train SVR regressor and collect metrics.

        Prerequisites:
        - temp/normalized/ must exist

        Outputs:
        - output/04_training/ - Training metrics and curves
        - output/models/ - Saved models
        """
        if not self._check_checkpoint_exists(self.checkpoint_normalized, 'Training', 'normalization'):
            raise FileNotFoundError(f"ERROR: Training - Normalized data not found. Run --steps normalization first.")
        output_dir = self._create_stage_output_dir('training')
        self.logger.info("Loading normalized data for training...")
        self.data_loader = DataLoader(self.config.data)
        self.data_loader.load_normalized_data(self.checkpoint_normalized)
        feature_config = {
            'hog_cell_size': self.config.features.get('hog_cell_size', [4, 4]),
            'image_size': self.config.features.get('image_size', [1080, 1080])
        }
        self.feature_extractor = FeatureExtractor(feature_config)
        training_data = list(self.feature_extractor.generate_training_data(
            self.data_loader.train_annotations, self.data_loader.load_image))
        config_dict = self.config if isinstance(self.config, dict) else vars(self.config)
        self.model_trainer = ModelTrainer(config_dict)
        training_metrics = self.model_trainer.train_count_regressor(training_data)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.model_trainer.save_model(str(self.models_dir))
        metrics_path = output_dir / 'training_metrics.json'
        # Always overwrite to ensure metrics are present
        with open(metrics_path, 'w') as f:
            json.dump(training_metrics, f, indent=2)
        return {
            'training_metrics': training_metrics,
            'models_dir': str(self.models_dir),
            'output_dir': str(output_dir)
        }
    
    def _run_evaluation_stage(self) -> Dict[str, Any]:
        """
        Stage 4: Evaluate trained SVR regressor and generate visualizations.

        Prerequisites:
        - temp/normalized/ must exist
        - output/models/ must have trained models

        Outputs:
        - output/05_evaluation/ - Evaluation metrics, report, and visualizations
        """
        if not self._check_checkpoint_exists(self.checkpoint_normalized, 'Evaluation', 'normalization'):
            raise FileNotFoundError(f"ERROR: Evaluation - Normalized data not found. Run --steps normalization first.")
        if not self.models_dir.exists() or not any(self.models_dir.glob('*.pkl')):
            raise FileNotFoundError(f"ERROR: Evaluation - No trained models found in {self.models_dir}. Run --steps training first.")
        output_dir = self._create_stage_output_dir('evaluation')
        self.logger.info("Loading normalized data for evaluation...")
        self.data_loader = DataLoader(self.config.data)
        self.data_loader.load_normalized_data(self.checkpoint_normalized)
        feature_config = {
            'hog_cell_size': self.config.features.get('hog_cell_size', [4, 4]),
            'image_size': self.config.features.get('image_size', [1080, 1080])
        }
        feature_extractor = FeatureExtractor(feature_config)
        model_trainer = ModelTrainer(self.config if isinstance(self.config, dict) else vars(self.config))
        model_trainer.load_model(str(self.models_dir))
        self.evaluator = ModelEvaluator()
        test_annots = self.data_loader.test_annotations
        y_true = []
        y_pred = []
        for annot in test_annots:
            image = self.data_loader.load_image(annot.image_path)
            features = feature_extractor.extract_image_features(image)
            pred_count = model_trainer.predict_count(features)
            y_true.append(annot.nurdle_count)
            y_pred.append(pred_count)
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((np.array(y_true) - np.array(y_pred)) / np.maximum(np.array(y_true), 1))) * 100
        eval_metrics = {
            'mae': float(mae),
            'rmse': float(rmse),
            'mape': float(mape),
            'n_test': len(y_true)
        }
        with open(output_dir / 'evaluation_metrics.json', 'w') as f:
            json.dump(eval_metrics, f, indent=2)
        import matplotlib.pyplot as plt
        training_output_dir = self.output_dirs['training']
        training_metrics_path = training_output_dir / 'training_metrics.json'
        if training_metrics_path.exists():
            with open(training_metrics_path, 'r') as f:
                training_metrics = json.load(f)
        else:
            training_metrics = None
            self.logger.warning(f"Training metrics not found at {training_metrics_path}; skipping train/test comparison plot")
        if training_metrics:
            metrics_names = ['MAE', 'RMSE', 'MAPE']
            train_vals = [
                training_metrics.get('mae', 0),
                training_metrics.get('rmse', 0),
                training_metrics.get('mape', 0)
            ]
            test_vals = [eval_metrics['mae'], eval_metrics['rmse'], eval_metrics['mape']]
            x = np.arange(len(metrics_names))
            plt.figure(figsize=(7, 5))
            plt.bar(x - 0.2, train_vals, width=0.4, label='Train')
            plt.bar(x + 0.2, test_vals, width=0.4, label='Test')
            plt.xticks(x, metrics_names)
            plt.ylabel('Metric Value')
            plt.title('Training vs Testing Metrics')
            plt.legend()
            plt.tight_layout()
            plt.savefig(output_dir / 'train_vs_test_metrics.png')
            plt.close()
        # Individual test image visualizations
        for ann, pred_count in zip(test_annots, y_pred):
            img = self.data_loader.load_image(ann.image_path)
            actual_count = ann.nurdle_count
            plt.figure(figsize=(6, 6))
            plt.imshow(img, cmap='gray')
            plt.title(f"Actual: {actual_count} | Predicted: {pred_count:.1f}")
            plt.axis('off')
            out_path = output_dir / f"{ann.image_id}_eval.png"
            plt.savefig(out_path, bbox_inches='tight')
            plt.close()
        return {
            'evaluation_metrics': eval_metrics,
            'visualization_dir': str(output_dir)
        }
    
    # Remove _generate_detailed_report (legacy, not used in single-stage pipeline)
    
    def clean_temp(self):
        """Clean temporary checkpoint files."""
        self.logger.info("Cleaning temporary files...")
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            self.logger.info(f"Removed {self.temp_dir}")
        self.temp_dir.mkdir(parents=True, exist_ok=True)
    
    def _run_save_stage(self) -> Dict[str, Any]:
        """Stage 7: Save a zipped copy of the output folder to configured save directory.
        
        Prerequisites:
        - output/ directory must exist with pipeline results
        
        Outputs:
        - Zipped copy of output folder in save_dir
        
        Returns:
            Dictionary with zip file path
        """
        save_dir = self.config.data.get('save_dir')
        
        if not save_dir:
            raise ValueError("ERROR: Save - No save_dir configured in config.yaml. Set data.save_dir to enable saving.")
        
        # Check if output directory exists
        output_dir = Path('output')
        if not output_dir.exists():
            raise FileNotFoundError(f"ERROR: Save - Output directory not found: {output_dir}. Run pipeline stages first.")
        
        # Create save directory if it doesn't exist
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Create zip filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        zip_filename = f"output_{timestamp}.zip"
        zip_path = save_path / zip_filename
        
        self.logger.info(f"Creating zip archive of output folder...")
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Walk through output directory and add all files
            file_count = 0
            for file_path in output_dir.rglob('*'):
                if file_path.is_file():
                    # Calculate relative path for archive
                    arcname = file_path.relative_to(output_dir.parent)
                    zipf.write(file_path, arcname)
                    self.logger.debug(f"Added to zip: {arcname}")
                    file_count += 1
        
        zip_size_mb = zip_path.stat().st_size / (1024*1024)
        self.logger.info(f"Output folder saved to: {zip_path}")
        self.logger.info(f"Archived {file_count} files, zip size: {zip_size_mb:.2f} MB")
        
        return {
            'zip_path': str(zip_path),
            'zip_size_mb': round(zip_size_mb, 2),
            'files_archived': file_count
        }


def setup_logging():
    """Setup basic logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def _shutdown_plot_backends():
    """Close any plotting backends (matplotlib/plotly-kaleido) to avoid hanging threads."""
    try:
        import matplotlib.pyplot as plt
        plt.close('all')
    except Exception:
        pass

    try:
        import plotly.io as pio
        # Kaleido keeps a worker process alive; try to shut it down if present
        scope = getattr(getattr(pio, "kaleido", None), "scope", None)
        if scope and hasattr(scope, "_shutdown_kaleido"):
            try:
                scope._shutdown_kaleido()
            except Exception:
                pass
    except Exception:
        pass


def _wait_for_background_threads(logger, timeout: float = 2.0) -> None:
    """
    Wait briefly for non-daemon threads to finish; log any lingerers.

    If stubborn threads remain, forcefully terminate the process so CLI prompts return.
    """
    alive = [
        t for t in threading.enumerate()
        if t is not threading.current_thread() and t is not threading.main_thread() and t.is_alive()
    ]
    if not alive:
        return

    logger.info("Waiting for background threads to finish...")
    for t in alive:
        t.join(timeout=timeout)

    remaining = [
        t for t in threading.enumerate()
        if t is not threading.current_thread() and t is not threading.main_thread() and t.is_alive()
    ]
    if remaining:
        names = ", ".join(f"{t.name}" for t in remaining)
        logger.warning(f"Forcing shutdown with lingering threads: {names}")
        logging.shutdown()
        os._exit(0)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Modular Nurdle Detection Pipeline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        'config',
        nargs='?',
        default='config.yaml',
        help='Path to configuration YAML file (default: config.yaml)'
    )
    
    parser.add_argument(
        '--steps',
        type=str,
        default=None,
        help='Comma-separated list of stages to run (default: all stages except save). Options: normalization,windows,features,tuning,training,evaluation,save'
    )
    
    parser.add_argument(
        '--clean-temp',
        action='store_true',
        help='Clean temporary checkpoint files'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()


def main():
    """Main entry point for the pipeline."""
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    setup_logging()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Initialize pipeline
        pipeline = NurdlePipeline(args.config)
        
        # Clean temp if requested
        if args.clean_temp:
            pipeline.clean_temp()
            print("Temporary files cleaned.")
            return 0
        
        # Parse and validate steps
        if not args.steps:
            # Default to all stages except save
            steps = [s for s in NurdlePipeline.AVAILABLE_STAGES if s not in ['save']]
            print(f"No steps specified, running default stages: {', '.join(steps)}")
        else:
            steps = [s.strip() for s in args.steps.split(',')]
        
        # Validate steps
        invalid_steps = [s for s in steps if s not in NurdlePipeline.AVAILABLE_STAGES]
        if invalid_steps:
            print(f"ERROR: Invalid steps: {invalid_steps}")
            print(f"Available steps: {NurdlePipeline.AVAILABLE_STAGES}")
            return 1
        
        # Run pipeline
        results = pipeline.run(steps)
        
        print("=" * 80)
        print("PIPELINE EXECUTION SUMMARY")
        print("=" * 80)
        for stage, result in results.items():
            print(f"\n{stage.upper()}:")
            for key, value in result.items():
                print(f"  {key}: {value}")
        print("=" * 80)
        
        return 0
        
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user")
        return 1
        
    except Exception as e:
        print(f"\nPipeline failed: {e}")
        logging.exception("Full error trace:")
        return 1
    finally:
        # Ensure plotting and background workers are shut down so the process can exit cleanly
        _shutdown_plot_backends()
        _wait_for_background_threads(logging.getLogger('NurdlePipeline'))
        logging.shutdown()


if __name__ == "__main__":
    sys.exit(main())
