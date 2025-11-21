#!/usr/bin/env python3
"""
Nurdle Detection Pipeline - Modular Step-by-Step Execution
==========================================================

Modular pipeline with checkpoint-based stages and intermediate outputs.

Features:
- Step-by-step execution via --steps flag
- Checkpoint save/load for normalized data and windows
- Intermediate visualizations for each stage
- Progress tracking with tqdm
- Enhanced error checking and logging

Usage:
    python pipeline.py --steps normalization,windows,features,tuning,training,evaluation
    python pipeline.py --steps normalization              # Run only normalization
    python pipeline.py --steps training,evaluation        # Skip to training if checkpoints exist
    python pipeline.py --clean-temp                       # Clean temporary files

Author: Aqua Team Middelburg  
Version: 3.0.0 (Modular Pipeline)
"""

import argparse
import logging
import sys
import shutil
import cv2
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

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
    AVAILABLE_STAGES = ['normalization', 'windows', 'features', 'tuning', 'training', 'evaluation']
    
    def __init__(self, config_path: str):
        """Initialize pipeline with configuration."""
        self.config_path = config_path
        self.config = None
        self.logger = self._setup_logging()
        
        # Define checkpoint and output directories
        self.temp_dir = Path('temp')
        self.checkpoint_normalized = self.temp_dir / 'normalized'
        self.checkpoint_windows = self.temp_dir / 'candidate_windows'
        
        # Stage output directories
        self.output_dirs = {
            'normalization': Path('output/01_normalization'),
            'windows': Path('output/02_windows'),
            'features': Path('output/03_features'),
            'tuning': Path('output/04_tuning'),
            'training': Path('output/05_training'),
            'evaluation': Path('output/06_evaluation')
        }
        
        # Models directory
        self.models_dir = Path('output/models')
        
        # Initialize components (will be created as needed per stage)
        self.data_loader = None
        self.feature_extractor = None
        self.model_trainer = None
        self.evaluator = None
        
        self.logger.info("Modular pipeline initialized successfully")
    
    def _get_feature_config(self) -> Dict[str, Any]:
        """Get unified feature extractor configuration."""
        return {
            **self.config.windows,
            **self.config.training,
            'hog_cell_size': self.config.features.get('hog_cell_size', [4, 4])
        }
    
    def _load_checkpoints(self):
        """Load normalized data checkpoint and initialize feature extractor."""
        self.data_loader = DataLoader(self.config.data)
        self.data_loader.load_normalized_data(self.checkpoint_normalized)
        self.feature_extractor = FeatureExtractor(self._get_feature_config())
    
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
    
    def _reload_config(self):
        """Reload configuration from file."""
        self.logger.debug(f"Reloading configuration from {self.config_path}")
        self.config = load_config(self.config_path)
    
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
        - windows: Generate candidate windows, save checkpoints  
        - features: Extract and visualize features (no saving)
        - tuning: Hyperparameter optimization with Optuna
        - training: Train models and save metrics
        - evaluation: Comprehensive evaluation with all metrics
        
        Args:
            steps: List of stage names to execute
            
        Returns:
            Dictionary with results from executed stages
        """
        self.logger.info("=" * 80)
        self.logger.info(f"STARTING MODULAR PIPELINE - STEPS: {', '.join(steps)}")
        self.logger.info("=" * 80)
        
        results = {}
        
        try:
            for stage in steps:
                if stage not in self.AVAILABLE_STAGES:
                    raise ValueError(f"Unknown stage: {stage}. Available: {self.AVAILABLE_STAGES}")
                
                # Reload config before each stage
                self._reload_config()
                
                self.logger.info(f"\n{'='*80}")
                self.logger.info(f"STAGE: {stage.upper()}")
                self.logger.info(f"{'='*80}")
                
                # Execute stage
                if stage == 'normalization':
                    results[stage] = self._run_normalization_stage()
                elif stage == 'windows':
                    results[stage] = self._run_windows_stage()
                elif stage == 'features':
                    results[stage] = self._run_features_stage()
                elif stage == 'tuning':
                    results[stage] = self._run_tuning_stage()
                elif stage == 'training':
                    results[stage] = self._run_training_stage()
                elif stage == 'evaluation':
                    results[stage] = self._run_evaluation_stage()
                
                self.logger.info(f"Stage '{stage}' completed successfully")
        
        except Exception as e:
            self.logger.error(f"Pipeline failed at stage '{stage}': {e}", exc_info=True)
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
    
    def _run_windows_stage(self) -> Dict[str, Any]:
        """
        Stage 2: Generate candidate windows and save checkpoints.
        
        Prerequisites:
        - temp/normalized/ must exist
        
        Outputs:
        - temp/candidate_windows/ - Checkpoint with window metadata
        - output/02_windows/ - Sample window visualizations
        """
        # Check prerequisites
        if not self._check_checkpoint_exists(self.checkpoint_normalized, 'Windows', 'normalization'):
            raise FileNotFoundError(f"ERROR: Windows - Normalized data not found. Run --steps normalization first.")
        
        output_dir = self._create_stage_output_dir('windows')
        
        # Load checkpoints
        self.logger.info("Loading checkpoints...")
        self._load_checkpoints()
        
        # Generate windows for all training images
        self.logger.info("Generating candidate windows...")
        all_windows = []
        
        for annotation in tqdm(self.data_loader.train_annotations, desc="Processing images"):
            image = self.data_loader.load_image(annotation.image_path)
            windows = self.feature_extractor.generate_windows_for_image(image, annotation)
            all_windows.extend(windows)
        
        # Save windows checkpoint (implemented in step 3)
        self.logger.info("Saving windows checkpoint...")
        self.feature_extractor.save_windows(all_windows, self.checkpoint_windows)
        
        # Generate sample visualizations
        self.logger.info("Generating window visualizations...")
        self.feature_extractor.visualize_windows_samples(
            all_windows, self.data_loader, output_dir, num_samples=3
        )
        
        return {
            'total_windows': len(all_windows),
            'checkpoint': str(self.checkpoint_windows)
        }
    
    def _run_features_stage(self) -> Dict[str, Any]:
        """
        Stage 3: Extract and visualize features.
        
        Prerequisites:
        - temp/normalized/ must exist
        - temp/candidate_windows/ must exist
        
        Outputs:
        - output/03_features/ - Feature visualizations (HOG overlays, LBP patterns)
        
        Note: Features are NOT saved, training uses generator.
        """
        # Check prerequisites
        if not self._check_checkpoint_exists(self.checkpoint_normalized, 'Features', 'normalization'):
            raise FileNotFoundError(f"ERROR: Features - Normalized data not found. Run --steps normalization first.")
        if not self._check_checkpoint_exists(self.checkpoint_windows, 'Features', 'windows'):
            raise FileNotFoundError(f"ERROR: Features - Window data not found. Run --steps windows first.")
        
        output_dir = self._create_stage_output_dir('features')
        
        # Load checkpoints
        self.logger.info("Loading checkpoints...")
        self._load_checkpoints()
        
        # Generate feature visualizations
        self.logger.info("Generating feature visualizations...")
        windows = self.feature_extractor.load_windows(self.checkpoint_windows)
        self.feature_extractor.visualize_features(windows, self.data_loader, output_dir)
        
        return {
            'visualization_dir': str(output_dir)
        }
    
    def _run_tuning_stage(self) -> Dict[str, Any]:
        """
        Stage 4: Hyperparameter optimization with Optuna.
        
        Prerequisites:
        - temp/normalized/ must exist
        - temp/candidate_windows/ must exist
        
        Outputs:
        - output/04_tuning/svm/ - SVM optimization results
        - output/04_tuning/svr/ - SVR optimization results
        """
        # Check prerequisites
        if not self._check_checkpoint_exists(self.checkpoint_normalized, 'Tuning', 'normalization'):
            raise FileNotFoundError(f"ERROR: Tuning - Normalized data not found. Run --steps normalization first.")
        if not self._check_checkpoint_exists(self.checkpoint_windows, 'Tuning', 'windows'):
            raise FileNotFoundError(f"ERROR: Tuning - Window data not found. Run --steps windows first.")
        
        output_dir = self._create_stage_output_dir('tuning')
        
        # Load checkpoints
        self.logger.info("Loading checkpoints...")
        self.data_loader = DataLoader(self.config.data)
        self.data_loader.load_normalized_data(self.checkpoint_normalized)
        
        feature_config = {
            **self.config.windows,
            **self.config.training,
            'hog_cell_size': self.config.features.get('hog_cell_size', [4, 4])
        }
        self.feature_extractor = FeatureExtractor(feature_config)
        
        # Run Optuna optimization with sequential strategy (SVM then SVR)
        self.logger.info("Running hyperparameter optimization...")
        from src.optuna import OptunaTuner
        
        # Create training function for Optuna
        def train_and_evaluate(params: dict) -> dict:
            """Train models with given parameters and return metrics."""
            # Temporarily suppress verbose logging during trials
            import logging
            prev_levels = {}
            for logger_name in ['src.features.feature_extractor', 'src.models.model_trainer']:
                logger = logging.getLogger(logger_name)
                prev_levels[logger_name] = logger.level
                logger.setLevel(logging.WARNING)
            
            try:
                # Create trial config
                trial_config = self.config._config.copy()
                
                # Update model parameters
                if 'model' not in trial_config:
                    trial_config['model'] = {}
                trial_config['model'].update(params)
                
                # Create model trainer with trial config
                from src.models import ModelTrainer
                trial_trainer = ModelTrainer(trial_config)
                
                # Train models
                window_batches = self.feature_extractor.generate_training_windows_batch(
                    self.data_loader.train_annotations,
                    self.data_loader.load_image
                )
                training_metrics = trial_trainer.train_models(window_batches)
                
                # Return actual metrics from training
                return {
                    'f1_score': training_metrics['svm']['f1_score'],
                    'avg_coordinate_error': training_metrics['svr']['mean_error']
                }
            finally:
                # Restore logging levels
                for logger_name, level in prev_levels.items():
                    logging.getLogger(logger_name).setLevel(level)
        
        # Initialize tuner
        optuna_config = self.config._config.get('optimization', {})
        tuner = OptunaTuner(optuna_config, self.logger)
        
        # Run sequential optimization
        best_params = tuner.optimize_sequential(train_and_evaluate, str(output_dir))
        
        return {
            'best_params': best_params,
            'output_dir': str(output_dir)
        }
    
    def _run_training_stage(self) -> Dict[str, Any]:
        """
        Stage 5: Train models and collect metrics.
        
        Prerequisites:
        - temp/normalized/ must exist
        - temp/candidate_windows/ must exist
        
        Outputs:
        - output/05_training/{model_name}/ - Training metrics and curves
        - output/models/ - Saved models
        """
        # Check prerequisites
        if not self._check_checkpoint_exists(self.checkpoint_normalized, 'Training', 'normalization'):
            raise FileNotFoundError(f"ERROR: Training - Normalized data not found. Run --steps normalization first.")
        if not self._check_checkpoint_exists(self.checkpoint_windows, 'Training', 'windows'):
            raise FileNotFoundError(f"ERROR: Training - Window data not found. Run --steps windows first.")
        
        output_dir = self._create_stage_output_dir('training')
        
        # Load checkpoints
        self.logger.info("Loading checkpoints...")
        self.data_loader = DataLoader(self.config.data)
        self.data_loader.load_normalized_data(self.checkpoint_normalized)
        
        feature_config = {
            **self.config.windows,
            **self.config.training,
            'hog_cell_size': self.config.features.get('hog_cell_size', [4, 4])
        }
        self.feature_extractor = FeatureExtractor(feature_config)
        
        # Load tuning results if available
        tuning_dir = self.output_dirs.get('tuning')
        # Initialize model trainer
        training_config = self.config._config.copy()
        self.model_trainer = ModelTrainer(training_config)
        
        # Load optimized hyperparameters if available
        best_params = self.model_trainer.load_tuning_results(tuning_dir) if tuning_dir and tuning_dir.exists() else {}
        
        if best_params:
            if 'model' not in training_config:
                training_config['model'] = {}
            training_config['model'].update(best_params)
            # Reinitialize trainer with updated config
            self.model_trainer = ModelTrainer(training_config)
        
        # Train models with metrics collection (implemented in step 6)
        self.logger.info("Training models...")
        window_batches = self.feature_extractor.generate_training_windows_batch(
            self.data_loader.train_annotations,
            self.data_loader.load_image
        )
        
        training_metrics = self.model_trainer.train_models(window_batches)
        
        # Save models
        self.logger.info("Saving models...")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.model_trainer.save_models(str(self.models_dir))
        
        # Save training metrics
        self.logger.info("Saving training metrics...")
        self.model_trainer.save_training_metrics(training_metrics, str(output_dir))
        
        return {
            'training_metrics': training_metrics,
            'models_dir': str(self.models_dir),
            'output_dir': str(output_dir)
        }
    
    def _run_evaluation_stage(self) -> Dict[str, Any]:
        """
        Stage 6: Comprehensive evaluation of trained models.
        
        Prerequisites:
        - temp/normalized/ must exist
        - output/models/ must have trained models
        
        Outputs:
        - output/06_evaluation/ - Evaluation metrics, visualizations, and report
        """
        # Check prerequisites
        if not self._check_checkpoint_exists(self.checkpoint_normalized, 'Evaluation', 'normalization'):
            raise FileNotFoundError(f"ERROR: Evaluation - Normalized data not found. Run --steps normalization first.")
        
        if not self.models_dir.exists() or not any(self.models_dir.glob('*.pkl')):
            raise FileNotFoundError(f"ERROR: Evaluation - No trained models found in {self.models_dir}. Run --steps training first.")
        
        output_dir = self._create_stage_output_dir('evaluation')
        
        # Load test data
        self.logger.info("Loading test data...")
        self._load_checkpoints()
        self.model_trainer = ModelTrainer(self.config._config)
        
        # Load trained models
        self.logger.info(f"Loading models from {self.models_dir}...")
        self.model_trainer.load_models(str(self.models_dir))
        
        # Initialize evaluator, NMS, and prediction engine
        from src.evaluation.evaluator import ModelEvaluator
        from src.evaluation.nms import NonMaximumSuppression
        from src.models import PredictionEngine
        
        self.evaluator = ModelEvaluator(self.logger)
        nms_config = self.config._config.get('evaluation', {}).get('nms', {})
        nms_distance = nms_config.get('distance_threshold', 20.0)
        self.nms = NonMaximumSuppression(default_min_distance=nms_distance)
        
        # Create prediction engine
        prediction_engine = PredictionEngine(
            self.feature_extractor,
            self.model_trainer,
            self.nms
        )
        
        # Create prediction function
        def predict_image_with_nms(image_path: str):
            """Predict nurdles for an image using prediction engine."""
            count, detections = prediction_engine.predict_image_from_path(
                image_path,
                self.data_loader,
                nms_distance
            )
            return count, detections
        
        # Run evaluation
        self.logger.info(f"Evaluating on {len(self.data_loader.test_annotations)} test images...")
        metrics = self.evaluator.evaluate_models(
            test_annotations=self.data_loader.test_annotations,
            predict_image_func=predict_image_with_nms
        )
        
        # Save evaluation results
        self.logger.info("Saving evaluation results...")
        self.evaluator.save_evaluation_results(metrics, str(output_dir))
        
        # Generate evaluation report
        self.logger.info("Generating evaluation report...")
        report_path = self.evaluator.generate_evaluation_report(metrics, str(output_dir))
        
        # Generate visualizations
        self.logger.info("Generating evaluation visualizations...")
        viz_dir = output_dir / 'visualizations'
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        self.evaluator.generate_evaluation_visualizations(
            test_annotations=self.data_loader.test_annotations,
            predict_image_func=predict_image_with_nms,
            data_loader=self.data_loader,
            metrics=metrics,
            output_dir=str(viz_dir)
        )
        
        return {
            'metrics': metrics,
            'report_path': report_path,
            'output_dir': str(output_dir),
            'visualizations_dir': str(viz_dir)
        }
    
    def clean_temp(self):
        """Clean temporary checkpoint files."""
        self.logger.info("Cleaning temporary files...")
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            self.logger.info(f"Removed {self.temp_dir}")
        self.temp_dir.mkdir(parents=True, exist_ok=True)


def setup_logging():
    """Setup basic logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Modular Nurdle Detection Pipeline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--config', 
        type=str, 
        default='config.yaml',
        help='Path to configuration YAML file'
    )
    
    parser.add_argument(
        '--steps',
        type=str,
        default=None,
        help='Comma-separated list of stages to run (default: all stages in order). Options: normalization,windows,features,tuning,training,evaluation'
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
            # Default to all stages in order
            steps = NurdlePipeline.AVAILABLE_STAGES.copy()
            print(f"No steps specified, running all stages: {', '.join(steps)}")
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
        
        print("\n" + "=" * 80)
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


if __name__ == "__main__":
    sys.exit(main())
