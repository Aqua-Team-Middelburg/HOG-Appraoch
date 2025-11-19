#!/usr/bin/env python3
"""
Nurdle Detection Pipeline - Main Entry Point
===========================================

Clean, organized pipeline for nurdle detection and localization.

Key features:
- RAM-friendly batch processing (15 images at a time)
- Two-stage detection: SVM Classifier → SVR Regressor
- Combined HOG+LBP features
- Optional stacking meta-learner for improved accuracy
- Modular, maintainable code structure

Usage:
    python pipeline.py                    # Run with default config.yaml
    python pipeline.py --config custom.yaml
    python pipeline.py --input input_dir --output output_dir

Author: Aqua Team Middelburg  
Version: 2.0.0 (Refactored)
"""

import argparse
import logging
import sys
import numpy as np
from pathlib import Path
from typing import Dict, Any

# Import pipeline components
try:
    from src.utils.config import load_config
    from src.utils.visualizer import PipelineVisualizer
    from src.data import DataLoader
    from src.features import FeatureExtractor
    from src.models import ModelTrainer, StackedNurdleDetector
    from src.evaluation import ModelEvaluator
except ImportError as e:
    print(f"Error importing pipeline components: {e}")
    print("Make sure all src modules are properly installed")
    sys.exit(1)


class NurdlePipeline:
    """
    Main pipeline orchestrator for nurdle detection.
    
    Coordinates data loading, feature extraction, model training, and evaluation.
    """
    
    def __init__(self, config_path: str):
        """Initialize pipeline with configuration."""
        self.config = load_config(config_path)
        self.logger = self._setup_logging()
        
        # Initialize components with proper configuration
        self.data_loader = DataLoader(self.config.data)
        
        # Combine window and training configs for feature extractor
        feature_config = {
            **self.config.windows,
            **self.config.training,
            'hog_cell_size': self.config.features.get('hog_cell_size', [4, 4])
        }
        self.feature_extractor = FeatureExtractor(feature_config)
        
        # ModelTrainer needs full config dict (not just training section)
        # because it accesses confidence.svr_method and other cross-section params
        self.model_trainer = ModelTrainer(self.config._config)
        self.evaluator = ModelEvaluator(self.logger)
        self.visualizer = PipelineVisualizer(self.logger)
        
        # Stacked model will be initialized after base models are trained
        self.stacked_model = None
        
        self.logger.info("Pipeline initialized successfully")
    
    def _setup_logging(self):
        """Setup pipeline-specific logging."""
        logs_dir = Path('output/logs')
        logs_dir.mkdir(parents=True, exist_ok=True)
        
        logger = logging.getLogger('NurdlePipeline')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            # Console handler
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.INFO)
            
            # File handler
            file_handler = logging.FileHandler(logs_dir / 'pipeline.log')
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
    
    def run_full_pipeline(self, input_dir: str, output_dir: str) -> dict:
        """
        Run the complete nurdle detection pipeline - Phase 2 Implementation.
        
        Single command runs everything:
        1. Image Normalization - Single resolution target  
        2. Train/Test Split - Simple ratio-based splitting
        3. Smart Sliding Windows - RAM-friendly batch processing
        4. Window Labeling - Binary classification with coordinate offsets
        5. Combined Feature Extraction - RGB HOG + LBP only
        6. Balanced Sampling - Distance-based negative selection
        7. Two-Stage Model Training - SVM classifier → SVR regressor
        8. Evaluation - Standard metrics + coordinate accuracy
        
        Args:
            input_dir: Directory containing images and JSON annotations
            output_dir: Directory to save results and models
            
        Returns:
            Evaluation metrics dictionary
        """
        self.logger.info("=" * 60)
        self.logger.info("STARTING NURDLE DETECTION PIPELINE")
        self.logger.info("=" * 60)
        
        try:
            # Step 1: Load and split data
            self.logger.info("Step 1: Loading data and annotations...")
            self.data_loader.load_annotations(input_dir)
            
            if self.data_loader.num_annotations == 0:
                raise ValueError("No annotations loaded from input directory")
                
            self.data_loader.split_train_test()
            self.logger.info(f"Loaded {self.data_loader.num_annotations} images: "
                           f"{self.data_loader.num_train} train, {self.data_loader.num_test} test")
            
            # Step 2: Extract features and train models (RAM-efficient batch processing)
            self.logger.info("Step 2: Extracting features and training models...")
            self.extract_features_and_train()
            
            # Step 3: Optionally train stacked model
            stacking_enabled = self.config.get('stacking').get('enabled', False)
            if stacking_enabled:
                self.logger.info("Step 3: Training stacked ensemble model...")
                self.train_stacked_model()
            else:
                self.logger.info("Step 3: Stacking disabled, skipping ensemble training...")
            
            # Step 4: Evaluate models
            self.logger.info("Step 4: Evaluating trained models...")
            if stacking_enabled:
                metrics = self.evaluate_both_models()
            else:
                metrics = self.evaluate_main_model()
            
            # Step 5: Save results
            self.logger.info("Step 5: Saving models and results...")
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save models
            self.model_trainer.save_models(output_dir)
            
            # Save stacked model if trained
            if self.stacked_model and hasattr(self.stacked_model, 'is_trained') and self.stacked_model.is_trained:
                import joblib
                model_path = output_path / "models" / "stacked_model.joblib"
                model_path.parent.mkdir(parents=True, exist_ok=True)
                joblib.dump(self.stacked_model, str(model_path))
                self.logger.info(f"Stacked model saved to {model_path}")
            
            # Save evaluation results
            self.evaluator.save_evaluation_results(metrics, output_dir)
            self.evaluator.generate_evaluation_report(metrics, output_dir)
            
            # Step 6: Generate visualizations
            self.logger.info("Step 6: Generating prediction visualizations...")
            self.generate_visualizations(output_dir, stacking_enabled)
            
            self.logger.info("=" * 60)
            self.logger.info("PIPELINE COMPLETED SUCCESSFULLY")
            self.logger.info("=" * 60)
            
            # Log results based on stacking mode
            if stacking_enabled:
                self._log_comparison_results(metrics)
            else:
                self._log_final_results(metrics.get('main_pipeline', metrics))
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            raise
    
    def generate_visualizations(self, output_dir: str, stacking_enabled: bool):
        """Generate prediction visualizations."""
        # Convert test annotations to format expected by visualizer
        test_data = {}
        for annotation in self.data_loader.test_annotations:
            test_data[annotation.image_path] = {
                'nurdles': [{'center_x': n.x, 'center_y': n.y} for n in annotation.nurdles]
            }
        
        # Create prediction function
        def predict_func(image_path):
            return self.predict_image(image_path, use_stacked=False)
        
        # Generate visualizations - visualizer handles folder structure
        self.visualizer.create_batch_visualizations(
            test_data,
            predict_func,
            output_dir  # Pass output_dir directly, not output_dir/visualizations
        )
    
    def _log_comparison_results(self, metrics: dict):
        """Log comparison results for stacking mode."""
        main = metrics.get('main_pipeline', {})
        stacked = metrics.get('stacked_model', {})
        comp = metrics.get('comparison', {})
        
        self.logger.info("MAIN PIPELINE RESULTS:")
        self.logger.info(f"  F1-Score: {main.get('f1_score', 0):.3f}")
        self.logger.info(f"  Precision: {main.get('precision', 0):.3f}")
        self.logger.info(f"  Recall: {main.get('recall', 0):.3f}")
        self.logger.info(f"  Coordinate Error: {main.get('avg_coordinate_error', 0):.1f} pixels")
        self.logger.info(f"  Count MAE: {main.get('count_mae', 0):.3f}")
        
        self.logger.info("STACKED MODEL RESULTS:")
        self.logger.info(f"  F1-Score: {stacked.get('f1_score', 0):.3f}")
        self.logger.info(f"  Precision: {stacked.get('precision', 0):.3f}")
        self.logger.info(f"  Recall: {stacked.get('recall', 0):.3f}")
        self.logger.info(f"  Coordinate Error: {stacked.get('avg_coordinate_error', 0):.1f} pixels")
        self.logger.info(f"  Count MAE: {stacked.get('count_mae', 0):.3f}")
        
        self.logger.info("IMPROVEMENTS:")
        self.logger.info(f"  F1-Score: {comp.get('f1_improvement', 0):+.3f}")
        self.logger.info(f"  Coordinate Error: {comp.get('coord_error_improvement', 0):+.1f} pixels")
    
    def extract_features_and_train(self):
        """
        RAM-efficient: Process images in batches, train incrementally.
        
        This implements the core two-stage detection approach:
        - Process training images in memory-friendly batches
        - Extract combined HOG+LBP features  
        - Generate balanced positive/negative windows
        - Train SVM classifier and SVR regressor incrementally
        """
        # Process training annotations in batches for memory efficiency
        window_batches = self.feature_extractor.generate_training_windows_batch(
            self.data_loader.train_annotations,
            self.data_loader.load_image
        )
        
        # Train models on batches
        self.model_trainer.train_models(window_batches)
        
        self.logger.info(f"Feature extraction and training completed")
    
    def train_stacked_model(self):
        """Train stacked model using base model predictions."""
        try:
            # Generate training windows again (or cache from training)
            window_batches = self.feature_extractor.generate_training_windows_batch(
                self.data_loader.train_annotations,
                self.data_loader.load_image
            )
            
            # Collect all windows
            all_windows = []
            for batch in window_batches:
                all_windows.extend(batch)
            
            # Initialize stacked model
            self.stacked_model = StackedNurdleDetector(
                self.model_trainer,
                self.config._config,
                self.logger
            )
            
            # Train meta-learner
            self.stacked_model.train_stacking_model(all_windows)
            
            self.logger.info("Stacked model training completed")
            
        except Exception as e:
            self.logger.error(f"Stacked model training failed: {e}")
            self.stacked_model = None
    
    def evaluate_main_model(self) -> dict:
        """Evaluate main pipeline only."""
        def predict_func(image_path):
            return self.predict_image(image_path, use_stacked=False)
        
        metrics = self.evaluator.evaluate_models(
            test_annotations=self.data_loader.test_annotations,
            predict_image_func=predict_func
        )
        
        return {'main_pipeline': metrics}
    
    def evaluate_both_models(self) -> dict:
        """Evaluate both main and stacked models for comparison."""
        # Evaluate main pipeline
        def predict_main(image_path):
            return self.predict_image(image_path, use_stacked=False)
        
        main_metrics = self.evaluator.evaluate_models(
            test_annotations=self.data_loader.test_annotations,
            predict_image_func=predict_main
        )
        
        # Evaluate stacked model
        def predict_stacked(image_path):
            return self.predict_image(image_path, use_stacked=True)
        
        stacked_metrics = self.evaluator.evaluate_models(
            test_annotations=self.data_loader.test_annotations,
            predict_image_func=predict_stacked
        )
        
        # Calculate improvements
        comparison = {
            'f1_improvement': stacked_metrics['f1_score'] - main_metrics['f1_score'],
            'coord_error_improvement': main_metrics['avg_coordinate_error'] - stacked_metrics['avg_coordinate_error'],
            'precision_improvement': stacked_metrics['precision'] - main_metrics['precision'],
            'recall_improvement': stacked_metrics['recall'] - main_metrics['recall'],
        }
        
        return {
            'main_pipeline': main_metrics,
            'stacked_model': stacked_metrics,
            'comparison': comparison
        }
    
    def predict_image(self, image_path: str, use_stacked: bool = False):
        """
        Predict nurdle count and coordinates using two-stage detection.
        
        Args:
            image_path: Path to image file
            use_stacked: If True, use stacked model; else use main pipeline
            
        Returns:
            Tuple of (nurdle_count, coordinates_list)
        """
        # Load original image to get dimensions
        import cv2
        original_image = cv2.imread(image_path)
        if original_image is None:
            raise ValueError(f"Could not load image: {image_path}")
        original_h, original_w = original_image.shape[:2]
        
        # Load and normalize image for processing
        normalized_image = self.data_loader.load_image(image_path)
        normalized_h, normalized_w = normalized_image.shape[:2]
        
        # Calculate scale factors to convert back to original coordinates
        scale_x = original_w / normalized_w
        scale_y = original_h / normalized_h
        
        # Extract features from all sliding windows
        detections = []
        
        for features, x, y in self.feature_extractor.extract_sliding_window_features(normalized_image):
            # Calculate window center in normalized coordinates
            window_center_x = x + self.feature_extractor.window_size[0] / 2.0
            window_center_y = y + self.feature_extractor.window_size[1] / 2.0
            
            # Get prediction based on model type
            if use_stacked and self.stacked_model and self.stacked_model.is_trained:
                # Use stacked model
                is_nurdle, confidence, offset_x, offset_y = \
                    self.stacked_model.predict_with_confidence(features)
            else:
                # Use main pipeline (SVM + SVR)
                is_nurdle, svm_conf, offset_x, offset_y, svr_conf = \
                    self.model_trainer.predict_with_confidence(features)
                confidence = svm_conf  # Use SVM confidence for filtering
            
            if is_nurdle:
                # Refine coordinates using SVR offset prediction
                refined_x = window_center_x + offset_x
                refined_y = window_center_y + offset_y
                
                # Scale back to original image coordinates
                original_x = refined_x * scale_x
                original_y = refined_y * scale_y
                
                detections.append((original_x, original_y, confidence))
        
        # Apply Non-Maximum Suppression to remove duplicates
        from src.evaluation import NonMaximumSuppression
        nms_distance = int(self.feature_extractor.window_size[0] * scale_x)
        nms = NonMaximumSuppression(default_min_distance=nms_distance)
        filtered_detections = nms.apply_nms(detections)
        
        # Extract final coordinates and count
        coordinates = [(d[0], d[1]) for d in filtered_detections]
        nurdle_count = len(coordinates)
        
        return nurdle_count, coordinates
    
    def _predict_for_optuna(self, image_path: str, feature_extractor, model_trainer):
        """
        Helper method for Optuna optimization - predicts using trial models.
        
        Args:
            image_path: Path to image
            feature_extractor: Trial feature extractor
            model_trainer: Trial model trainer
            
        Returns:
            Tuple of (count, coordinates)
        """
        # Load and normalize image
        image = self.data_loader.load_image(image_path)
        
        # Extract features and predict
        detections = []
        threshold = self.config.get('confidence').get('svm_threshold', 0.5)
        
        for features, x, y in feature_extractor.extract_sliding_window_features(image):
            is_nurdle, svm_conf, offset_x, offset_y, _ = model_trainer.predict_with_confidence(features)
            
            if is_nurdle and svm_conf > threshold:
                center_x = x + feature_extractor.window_size[0] / 2.0
                center_y = y + feature_extractor.window_size[1] / 2.0
                refined_x = center_x + offset_x
                refined_y = center_y + offset_y
                detections.append((refined_x, refined_y, svm_conf))
        
        # Apply NMS
        from src.evaluation import NonMaximumSuppression
        nms = NonMaximumSuppression(default_min_distance=feature_extractor.window_size[0])
        filtered = nms.apply_nms(detections)
        
        coordinates = [(d[0], d[1]) for d in filtered]
        return len(coordinates), coordinates
    
    def run_optimization(self, input_dir: str, output_dir: str, 
                        n_trials: int = 20, metric: str = 'f1_score'):
        """
        Run hyperparameter optimization using Optuna (Phase 8).
        
        Args:
            input_dir: Input directory containing images and annotations
            output_dir: Output directory for results and optimized models
            n_trials: Number of optimization trials
            metric: Metric to optimize ('f1_score', 'precision', 'recall', 'count_accuracy')
            
        Returns:
            Dictionary of best parameters found
        """
        self.logger.info("=" * 60)
        self.logger.info("STARTING HYPERPARAMETER OPTIMIZATION (PHASE 8)")
        self.logger.info("=" * 60)
        
        try:
            # Import Optuna components
            from src.optuna import OptunaTuner
            
            # Setup directories
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Load data if not already loaded
            if not hasattr(self, 'data_loader') or not self.data_loader:
                config_dict = self.config.load() if self.config._config is None else self.config._config
                self.data_loader = DataLoader(config_dict)
                self.data_loader.load_annotations(input_dir)
            
            # Configure Optuna optimization
            optuna_config = {
                'n_trials': n_trials,
                'metric': metric,
                'direction': 'maximize',
                'save_plots': True
            }
            
            # Initialize tuner
            tuner = OptunaTuner(optuna_config, self.logger)
            
            # Create training function for optimization
            def train_func(params: Dict[str, Any]) -> Dict[str, float]:
                """Train and evaluate with given parameters."""
                # Update config with trial parameters
                trial_config = self.config._config.copy() if self.config._config else {}
                trial_config['model']['svm'].update({
                    'C': params.get('svm_c', 1.0),
                    'kernel': params.get('svm_kernel', 'linear'),
                    'gamma': params.get('svm_gamma', 'scale')
                })
                trial_config['model']['svr'].update({
                    'C': params.get('svr_c', 1.0),
                    'kernel': params.get('svr_kernel', 'linear'),
                    'gamma': params.get('svr_gamma', 'scale')
                })
                
                # Train models with trial parameters
                feature_extractor = FeatureExtractor(trial_config['features'])
                model_trainer = ModelTrainer(trial_config['model'])
                
                window_batches = feature_extractor.generate_training_windows_batch(
                    self.data_loader.train_annotations,
                    self.data_loader.load_image
                )
                model_trainer.train_models(window_batches)
                
                # Evaluate on test set
                evaluator = ModelEvaluator(self.logger)
                metrics = evaluator.evaluate_models(
                    self.data_loader.test_annotations,
                    lambda img_path: self._predict_for_optuna(img_path, feature_extractor, model_trainer)
                )
                return metrics
            
            # Run sequential optimization
            self.logger.info(f"Optimizing {metric} over {n_trials} trials (sequential strategy)...")
            best_params = tuner.optimize_sequential(
                train_func,
                str(output_path)
            )
            
            self.logger.info("=" * 60)
            self.logger.info("OPTIMIZATION COMPLETED SUCCESSFULLY")
            self.logger.info("=" * 60)
            self.logger.info(f"Best {metric}: {tuner.best_value:.4f}")
            self.logger.info("Best parameters:")
            for key, value in best_params.items():
                self.logger.info(f"  {key}: {value}")
            
            # Optionally retrain with best parameters and evaluate
            optimization_config = self.config.get('optimization')
            if optimization_config.get('retrain_with_best', True):
                self.logger.info("\nRetraining pipeline with optimized parameters...")
                final_metrics = train_func(best_params)
                self._log_final_results(final_metrics)
                
                return best_params, final_metrics
            
            return best_params, None
            
        except ImportError as e:
            self.logger.error(f"Optuna not available: {e}")
            self.logger.error("Install optuna and plotly: pip install optuna plotly")
            raise
            
        except Exception as e:
            self.logger.error(f"Optimization failed: {e}")
            raise
    
    def _log_final_results(self, metrics: dict):
        """Log final pipeline results."""
        self.logger.info("FINAL RESULTS:")
        self.logger.info(f"  Count Accuracy: {metrics.get('count_accuracy', 0):.3f}")
        self.logger.info(f"  Count MAE: {metrics.get('count_mae', 0):.3f}")
        self.logger.info(f"  Coordinate Error: {metrics.get('avg_coordinate_error', 0):.1f} pixels")
        self.logger.info(f"  Precision: {metrics.get('precision', 0):.3f}")
        self.logger.info(f"  Recall: {metrics.get('recall', 0):.3f}")
        self.logger.info(f"  F1-Score: {metrics.get('f1_score', 0):.3f}")


def setup_logging():
    """Setup basic logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Nurdle Detection Pipeline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--config', 
        type=str, 
        default='config.yaml',
        help='Path to configuration YAML file'
    )
    
    parser.add_argument(
        '--input',
        type=str,
        help='Input directory (overrides config)'
    )
    
    parser.add_argument(
        '--output', 
        type=str,
        help='Output directory (overrides config)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    # Phase 8: Optuna optimization arguments
    parser.add_argument(
        '--optimize',
        action='store_true',
        help='Run hyperparameter optimization using Optuna'
    )
    
    parser.add_argument(
        '--n-trials',
        type=int,
        default=20,
        help='Number of optimization trials for Optuna'
    )
    
    parser.add_argument(
        '--metric',
        type=str,
        default='f1_score',
        choices=['f1_score', 'precision', 'recall', 'count_accuracy'],
        help='Metric to optimize'
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
        
        # Get directories
        input_dir = args.input or pipeline.config.data.get('input_dir', 'input')
        output_dir = args.output or pipeline.config.data.get('output_dir', 'output')
        
        # Check if we should run optimization (Phase 8)
        if args.optimize:
            print("\n" + "=" * 60)
            print("RUNNING HYPERPARAMETER OPTIMIZATION (PHASE 8)")
            print("=" * 60)
            
            # Run optimization
            best_params, final_metrics = pipeline.run_optimization(
                input_dir, output_dir, args.n_trials, args.metric
            )
            
            # Print optimization results
            print("\n" + "=" * 60)
            print("OPTIMIZATION COMPLETED")
            print("=" * 60)
            # Best metric value is already logged by the optimization method
            print("Best Parameters:")
            for key, value in best_params.items():
                print(f"  {key}: {value}")
            
            if final_metrics:
                print("\nFinal Model Performance:")
                print(f"  Count Accuracy: {final_metrics['count_accuracy']:.3f}")
                print(f"  Count MAE: {final_metrics['count_mae']:.3f}")
                print(f"  Avg Coordinate Error: {final_metrics['avg_coordinate_error']:.1f} pixels")
                print(f"  Precision: {final_metrics['precision']:.3f}")
                print(f"  Recall: {final_metrics['recall']:.3f}")
                print(f"  F1-Score: {final_metrics['f1_score']:.3f}")
                
            print("=" * 60)
            
        else:
            # Run standard pipeline
            metrics = pipeline.run_full_pipeline(input_dir, output_dir)
            
            # Print final results
            print("\n" + "=" * 50)
            print("FINAL RESULTS:")
            print(f"  Count Accuracy: {metrics['count_accuracy']:.3f}")
            print(f"  Count MAE: {metrics['count_mae']:.3f}")  
            print(f"  Avg Coordinate Error: {metrics['avg_coordinate_error']:.1f} pixels")
            print(f"  Test Images: {metrics['n_test_images']}")
            print("=" * 50)
        
        return 0
        
    except KeyboardInterrupt:
        print("Pipeline interrupted by user")
        return 1
        
    except Exception as e:
        print(f"Pipeline failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())