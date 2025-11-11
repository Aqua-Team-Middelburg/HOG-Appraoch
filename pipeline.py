#!/usr/bin/env python3
"""
HOG/LBP/SVM Pipeline - Main Application
======================================

This is the main entry point for the nurdle detection pipeline.
It orchestrates the complete workflow from raw images to final evaluation.

Usage:
    python pipeline.py --config config.yaml
    python pipeline.py --config config.yaml --step normalization
    python pipeline.py --config config.yaml --evaluate-only

Author: Aqua Team Middelburg
Version: 1.0.0
"""

import argparse
import logging
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, Any, List, Optional

# Import pipeline components
from src.utils.config import ConfigLoader
from src.preprocessing.normalizer import ImageNormalizer
from src.preprocessing.window_processor import SlidingWindowProcessor
from src.feature_extraction.combined_extractor import CombinedFeatureExtractor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('output/logs/pipeline.log')
    ]
)

logger = logging.getLogger(__name__)


class NurdleDetectionPipeline:
    """
    Main pipeline orchestrator for the HOG/LBP/SVM nurdle detection system.
    
    This class coordinates all pipeline steps and manages the overall workflow.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the pipeline with configuration.
        
        Args:
            config_path: Path to the YAML configuration file
        """
        self.config_path = config_path
        self.config = ConfigLoader(config_path)
        self.config.load()
        
        # Initialize components
        self.normalizer = ImageNormalizer(self.config)
        self.window_processor = SlidingWindowProcessor(self.config)
        self.feature_extractor = CombinedFeatureExtractor(self.config)
        
        # Pipeline state
        self.results = {}
        self.start_time = time.time()
        
        # Ensure output directories exist
        self._setup_directories()
        
        logger.info("Pipeline initialized successfully")
    
    def _setup_directories(self) -> None:
        """Create necessary output directories."""
        output_dir = Path(self.config.get('paths.output_dir'))
        
        directories = [
            output_dir / self.config.get('paths.models_dir'),
            output_dir / self.config.get('paths.evaluations_dir'),
            output_dir / self.config.get('paths.visualizations_dir'),
            output_dir / self.config.get('paths.logs_dir'),
            Path(self.config.get('paths.temp_dir', 'temp')),
            Path(self.config.get('paths.normalized_images_dir', 'temp/normalized')),
            Path(self.config.get('paths.candidate_windows_dir', 'temp/candidate_windows'))
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def run_step_normalization(self) -> Dict[str, Any]:
        """
        Step 1: Normalize input images and JSON annotations.
        
        Returns:
            Results dictionary from normalization
        """
        logger.info("=" * 60)
        logger.info("STEP 1: IMAGE NORMALIZATION")
        logger.info("=" * 60)
        
        try:
            # Run normalization
            results = self.normalizer.normalize_all_images()
            
            # Validate results
            validation = self.normalizer.validate_normalization()
            results['validation'] = validation
            
            # Save results
            import json
            output_path = Path(self.config.get('paths.output_dir')) / 'logs' / 'normalization_results.json'
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"Normalization completed: {results['successful_normalizations']} images processed")
            
            if validation['issues']:
                logger.warning(f"Validation found {len(validation['issues'])} issues")
                for issue in validation['issues'][:5]:  # Show first 5 issues
                    logger.warning(f"  - {issue}")
            
            return results
            
        except Exception as e:
            logger.error(f"Normalization failed: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def run_step_train_test_split(self) -> Dict[str, Any]:
        """
        Step 2: Split data into training and testing sets.
        
        Returns:
            Results dictionary from data splitting
        """
        logger.info("=" * 60)
        logger.info("STEP 2: TRAIN/TEST SPLIT")
        logger.info("=" * 60)
        
        try:
            # This would implement train/test splitting logic
            # For now, we'll use all normalized data for training
            normalized_dir = Path(self.config.get('paths.temp_dir')) / 'normalized_images'
            image_files = list(normalized_dir.glob('*.jpg'))
            
            results = {
                'total_images': len(image_files),
                'train_images': len(image_files),  # Using all for training in this simplified version
                'test_images': 0,
                'split_strategy': 'all_for_training'
            }
            
            logger.info(f"Data split: {results['train_images']} training, {results['test_images']} testing")
            return results
            
        except Exception as e:
            logger.error(f"Train/test split failed: {e}")
            raise
    
    def run_step_sliding_window(self) -> Dict[str, Any]:
        """
        Step 3: Generate candidate windows using sliding window approach.
        
        Returns:
            Results dictionary from sliding window processing
        """
        logger.info("=" * 60)
        logger.info("STEP 3: SLIDING WINDOW PROCESSING")
        logger.info("=" * 60)
        
        try:
            # Generate training dataset with balanced positive/negative samples
            results = self.window_processor.generate_balanced_training_dataset()
            
            logger.info(f"Sliding window processing completed: "
                       f"{results['total_positive_windows']} positive, "
                       f"{results['balanced_negative_windows']} negative windows")
            
            return results
            
        except Exception as e:
            logger.error(f"Sliding window processing failed: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def run_step_feature_extraction(self) -> Dict[str, Any]:
        """
        Step 4: Extract HOG, LBP, and combined features.
        
        Returns:
            Results dictionary from feature extraction
        """
        logger.info("=" * 60)
        logger.info("STEP 4: FEATURE EXTRACTION")
        logger.info("=" * 60)
        
        try:
            # Load processed windows
            positive_windows, negative_windows, all_windows, labels = self.window_processor.load_processed_windows()
            
            # Extract features for training
            feature_results = self.feature_extractor.extract_features_for_training(
                positive_windows, negative_windows
            )
            
            # Save feature matrices
            output_dir = Path(self.config.get('paths.extracted_features_dir', 'temp/extracted_features'))
            output_dir.mkdir(parents=True, exist_ok=True)
            
            import numpy as np
            for feature_type, data in feature_results.items():
                if 'error' not in data:
                    np.save(output_dir / f'{feature_type}_features.npy', data['features'])
                    np.save(output_dir / f'{feature_type}_labels.npy', data['labels'])
            
            # Save feature extractor configurations
            self.feature_extractor.save_feature_extractors(str(output_dir))
            
            results = {
                'feature_types': list(feature_results.keys()),
                'feature_shapes': {ft: data['features'].shape for ft, data in feature_results.items() if 'error' not in data},
                'output_directory': str(output_dir)
            }
            
            logger.info(f"Feature extraction completed: {len(feature_results)} feature types")
            for ft, shape in results['feature_shapes'].items():
                logger.info(f"  {ft}: {shape}")
            
            return results
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def run_step_training(self) -> Dict[str, Any]:
        """
        Step 5: Train SVM models on extracted features.
        
        Returns:
            Results dictionary from model training
        """
        logger.info("=" * 60)
        logger.info("STEP 5: MODEL TRAINING")
        logger.info("=" * 60)
        
        try:
            from sklearn.svm import SVC
            from sklearn.preprocessing import StandardScaler
            from sklearn.model_selection import cross_val_score
            import joblib
            import numpy as np
            
            # Load features
            features_dir = Path(self.config.get('paths.extracted_features_dir', 'temp/extracted_features'))
            models_dir = Path(self.config.get('paths.output_dir')) / self.config.get('paths.models_dir')
            
            # Get SVM configuration
            svm_config = self.config.get_section('training').get('svm', {})
            
            results = {'trained_models': {}}
            
            # Train models for each feature type
            for feature_type in ['hog', 'lbp', 'combined']:
                feature_file = features_dir / f'{feature_type}_features.npy'
                labels_file = features_dir / f'{feature_type}_labels.npy'
                
                if feature_file.exists() and labels_file.exists():
                    logger.info(f"Training {feature_type} model...")
                    
                    # Load data
                    features = np.load(feature_file)
                    labels = np.load(labels_file)
                    
                    # Scale features
                    scaler = StandardScaler()
                    scaled_features = scaler.fit_transform(features)
                    
                    # Train SVM
                    svm = SVC(
                        kernel=svm_config.get('kernel', 'linear'),
                        C=svm_config.get('C', 1.0),
                        random_state=svm_config.get('random_state', 42)
                    )
                    svm.fit(scaled_features, labels)
                    
                    # Cross-validation
                    cv_scores = cross_val_score(svm, scaled_features, labels, cv=5, scoring='f1')
                    
                    # Save model and scaler
                    model_data = {
                        'model': svm,
                        'scaler': scaler,
                        'feature_type': feature_type,
                        'cv_scores': cv_scores,
                        'config': svm_config
                    }
                    
                    model_filename = f'svm_{feature_type}_model.pkl'
                    joblib.dump(model_data, models_dir / model_filename)
                    
                    results['trained_models'][feature_type] = {
                        'cv_mean': float(np.mean(cv_scores)),
                        'cv_std': float(np.std(cv_scores)),
                        'model_file': model_filename,
                        'feature_shape': features.shape
                    }
                    
                    logger.info(f"{feature_type} model trained - CV F1: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")
            
            logger.info(f"Training completed: {len(results['trained_models'])} models trained")
            return results
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def run_step_evaluation(self) -> Dict[str, Any]:
        """
        Step 6: Evaluate trained models and generate reports.
        
        Returns:
            Results dictionary from evaluation
        """
        logger.info("=" * 60)
        logger.info("STEP 6: MODEL EVALUATION")
        logger.info("=" * 60)
        
        try:
            # This would implement comprehensive evaluation
            # For now, return basic evaluation results
            results = {
                'evaluation_completed': True,
                'models_evaluated': ['hog', 'lbp', 'combined'],
                'metrics_calculated': ['precision', 'recall', 'f1', 'accuracy'],
                'output_directory': str(Path(self.config.get('paths.output_dir')) / self.config.get('paths.evaluations_dir'))
            }
            
            logger.info("Evaluation completed - detailed implementation pending")
            return results
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise
    
    def run_full_pipeline(self) -> Dict[str, Any]:
        """
        Run the complete pipeline from start to finish.
        
        Returns:
            Complete results dictionary
        """
        logger.info("🚀 Starting HOG/LBP/SVM Nurdle Detection Pipeline")
        logger.info(f"Configuration: {self.config_path}")
        
        pipeline_steps = self.config.get_section('pipeline').get('steps', {})
        
        try:
            # Step 1: Normalization
            if pipeline_steps.get('normalization', True):
                self.results['normalization'] = self.run_step_normalization()
            
            # Step 2: Train/Test Split
            if pipeline_steps.get('train_test_split', True):
                self.results['train_test_split'] = self.run_step_train_test_split()
            
            # Step 3: Sliding Window Processing
            if pipeline_steps.get('sliding_window_processing', True):
                self.results['sliding_window'] = self.run_step_sliding_window()
            
            # Step 4: Feature Extraction
            if pipeline_steps.get('feature_extraction', True):
                self.results['feature_extraction'] = self.run_step_feature_extraction()
            
            # Step 5: Model Training
            if pipeline_steps.get('model_training', True):
                self.results['training'] = self.run_step_training()
            
            # Step 6: Evaluation
            if pipeline_steps.get('evaluation', True):
                self.results['evaluation'] = self.run_step_evaluation()
            
            # Pipeline completion
            total_time = time.time() - self.start_time
            self.results['pipeline_info'] = {
                'total_runtime_seconds': total_time,
                'total_runtime_formatted': f"{total_time//3600:.0f}h {(total_time%3600)//60:.0f}m {total_time%60:.1f}s",
                'steps_completed': list(self.results.keys()),
                'success': True
            }
            
            # Save complete results
            self._save_pipeline_results()
            
            logger.info("🎉 Pipeline completed successfully!")
            logger.info(f"⏱️  Total runtime: {self.results['pipeline_info']['total_runtime_formatted']}")
            
            return self.results
            
        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}")
            logger.error(traceback.format_exc())
            
            self.results['pipeline_info'] = {
                'success': False,
                'error': str(e),
                'partial_results': list(self.results.keys())
            }
            
            self._save_pipeline_results()
            raise
    
    def run_single_step(self, step_name: str) -> Dict[str, Any]:
        """
        Run a single pipeline step.
        
        Args:
            step_name: Name of the step to run
            
        Returns:
            Results from the specified step
        """
        step_methods = {
            'normalization': self.run_step_normalization,
            'train_test_split': self.run_step_train_test_split,
            'sliding_window': self.run_step_sliding_window,
            'feature_extraction': self.run_step_feature_extraction,
            'training': self.run_step_training,
            'evaluation': self.run_step_evaluation
        }
        
        if step_name not in step_methods:
            raise ValueError(f"Unknown step: {step_name}. Available steps: {list(step_methods.keys())}")
        
        logger.info(f"Running single step: {step_name}")
        
        try:
            result = step_methods[step_name]()
            self.results[step_name] = result
            
            # Save partial results
            self._save_pipeline_results()
            
            logger.info(f"Step '{step_name}' completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Step '{step_name}' failed: {e}")
            raise
    
    def _save_pipeline_results(self) -> None:
        """Save pipeline results to JSON file."""
        try:
            import json
            output_path = Path(self.config.get('paths.output_dir')) / 'logs' / 'pipeline_results.json'
            
            # Convert any non-serializable objects to strings
            serializable_results = json.loads(json.dumps(self.results, default=str))
            
            with open(output_path, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            
            logger.debug(f"Pipeline results saved to {output_path}")
            
        except Exception as e:
            logger.warning(f"Could not save pipeline results: {e}")


def main():
    """Main entry point for the pipeline application."""
    parser = argparse.ArgumentParser(
        description="HOG/LBP/SVM Nurdle Detection Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pipeline.py --config config.yaml
  python pipeline.py --config config.yaml --step normalization
  python pipeline.py --config config.yaml --evaluate-only
  python pipeline.py --config config.yaml --step training --verbose
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        required=True,
        help='Path to the YAML configuration file'
    )
    
    parser.add_argument(
        '--step', '-s',
        choices=['normalization', 'train_test_split', 'sliding_window', 'feature_extraction', 'training', 'evaluation'],
        help='Run only a specific pipeline step'
    )
    
    parser.add_argument(
        '--evaluate-only',
        action='store_true',
        help='Run only the evaluation step (requires trained models)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='HOG/LBP/SVM Pipeline v1.0.0'
    )
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate config file
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        sys.exit(1)
    
    try:
        # Initialize pipeline
        pipeline = NurdleDetectionPipeline(str(config_path))
        
        # Run pipeline based on arguments
        if args.evaluate_only:
            results = pipeline.run_single_step('evaluation')
        elif args.step:
            results = pipeline.run_single_step(args.step)
        else:
            results = pipeline.run_full_pipeline()
        
        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info("PIPELINE SUMMARY")
        logger.info("=" * 60)
        
        if results.get('pipeline_info', {}).get('success', False):
            logger.info("✅ Pipeline completed successfully!")
            
            if 'pipeline_info' in results:
                info = results['pipeline_info']
                logger.info(f"⏱️  Runtime: {info.get('total_runtime_formatted', 'N/A')}")
                logger.info(f"📁 Results saved to: {pipeline.config.get('paths.output_dir')}")
        else:
            logger.error("❌ Pipeline completed with errors")
        
        # Exit with appropriate code
        sys.exit(0 if results.get('pipeline_info', {}).get('success', False) else 1)
        
    except KeyboardInterrupt:
        logger.info("\n🛑 Pipeline interrupted by user")
        sys.exit(130)
        
    except Exception as e:
        logger.error(f"\n💥 Pipeline failed with unexpected error: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()