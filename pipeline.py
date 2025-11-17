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
from typing import Dict, Any, List, Optional, Tuple

# Import pipeline components with error handling
try:
    from src.utils.config import ConfigLoader
    from src.preprocessing.normalizer import ImageNormalizer
    from src.preprocessing.window_processor import SlidingWindowProcessor
    from src.feature_extraction.combined_extractor import CombinedFeatureExtractor
except ImportError as e:
    print(f"Error importing pipeline components: {e}")
    print("Make sure you're running from the HOG_Pipeline_App directory")
    print("Try: cd HOG_Pipeline_App && python pipeline.py --config config.yaml")
    sys.exit(1)

# Configure logging
# Create logs directory if it doesn't exist
logs_dir = Path('output/logs')
logs_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(logs_dir / 'pipeline.log')
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
    
    def _load_bbox_regression_data(self) -> Tuple[List[Tuple[int, int, int, int]], List[Tuple[int, int, int, int]]]:
        """
        Load positive windows and ground truth bboxes for bbox regression training.
        
        Returns:
            Tuple of (positive_windows, ground_truth_bboxes)
        """
        import json
        import numpy as np
        from pathlib import Path
        
        # Load window metadata from sliding window processing
        candidate_dir = Path(self.config.get('paths.candidate_windows_dir', 'temp/candidate_windows'))
        metadata_file = candidate_dir / 'window_metadata.json'
        
        positive_windows = []
        gt_bboxes = []
        
        if metadata_file.exists():
            # Load from saved metadata
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            for item in metadata:
                if item.get('label') == 1:  # Positive windows
                    window_bbox = tuple(item['window_bbox'])
                    gt_bbox = tuple(item['gt_bbox'])
                    positive_windows.append(window_bbox)
                    gt_bboxes.append(gt_bbox)
        
        else:
            # Fallback: load from raw annotations
            logger.warning("No window metadata found, using fallback bbox generation")
            input_dir = Path(self.config.get('paths.input_dir'))
            
            # Find JSON files
            json_files = list(input_dir.glob('*.json'))
            
            for json_file in json_files[:10]:  # Limit for fallback
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                    
                    objects = data.get('objects', [])
                    for obj in objects:
                        # Create normalized ground truth bbox
                        if 'center_x' in obj and 'center_y' in obj:
                            center_x = obj['center_x']
                            center_y = obj['center_y']
                            size = 40  # Default window size
                            
                            window_bbox = (center_x - size//2, center_y - size//2, size, size)
                            gt_bbox = window_bbox  # For fallback, use same as window
                            
                            positive_windows.append(window_bbox)
                            gt_bboxes.append(gt_bbox)
                            
                except Exception as e:
                    logger.error(f"Error loading {json_file}: {e}")
                    continue
        
        logger.info(f"Loaded {len(positive_windows)} positive windows for bbox regression")
        return positive_windows, gt_bboxes
    
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
            
            # Convert results to JSON-serializable format
            def make_serializable(obj):
                if isinstance(obj, dict):
                    # Convert tuple keys to strings
                    return {str(k): make_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, (list, tuple)):
                    return [make_serializable(item) for item in obj]
                else:
                    return obj
            
            serializable_results = make_serializable(results)
            
            with open(output_path, 'w') as f:
                json.dump(serializable_results, f, indent=2, default=str)
            
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
            # Implement proper train/test splitting logic
            normalized_dir = Path(self.config.get('paths.temp_dir')) / 'normalized_images'
            image_files = list(normalized_dir.glob('*.jpg'))
            
            # Get split ratio from config or use default
            split_config = self.config.get_section('training').get('data_split', {})
            test_ratio = split_config.get('test_ratio', 0.2)  # Default 20% for testing
            
            # Calculate split sizes
            total_images = len(image_files)
            test_size = max(1, int(total_images * test_ratio))  # At least 1 test image
            train_size = total_images - test_size
            
            # Store split information for other pipeline steps
            self.train_data = {
                'image_files': image_files[:train_size],
                'size': train_size
            }
            self.test_data = {
                'image_files': image_files[train_size:],
                'size': test_size
            }
            
            results = {
                'total_images': total_images,
                'train_images': train_size,
                'test_images': test_size,
                'split_strategy': f'{test_ratio*100:.0f}% test split',
                'train_files': [str(f) for f in self.train_data['image_files']],
                'test_files': [str(f) for f in self.test_data['image_files']]
            }
            
            logger.info(f"Data split: {results['train_images']} training, {results['test_images']} testing")
            logger.info(f"Split strategy: {results['split_strategy']}")
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
            # Check if processed windows exist, if not, generate them
            try:
                positive_windows, negative_windows, all_windows, labels = self.window_processor.load_processed_windows()
                logger.info("Using existing processed windows")
            except FileNotFoundError:
                logger.info("Processed windows not found, generating sliding window dataset...")
                # Generate sliding window dataset first
                sliding_results = self.run_step_sliding_window()
                logger.info(f"Generated {sliding_results.get('total_positive_windows', 0)} positive windows")
                
                # Now load the processed windows
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
    
    def _run_incremental_training(self, trainer) -> Dict[str, Any]:
        """
        Run incremental training using generators for memory efficiency.
        
        Args:
            trainer: Initialized SVM trainer
            
        Returns:
            Training results dictionary
        """
        logger.info("Setting up incremental training pipeline...")
        
        # Get training images
        train_dir = Path(self.config.get('paths.input_dir')) / 'train'
        if not train_dir.exists():
            # Fallback to main input directory
            train_dir = Path(self.config.get('paths.input_dir'))
        
        # Get list of training images
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_list = []
        for ext in image_extensions:
            image_list.extend(list(train_dir.glob(f'*{ext}')))
            image_list.extend(list(train_dir.glob(f'*{ext.upper()}')))
        
        if not image_list:
            raise ValueError(f"No training images found in {train_dir}")
        
        logger.info(f"Found {len(image_list)} training images")
        
        # Initialize components
        from src.preprocessing.window_processor import SlidingWindowProcessor
        from src.preprocessing.training_dataset_builder import TrainingDatasetBuilder
        from src.feature_extraction.combined_extractor import CombinedFeatureExtractor
        
        window_processor = SlidingWindowProcessor(self.config)
        dataset_builder = TrainingDatasetBuilder(self.config, window_processor)
        feature_extractor = CombinedFeatureExtractor(self.config)
        
        # Pass dataset builder to trainer method
        
        # Train models incrementally for each feature type
        training_results = {}
        
        for feature_type in ['hog', 'lbp', 'combined']:
            logger.info(f"Training {feature_type} model incrementally...")
            try:
                # Get appropriate extractor
                if feature_type == 'hog':
                    extractor = feature_extractor.hog_extractor
                elif feature_type == 'lbp':
                    extractor = feature_extractor.lbp_extractor
                else:  # combined
                    extractor = feature_extractor
                
                # Get batch size from config
                performance_config = self.config.get_section('system').get('performance', {})
                batch_size = performance_config.get('incremental_training', {}).get('partial_fit_batch_size', 1000)
                
                # Train using incremental method
                result = trainer.train_incremental_model(
                    image_list, extractor, feature_type, dataset_builder
                )
                
                if result:
                    training_results[feature_type] = result
                    logger.info(f"[SUCCESS] {feature_type} incremental training completed")
                else:
                    logger.warning(f"[FAIL] {feature_type} incremental training failed")
                    
            except Exception as e:
                logger.error(f"[FAIL] {feature_type} incremental training failed: {e}")
                training_results[feature_type] = {'error': str(e)}
        
        # Note: Stacking ensemble not supported with incremental training
        # Would require storing base model predictions
        
        return {
            'training_results': training_results,
            'training_method': 'incremental',
            'trained_model_types': list(training_results.keys()),
            'models_directory': str(trainer.models_dir)
        }
    
    def run_step_training(self) -> Dict[str, Any]:
        """
        Step 5: Train SVM models on extracted features with comprehensive optimization.
        
        Returns:
            Results dictionary from model training
        """
        logger.info("=" * 60)
        logger.info("STEP 5: MODEL TRAINING")
        logger.info("=" * 60)
        
        try:
            import numpy as np
            from src.training import SVMTrainer
            from src.preprocessing.training_dataset_builder import TrainingDatasetBuilder
            
            # Initialize SVM trainer
            trainer = SVMTrainer(self.config)
            
            # Check if incremental training is enabled
            performance_config = self.config.get_section('system').get('performance', {})
            incremental_enabled = performance_config.get('incremental_training', {}).get('enabled', False)
            
            if incremental_enabled:
                logger.info("[INCREMENTAL] Incremental training enabled - using memory-efficient approach")
                return self._run_incremental_training(trainer)
            else:
                logger.info("📊 Standard training - loading all features into memory")
            
            # Load extracted features
            features_dir = Path(self.config.get('paths.extracted_features_dir', 'temp/extracted_features'))
            
            # Prepare feature data
            feature_data = {}
            
            for feature_type in ['hog', 'lbp', 'combined']:
                feature_file = features_dir / f'{feature_type}_features.npy'
                labels_file = features_dir / f'{feature_type}_labels.npy'
                
                if feature_file.exists() and labels_file.exists():
                    logger.info(f"Loading {feature_type} features from {feature_file}")
                    
                    features = np.load(feature_file)
                    labels = np.load(labels_file)
                    
                    feature_data[feature_type] = {
                        'features': features,
                        'labels': labels
                    }
                    
                    logger.info(f"Loaded {feature_type}: {features.shape} features, {len(labels)} labels")
                else:
                    logger.warning(f"Features not found for {feature_type}")
            
            if not feature_data:
                raise ValueError("No feature data found for training")
            
            # Train all models (HOG, LBP, Combined, Stacking)
            logger.info("Starting comprehensive model training...")
            training_results = trainer.train_all_models(feature_data)
            
            # Train bounding box regressors if enabled
            bbox_config = self.config.get_section('training').get('bounding_box_regression', {})
            bbox_results = {}
            
            if bbox_config.get('enabled', False):
                logger.info("Training bounding box regressors...")
                
                # Load positive windows and ground truth for bbox regression
                try:
                    # Load window metadata needed for bbox regression training
                    positive_windows, gt_bboxes = self._load_bbox_regression_data()
                    
                    for feature_type in ['hog', 'lbp', 'combined']:
                        if feature_type in feature_data and feature_type in training_results:
                            if 'error' not in training_results[feature_type]:
                                try:
                                    # Get positive features for this feature type
                                    features = feature_data[feature_type]['features']
                                    labels = feature_data[feature_type]['labels']
                                    positive_features = features[labels == 1]
                                    
                                    if len(positive_features) > 0:
                                        logger.info(f"Training {feature_type} bbox regressor...")
                                        bbox_result = trainer.train_bbox_regressor(
                                            positive_features,
                                            positive_windows[:len(positive_features)],  # Match positive sample count
                                            gt_bboxes[:len(positive_features)],
                                            feature_type
                                        )
                                        bbox_results[feature_type] = bbox_result
                                        logger.info(f"[SUCCESS] {feature_type} bbox regressor training completed")
                                    else:
                                        logger.warning(f"No positive samples for {feature_type} bbox regression")
                                        
                                except Exception as e:
                                    logger.error(f"[FAIL] {feature_type} bbox regressor training failed: {e}")
                                    bbox_results[feature_type] = {'error': str(e)}
                
                except Exception as e:
                    logger.error(f"Failed to load bbox regression data: {e}")
                    logger.info("Skipping bbox regression training")
            
            # Generate training summary
            model_summary = trainer.get_model_summary()
            
            results = {
                'training_results': training_results,
                'bbox_regressor_results': bbox_results,
                'model_summary': model_summary,
                'feature_data_shapes': {
                    ft: data['features'].shape 
                    for ft, data in feature_data.items()
                },
                'models_directory': str(trainer.models_dir),
                'trained_model_types': list(training_results.keys()),
                'bbox_regressors_trained': list(bbox_results.keys()) if bbox_results else []
            }
            
            # Log summary
            logger.info("=" * 40)
            logger.info("TRAINING SUMMARY")
            logger.info("=" * 40)
            
            successful_models = 0
            for model_type, result in training_results.items():
                if 'error' not in result:
                    cv_mean = result.get('cv_mean', 0)
                    cv_std = result.get('cv_std', 0)
                    logger.info(f"[SUCCESS] {model_type}: F1 = {cv_mean:.4f} ± {cv_std:.4f}")
                    successful_models += 1
                else:
                    logger.error(f"[FAIL] {model_type}: {result['error']}")
            
            logger.info(f"Training completed: {successful_models}/{len(training_results)} models successful")
            
            return results
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def run_step_evaluation(self) -> Dict[str, Any]:
        """
        Step 6: Evaluate trained models with comprehensive metrics and visualizations.
        
        Returns:
            Results dictionary from evaluation
        """
        logger.info("=" * 60)
        logger.info("STEP 6: MODEL EVALUATION")
        logger.info("=" * 60)
        
        try:
            import numpy as np
            from src.training import SVMTrainer
            from src.evaluation import ModelEvaluator
            
            # Initialize evaluator
            evaluator = ModelEvaluator(self.config)
            
            # Initialize trainer to load models
            trainer = SVMTrainer(self.config)
            
            # Load trained models
            models_dir = trainer.models_dir
            model_files = list(models_dir.glob('svm_*_model_*.pkl'))
            
            if not model_files:
                logger.warning("No trained models found for evaluation")
                return {'error': 'No trained models found', 'models_directory': str(models_dir)}
            
            # Load most recent models
            loaded_models = {}
            for model_file in model_files:
                try:
                    model_package = trainer.load_model(str(model_file))
                    feature_type = model_package['feature_type']
                    loaded_models[feature_type] = model_package
                    logger.info(f"Loaded {feature_type} model from {model_file.name}")
                except Exception as e:
                    logger.error(f"Failed to load model {model_file.name}: {e}")
            
            if not loaded_models:
                raise ValueError("Failed to load any trained models")
            
            # Load bbox regressors if available
            bbox_regressors = {}
            bbox_files = list(models_dir.glob('bbox_regressor_*.pkl'))
            
            for bbox_file in bbox_files:
                try:
                    from src.training.bbox_regressor import BoundingBoxRegressor
                    bbox_regressor = BoundingBoxRegressor(self.config)
                    bbox_regressor.load(str(bbox_file))
                    
                    # Extract feature type from filename
                    feature_type = bbox_file.stem.replace('bbox_regressor_', '')
                    bbox_regressors[feature_type] = bbox_regressor
                    logger.info(f"Loaded {feature_type} bbox regressor from {bbox_file.name}")
                except Exception as e:
                    logger.error(f"Failed to load bbox regressor {bbox_file.name}: {e}")
            
            if bbox_regressors:
                logger.info(f"Loaded {len(bbox_regressors)} bbox regressors for evaluation")
            
            # Load test set with proper metadata handling
            logger.info("Loading test set with enhanced metadata support...")
            test_set_info = evaluator.test_loader.load_test_set_features_and_metadata()
            test_data = test_set_info['test_data']
            window_metadata = test_set_info['window_metadata']
            
            logger.info(f"Test set loaded from: {test_set_info['source']}")
            for feature_type, data in test_data.items():
                logger.info(f"Test {feature_type}: {data['features'].shape}")
            
            # Evaluate each model
            evaluation_results = {}
            
            for model_name, model_package in loaded_models.items():
                if model_name in test_data:
                    logger.info(f"Evaluating {model_name} model...")
                    
                    test_features = test_data[model_name]['features']
                    test_labels = test_data[model_name]['labels']
                    
                    # Perform evaluation
                    eval_result = evaluator.evaluate_single_model(
                        model_package, test_features, test_labels, model_name
                    )
                    evaluation_results[model_name] = eval_result
                    
                    # Log key metrics
                    metrics = eval_result['metrics']
                    logger.info(f"{model_name} Results:")
                    logger.info(f"  Accuracy: {metrics.get('accuracy', 0):.4f}")
                    logger.info(f"  Precision: {metrics.get('precision', 0):.4f}")
                    logger.info(f"  Recall: {metrics.get('recall', 0):.4f}")
                    logger.info(f"  F1-Score: {metrics.get('f1_score', 0):.4f}")
                    if metrics.get('roc_auc'):
                        logger.info(f"  ROC AUC: {metrics['roc_auc']:.4f}")
            
            # Generate model comparison
            if len(evaluation_results) > 1:
                logger.info("Generating model comparison...")
                comparison_results = evaluator.compare_models(evaluation_results)
            else:
                comparison_results = {}
            
            # Generate image-level evaluation with proper metadata and bbox refinement
            logger.info("Calculating image-level performance with NMS, bbox refinement, and proper aggregation...")
            image_level_results = evaluator.evaluate_image_level_performance(
                evaluation_results, window_metadata, bbox_regressors, test_data
            )
            
            # Create visualizations
            logger.info("Creating evaluation visualizations...")
            visualization_paths = evaluator.create_visualizations(evaluation_results)
            
            # Generate summary statistics
            summary_stats = evaluator.generate_summary_statistics(evaluation_results)
            
            # Save comprehensive evaluation report
            report_path = evaluator.save_evaluation_report(
                evaluation_results, 
                image_level_results, 
                visualization_paths
            )
            
            # Compile final results
            results = {
                'evaluation_results': evaluation_results,
                'model_comparison': comparison_results,
                'image_level_results': image_level_results,
                'summary_statistics': summary_stats,
                'visualization_paths': visualization_paths,
                'evaluation_report': report_path,
                'models_evaluated': list(evaluation_results.keys()),
                'test_samples_per_model': {
                    model: len(test_data[model]['labels'])
                    for model in evaluation_results.keys()
                    if model in test_data
                }
            }
            
            # Log final summary
            logger.info("=" * 40)
            logger.info("EVALUATION SUMMARY")
            logger.info("=" * 40)
            
            if comparison_results.get('model_ranking'):
                logger.info("Model Ranking (by overall score):")
                for i, model in enumerate(comparison_results['model_ranking'], 1):
                    logger.info(f"  {i}. {model}")
            
            logger.info(f"Generated {len(visualization_paths)} visualizations")
            logger.info(f"Evaluation report saved: {report_path}")
            logger.info("Evaluation completed successfully!")
            
            return results
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def run_full_pipeline(self) -> Dict[str, Any]:
        """
        Run the complete pipeline from start to finish.
        
        Returns:
            Complete results dictionary
        """
        logger.info("[PIPELINE] Starting HOG/LBP/SVM Nurdle Detection Pipeline")
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
            
            logger.info("[SUCCESS] Pipeline completed successfully!")
            logger.info(f"[TIME] Total runtime: {self.results['pipeline_info']['total_runtime_formatted']}")
            
            return self.results
            
        except Exception as e:
            logger.error(f"[FAIL] Pipeline failed: {e}")
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
        '--steps',
        help='Run specific pipeline steps (comma-separated list: normalize,sliding_window,feature_extraction,training,evaluation)'
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
        elif args.steps:
            # Parse comma-separated steps
            step_names = [s.strip() for s in args.steps.split(',')]
            
            # Map step names to internal names
            step_mapping = {
                'normalize': 'normalization',
                'normalization': 'normalization',
                'sliding_window': 'sliding_window',
                'feature_extraction': 'feature_extraction',
                'training': 'training',
                'evaluation': 'evaluation'
            }
            
            # Run specified steps in order
            results = {}
            for step_name in step_names:
                internal_step = step_mapping.get(step_name, step_name)
                if internal_step:  # Ensure not None
                    logger.info(f"Running step: {step_name} -> {internal_step}")
                    step_result = pipeline.run_single_step(internal_step)
                    results[internal_step] = step_result
                else:
                    logger.warning(f"Unknown step: {step_name}")
        else:
            results = pipeline.run_full_pipeline()
        
        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info("PIPELINE SUMMARY")
        logger.info("=" * 60)
        
        if results.get('pipeline_info', {}).get('success', False):
            logger.info("[SUCCESS] Pipeline completed successfully!")
            
            if 'pipeline_info' in results:
                info = results['pipeline_info']
                logger.info(f"[TIME] Runtime: {info.get('total_runtime_formatted', 'N/A')}")
                logger.info(f"[OUTPUT] Results saved to: {pipeline.config.get('paths.output_dir')}")
        else:
            logger.error("[FAIL] Pipeline completed with errors")
        
        # Exit with appropriate code
        sys.exit(0 if results.get('pipeline_info', {}).get('success', False) else 1)
        
    except KeyboardInterrupt:
        logger.info("\n[INTERRUPT] Pipeline interrupted by user")
        sys.exit(130)
        
    except Exception as e:
        logger.error(f"\n[ERROR] Pipeline failed with unexpected error: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()