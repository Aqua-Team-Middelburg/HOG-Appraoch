"""
Example usage of the HOG/LBP/SVM Pipeline
========================================

This script demonstrates how to use the pipeline programmatically
and provides examples of common usage patterns.
"""

from pathlib import Path
import sys
import logging

# Add the src directory to path for imports
sys.path.append('src')

from src.utils.config import ConfigLoader
from pipeline import NurdleDetectionPipeline

# Configure logging for example
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_full_pipeline():
    """Example of running the complete pipeline."""
    logger.info("=" * 60)
    logger.info("EXAMPLE: Full Pipeline Execution")
    logger.info("=" * 60)
    
    try:
        # Initialize pipeline
        pipeline = NurdleDetectionPipeline("config.yaml")
        
        # Run complete pipeline
        results = pipeline.run_full_pipeline()
        
        # Print summary
        if results.get('pipeline_info', {}).get('success', False):
            logger.info("✅ Pipeline completed successfully!")
            logger.info(f"Runtime: {results['pipeline_info']['total_runtime_formatted']}")
            
            # Print step results
            for step_name, step_result in results.items():
                if step_name != 'pipeline_info':
                    logger.info(f"Step '{step_name}': {type(step_result).__name__}")
        else:
            logger.error("❌ Pipeline failed")
            
    except Exception as e:
        logger.error(f"Pipeline error: {e}")


def example_step_by_step():
    """Example of running pipeline step by step."""
    logger.info("=" * 60)
    logger.info("EXAMPLE: Step-by-Step Execution")
    logger.info("=" * 60)
    
    try:
        pipeline = NurdleDetectionPipeline("config.yaml")
        
        # Run steps individually
        steps = [
            'normalization',
            'train_test_split', 
            'sliding_window',
            'feature_extraction',
            'training',
            'evaluation'
        ]
        
        for step in steps:
            logger.info(f"Running step: {step}")
            result = pipeline.run_single_step(step)
            logger.info(f"Step '{step}' completed successfully")
            
    except Exception as e:
        logger.error(f"Step execution error: {e}")


def example_custom_config():
    """Example of using custom configuration."""
    logger.info("=" * 60)
    logger.info("EXAMPLE: Custom Configuration")
    logger.info("=" * 60)
    
    try:
        # Load configuration
        config = ConfigLoader("config.yaml")
        config.load()
        
        # Modify parameters
        config.update("preprocessing.target_size", [32, 32])  # Smaller windows
        config.update("training.svm.C", 10.0)  # Higher regularization
        config.update("features.hog.orientations", 12)  # More orientations
        
        # Save modified config
        config.save("config_custom.yaml")
        
        # Run pipeline with custom config
        pipeline = NurdleDetectionPipeline("config_custom.yaml")
        results = pipeline.run_full_pipeline()
        
        logger.info("Custom configuration pipeline completed")
        
    except Exception as e:
        logger.error(f"Custom config error: {e}")


def example_feature_analysis():
    """Example of analyzing extracted features."""
    logger.info("=" * 60)
    logger.info("EXAMPLE: Feature Analysis")
    logger.info("=" * 60)
    
    try:
        from src.feature_extraction.combined_extractor import CombinedFeatureExtractor
        import numpy as np
        
        # Initialize feature extractor
        config = ConfigLoader("config.yaml")
        config.load()
        extractor = CombinedFeatureExtractor(config)
        
        # Load some sample windows (this would be your actual data)
        # For demo, create random windows
        sample_windows = np.random.randint(0, 255, (10, 40, 40, 3), dtype=np.uint8)
        
        # Extract different feature types
        hog_features = extractor.extract_hog_features(sample_windows)
        lbp_features = extractor.extract_lbp_features(sample_windows) 
        combined_features = extractor.extract_combined_features(sample_windows)
        
        # Analyze feature distributions
        hog_analysis = extractor.analyze_feature_distribution(hog_features, 'hog')
        lbp_analysis = extractor.analyze_feature_distribution(lbp_features, 'lbp')
        
        logger.info(f"HOG features shape: {hog_features.shape}")
        logger.info(f"LBP features shape: {lbp_features.shape}")
        logger.info(f"Combined features shape: {combined_features.shape}")
        
        logger.info(f"HOG mean: {hog_analysis['global_stats']['overall_mean']:.3f}")
        logger.info(f"LBP mean: {lbp_analysis['global_stats']['overall_mean']:.3f}")
        
    except Exception as e:
        logger.error(f"Feature analysis error: {e}")


def example_model_evaluation():
    """Example of evaluating trained models."""
    logger.info("=" * 60)
    logger.info("EXAMPLE: Model Evaluation")
    logger.info("=" * 60)
    
    try:
        import joblib
        from pathlib import Path
        
        models_dir = Path("output/models")
        
        # Check if models exist
        model_files = list(models_dir.glob("*.pkl"))
        
        if not model_files:
            logger.warning("No trained models found. Run training first.")
            return
        
        # Load and examine models
        for model_file in model_files:
            logger.info(f"Loading model: {model_file.name}")
            
            try:
                model_data = joblib.load(model_file)
                
                logger.info(f"  Feature type: {model_data['feature_type']}")
                logger.info(f"  CV mean F1: {model_data['cv_scores'].mean():.3f}")
                logger.info(f"  CV std F1: {model_data['cv_scores'].std():.3f}")
                
            except Exception as e:
                logger.error(f"  Error loading {model_file.name}: {e}")
        
    except Exception as e:
        logger.error(f"Model evaluation error: {e}")


def main():
    """Run all examples."""
    examples = [
        ("Full Pipeline", example_full_pipeline),
        ("Step by Step", example_step_by_step), 
        ("Custom Config", example_custom_config),
        ("Feature Analysis", example_feature_analysis),
        ("Model Evaluation", example_model_evaluation)
    ]
    
    print("\n🎯 HOG/LBP/SVM Pipeline Examples")
    print("=" * 60)
    
    for i, (name, func) in enumerate(examples, 1):
        print(f"{i}. {name}")
    
    print(f"{len(examples) + 1}. Run all examples")
    print("0. Exit")
    
    try:
        choice = input("\nEnter your choice (0-{}): ".format(len(examples) + 1))
        choice = int(choice)
        
        if choice == 0:
            print("Goodbye!")
            return
        elif choice == len(examples) + 1:
            # Run all examples
            for name, func in examples:
                print(f"\n\n🚀 Running: {name}")
                func()
        elif 1 <= choice <= len(examples):
            name, func = examples[choice - 1]
            print(f"\n\n🚀 Running: {name}")
            func()
        else:
            print("Invalid choice!")
            
    except (ValueError, KeyboardInterrupt):
        print("\nExiting...")


if __name__ == "__main__":
    main()