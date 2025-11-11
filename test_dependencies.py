#!/usr/bin/env python3
"""
Simple test script to verify the pipeline can be imported and initialized.
This is useful for checking if all dependencies are properly installed.
"""

import sys
import logging
from pathlib import Path

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_imports():
    """Test if all required modules can be imported."""
    try:
        logger.info("Testing core imports...")
        
        import numpy
        logger.info(f"✅ NumPy {numpy.__version__}")
        
        import sklearn
        logger.info(f"✅ scikit-learn {sklearn.__version__}")
        
        import skimage
        logger.info(f"✅ scikit-image {skimage.__version__}")
        
        import cv2
        logger.info(f"✅ OpenCV {cv2.__version__}")
        
        import yaml
        logger.info("✅ PyYAML")
        
        import matplotlib
        logger.info(f"✅ Matplotlib {matplotlib.__version__}")
        
        # Optional imports
        try:
            import optuna
            logger.info(f"✅ Optuna {optuna.__version__} (optional)")
        except ImportError:
            logger.warning("⚠️  Optuna not available (optional - will use GridSearch)")
        
        return True
        
    except ImportError as e:
        logger.error(f"❌ Import failed: {e}")
        return False

def test_pipeline_import():
    """Test if the pipeline modules can be imported."""
    try:
        logger.info("Testing pipeline imports...")
        
        from src.utils.config import ConfigLoader
        logger.info("✅ ConfigLoader")
        
        from src.preprocessing.normalizer import ImageNormalizer
        logger.info("✅ ImageNormalizer")
        
        from src.preprocessing.window_processor import SlidingWindowProcessor
        logger.info("✅ SlidingWindowProcessor")
        
        from src.feature_extraction.combined_extractor import CombinedFeatureExtractor
        logger.info("✅ CombinedFeatureExtractor")
        
        from src.training.svm_trainer import SVMTrainer
        logger.info("✅ SVMTrainer")
        
        from src.evaluation.evaluator import ModelEvaluator
        logger.info("✅ ModelEvaluator")
        
        return True
        
    except ImportError as e:
        logger.error(f"❌ Pipeline import failed: {e}")
        return False

def test_config_loading():
    """Test if the configuration file can be loaded."""
    try:
        logger.info("Testing configuration loading...")
        
        config_path = Path("config.yaml")
        if not config_path.exists():
            logger.error("❌ config.yaml not found")
            return False
        
        from src.utils.config import ConfigLoader
        config = ConfigLoader(str(config_path))
        
        # Test some key configurations
        paths = config.get_section('paths')
        logger.info(f"✅ Config loaded - {len(paths)} path configurations")
        
        features = config.get_section('feature_extraction')
        logger.info(f"✅ Feature config - {len(features)} feature settings")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Config loading failed: {e}")
        return False

def main():
    """Run all tests."""
    logger.info("=" * 60)
    logger.info("HOG/LBP/SVM Pipeline - Dependency Check")
    logger.info("=" * 60)
    
    all_passed = True
    
    # Test core imports
    if not test_imports():
        all_passed = False
    
    logger.info("")
    
    # Test pipeline imports
    if not test_pipeline_import():
        all_passed = False
    
    logger.info("")
    
    # Test configuration
    if not test_config_loading():
        all_passed = False
    
    logger.info("")
    logger.info("=" * 60)
    
    if all_passed:
        logger.info("🎉 All tests passed! Pipeline should be ready to run.")
        logger.info("")
        logger.info("To run the full pipeline:")
        logger.info("  python pipeline.py --config config.yaml --verbose")
        logger.info("")
        logger.info("To run specific steps:")
        logger.info("  python pipeline.py --config config.yaml --steps normalize,feature_extraction,training")
        return 0
    else:
        logger.error("❌ Some tests failed. Please check the error messages above.")
        logger.error("")
        logger.error("To install missing dependencies:")
        logger.error("  pip install -r requirements.txt")
        return 1

if __name__ == "__main__":
    sys.exit(main())