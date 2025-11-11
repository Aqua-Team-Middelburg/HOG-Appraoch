#!/usr/bin/env python3
"""
Simple setup and run script for the HOG/LBP/SVM Pipeline.

This script checks if dependencies are installed and provides
easy commands to run the pipeline in any environment.
"""

import subprocess
import sys
from pathlib import Path

def check_dependency(package):
    """Check if a Python package is installed."""
    try:
        __import__(package)
        return True
    except ImportError:
        return False

def install_requirements():
    """Install requirements from requirements.txt."""
    print("🔧 Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        print("✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def main():
    """Main setup function."""
    print("🚀 HOG/LBP/SVM Pipeline Setup")
    print("=" * 40)
    
    # Check if we're in the right directory
    if not Path('config.yaml').exists():
        print("❌ config.yaml not found!")
        print("Make sure you're in the HOG_Pipeline_App directory")
        return 1
    
    # Check critical dependencies
    critical_deps = ['numpy', 'sklearn', 'cv2', 'yaml', 'PIL']
    missing_deps = [dep for dep in critical_deps if not check_dependency(dep)]
    
    if missing_deps:
        print(f"❌ Missing dependencies: {', '.join(missing_deps)}")
        print("Installing dependencies...")
        
        if not install_requirements():
            return 1
    else:
        print("✅ All dependencies are available!")
    
    # Create output directories
    output_dirs = [
        'output/models',
        'output/figures', 
        'output/logs',
        'temp/normalized_images',
        'temp/sliding_windows',
        'temp/extracted_features'
    ]
    
    for dir_path in output_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    print("✅ Output directories created!")
    
    print("\n🎯 Ready to run! Use these commands:")
    print("")
    print("Full pipeline:")
    print("  python pipeline.py --config config.yaml")
    print("")
    print("Specific steps:")
    print("  python pipeline.py --config config.yaml --steps normalize,feature_extraction,training")
    print("")
    print("Individual step:")
    print("  python pipeline.py --config config.yaml --step training")
    print("")
    print("Evaluation only:")
    print("  python pipeline.py --config config.yaml --evaluate-only")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())