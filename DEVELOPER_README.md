# HOG/LBP/SVM Pipeline - Developer Documentation

This document provides detailed technical information for developers working on or extending the nurdle detection pipeline.

## 🏗️ Architecture Overview

### Project Structure
```
HOG_Pipeline_App/
├── config.yaml                 # Master configuration
├── pipeline.py                 # Main application entry point
├── requirements.txt            # Python dependencies
├── README.md                   # User documentation
├── DEVELOPER_README.md         # This file
├── src/                        # Source code modules
│   ├── __init__.py
│   ├── utils/                  # Utility modules
│   │   ├── config.py          # Configuration management
│   │   └── __init__.py
│   ├── preprocessing/          # Image preprocessing
│   │   ├── normalizer.py      # Image normalization
│   │   ├── window_processor.py # Sliding window processing
│   │   └── __init__.py
│   ├── feature_extraction/     # Feature extractors
│   │   ├── hog_extractor.py   # HOG features
│   │   ├── lbp_extractor.py   # LBP features
│   │   ├── combined_extractor.py # Combined features
│   │   └── __init__.py
│   ├── training/              # Model training (extensible)
│   │   └── __init__.py
│   └── evaluation/            # Evaluation metrics (extensible)
│       └── __init__.py
├── input/                     # User data input
├── output/                    # Generated results
│   ├── models/               # Trained models
│   ├── evaluations/          # Evaluation reports
│   ├── visualizations/       # Result visualizations
│   └── logs/                 # Execution logs
└── tests/                    # Unit tests (future)
```

## 🔧 Core Components

### 1. Configuration Management (`src/utils/config.py`)

**Purpose**: Centralized configuration management with type safety and validation.

**Key Classes**:
- `ConfigLoader`: Loads and validates YAML configuration
- `TypedConfig`: Provides type-safe access to configuration parameters
- `ModelConfig`, `HOGConfig`, `LBPConfig`: Type-safe dataclasses

**Usage**:
```python
from src.utils.config import ConfigLoader

config = ConfigLoader("config.yaml")
config.load()
target_size = config.get("preprocessing.target_size")
hog_params = config.get_section("features")["hog"]
```

**Key Features**:
- Environment variable substitution (`${VAR_NAME}`)
- Configuration validation and error reporting
- Configuration hashing for versioning
- Dot-notation access to nested parameters

### 2. Image Preprocessing (`src/preprocessing/`)

#### ImageNormalizer (`normalizer.py`)
**Purpose**: Normalize input images to consistent dimensions while preserving annotations.

**Key Methods**:
- `normalize_all_images()`: Process all input images
- `calculate_target_dimensions()`: Compute scaling parameters
- `scale_coordinates()`: Update JSON annotations
- `validate_normalization()`: Verify processing results

**Design Principles**:
- Maintains aspect ratios during resizing
- Updates corresponding JSON annotations automatically
- Provides comprehensive statistics and validation
- Handles various image formats consistently

#### SlidingWindowProcessor (`window_processor.py`)
**Purpose**: Extract candidate windows using sliding window approach for training and inference.

**Key Methods**:
- `extract_sliding_windows()`: Generate windows from image
- `label_window()`: Assign positive/negative labels based on IoU
- `generate_balanced_training_dataset()`: Create balanced training data
- `calculate_iou()`: Intersection over Union calculation

**Critical Features**:
- Consistent window extraction for training and testing
- IoU-based labeling for ground truth matching
- Balanced dataset generation to prevent class imbalance
- Efficient batch processing with progress tracking

### 3. Feature Extraction (`src/feature_extraction/`)

#### HOGExtractor (`hog_extractor.py`)
**Purpose**: Extract Histogram of Oriented Gradients features with consistent parameters.

**Key Methods**:
- `extract_features()`: Single window HOG extraction
- `extract_features_batch()`: Efficient batch processing
- `extract_features_with_visualization()`: Generate HOG visualizations

**Configuration Parameters**:
```yaml
features:
  hog:
    orientations: 9              # Number of gradient orientation bins
    pixels_per_cell: [8, 8]     # Cell size in pixels
    cells_per_block: [2, 2]     # Block size in cells
    block_norm: "L2-Hys"        # Normalization method
```

#### LBPExtractor (`lbp_extractor.py`)
**Purpose**: Extract Local Binary Pattern features with histogram-based representation.

**Key Methods**:
- `extract_features()`: Single window LBP histogram
- `extract_lbp_image()`: Raw LBP pattern visualization
- `analyze_lbp_distribution()`: Statistical analysis

**Configuration Parameters**:
```yaml
features:
  lbp:
    n_points: 8                 # Number of sample points
    radius: 1                   # Sampling radius
    method: "uniform"           # LBP variant
```

#### CombinedFeatureExtractor (`combined_extractor.py`)
**Purpose**: Unified interface for multiple feature types with optional normalization.

**Key Methods**:
- `extract_features_for_training()`: Extract all feature types for training
- `extract_features()`: Runtime feature extraction by type
- `fit_feature_normalizers()`: Optional feature scaling

**Advanced Features**:
- Support for multiple feature combinations
- Optional per-feature normalization before concatenation
- Feature distribution analysis and validation
- Serializable configuration for model deployment

## 🔬 Data Flow Architecture

### Training Pipeline Flow
```
Raw Images + JSON → Normalization → Sliding Windows → Feature Extraction → Model Training
     ↓                   ↓               ↓                    ↓               ↓
  Resize &          40x40 windows    HOG + LBP          SVM Training      Saved Models
  Contrast          with IoU         Feature Vectors    with CV           + Scalers
  Enhancement       labeling         (108 + 10 dims)   Validation        + Metadata
```

### Inference Pipeline Flow
```
Test Image → Sliding Windows → Feature Extraction → Model Prediction → NMS → Final Detections
     ↓             ↓                   ↓                    ↓          ↓           ↓
  Same preprocessing  Same parameters   Same features   Saved model  Overlap     Bounding boxes
  as training        as training       as training     + scaler     removal     + confidences
```

### Critical Design Principle: **IDENTICAL PREPROCESSING**

The pipeline ensures that **exactly the same preprocessing steps** are applied during both training and inference:

1. **Window Preprocessing**: Same resize, contrast enhancement, and color space conversion
2. **Feature Extraction**: Identical HOG/LBP parameters and implementations
3. **Feature Scaling**: Use training-fitted scalers during inference (no data leakage)

## 🔍 Key Algorithms

### 1. IoU-Based Window Labeling

```python
def calculate_iou(self, box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
    """Calculate Intersection over Union between two bounding boxes."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # Calculate intersection
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection = (x_right - x_left) * (y_bottom - y_top)
    union = w1 * h1 + w2 * h2 - intersection
    
    return intersection / union if union > 0 else 0.0
```

### 2. Balanced Dataset Generation

The pipeline addresses class imbalance by:
- Extracting all positive windows (IoU ≥ threshold)
- Randomly sampling negative windows to achieve desired ratio
- Ensuring representative distribution of negative examples

### 3. Feature Vector Construction

**HOG Features** (81 dimensions for 40×40 window):
- Gradient orientation histograms in overlapping blocks
- L2-Hys normalization for illumination invariance

**LBP Features** (10 dimensions for uniform patterns):
- Histogram of local binary patterns
- Rotation-invariant texture descriptors

**Combined Features** (91 dimensions):
- Concatenation of HOG + LBP
- Optional individual feature normalization

## 🧪 Testing and Validation

### Configuration Validation
The pipeline includes comprehensive configuration validation:
- Required section presence checking
- Parameter type and range validation
- Cross-parameter consistency verification

### Data Integrity Checks
- Image format and dimension validation
- JSON structure and coordinate verification
- Feature vector dimension consistency

### Processing Validation
- Intermediate result verification
- Statistical distribution analysis
- Error tracking and recovery

## 🚀 Extension Points

### Adding New Feature Extractors

1. **Create New Extractor Class**:
```python
class NewFeatureExtractor:
    def __init__(self, config: ConfigLoader):
        self.config = config
        # Load parameters from config
        
    def extract_features(self, window: np.ndarray) -> np.ndarray:
        # Implement feature extraction
        pass
        
    def get_config(self) -> Dict[str, Any]:
        # Return configuration for serialization
        pass
```

2. **Update CombinedFeatureExtractor**:
- Add new extractor initialization
- Extend `extract_features()` method
- Update configuration handling

3. **Update Configuration Schema**:
```yaml
features:
  new_feature:
    parameter1: value1
    parameter2: value2
```

### Adding New Model Types

1. **Create Training Module** (`src/training/new_model.py`):
```python
class NewModelTrainer:
    def train(self, features: np.ndarray, labels: np.ndarray) -> Any:
        # Implement training logic
        pass
        
    def predict(self, features: np.ndarray) -> np.ndarray:
        # Implement prediction logic
        pass
```

2. **Update Pipeline Integration**:
- Add model training step in `pipeline.py`
- Update configuration schema
- Extend evaluation metrics

### Adding New Evaluation Metrics

1. **Create Evaluation Module** (`src/evaluation/metrics.py`):
```python
def calculate_new_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # Implement metric calculation
    pass
```

2. **Update Pipeline**:
- Add metric calculation to evaluation step
- Include in result reporting
- Add to visualization generation

## ⚡ Performance Optimization

### Memory Management
- Use batch processing for large datasets
- Implement lazy loading for images
- Clear intermediate results when possible

### Computational Efficiency
- Vectorize operations where possible
- Use efficient data structures (numpy arrays)
- Implement parallel processing for independent operations

### I/O Optimization
- Minimize file system operations
- Use efficient serialization formats
- Implement result caching where appropriate

## 🐛 Debugging Guidelines

### Common Issues and Solutions

**Feature Dimension Mismatches**:
- Verify window sizes match between training and inference
- Check HOG/LBP parameter consistency
- Validate preprocessing pipeline consistency

**Poor Model Performance**:
- Analyze feature distributions for anomalies
- Check class balance in training data
- Verify ground truth annotation quality

**Memory Issues**:
- Reduce batch sizes
- Implement streaming processing
- Clear unused variables explicitly

**Configuration Errors**:
- Use configuration validation features
- Check file paths and permissions
- Verify parameter data types

### Debugging Tools

**Logging**:
```python
import logging
logger = logging.getLogger(__name__)
logger.debug("Detailed debugging information")
logger.info("General information")
logger.warning("Warning messages")
logger.error("Error messages")
```

**Configuration Debugging**:
```python
# Check loaded configuration
print(config.get_section('features'))

# Validate specific parameters
print(config.get('preprocessing.target_size'))
```

**Feature Analysis**:
```python
# Analyze feature distributions
analysis = feature_extractor.analyze_feature_distribution(features, 'hog')
print(f"Feature stats: {analysis['global_stats']}")
```

## 📊 Performance Monitoring

### Metrics to Track
- Processing times per pipeline step
- Memory usage during processing
- Feature extraction statistics
- Model training convergence
- Evaluation metric trends

### Profiling Integration
```python
import cProfile
import pstats

def profile_pipeline_step():
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Run pipeline step
    result = pipeline.run_step_feature_extraction()
    
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)
```

## 🔒 Production Considerations

### Error Handling
- Implement comprehensive exception handling
- Provide meaningful error messages
- Enable graceful degradation where possible

### Logging and Monitoring
- Structured logging with appropriate levels
- Performance metric collection
- Error tracking and alerting

### Configuration Management
- Environment-specific configurations
- Parameter validation and defaults
- Configuration versioning and migration

### Model Deployment
- Model versioning and metadata
- Backward compatibility considerations
- A/B testing framework integration

## 📚 References and Resources

### Academic Papers
- Dalal, N. & Triggs, B. (2005). "Histograms of Oriented Gradients for Human Detection"
- Ojala, T., Pietikäinen, M. & Mäenpää, T. (2002). "Multiresolution Gray-Scale and Rotation Invariant Texture Classification with Local Binary Patterns"

### Technical Documentation
- [scikit-image Feature Documentation](https://scikit-image.org/docs/dev/api/skimage.feature.html)
- [scikit-learn SVM Documentation](https://scikit-learn.org/stable/modules/svm.html)
- [OpenCV Python Tutorials](https://docs.opencv.org/master/d6/d00/tutorial_py_root.html)

### Development Tools
- **IDE**: VS Code with Python extension
- **Debugging**: Python debugger (pdb), VS Code debugger
- **Profiling**: cProfile, line_profiler, memory_profiler
- **Testing**: pytest, unittest
- **Documentation**: Sphinx, mkdocs

---

For questions or contributions, please contact the Aqua Team Middelburg development team.