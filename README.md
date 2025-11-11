# HOG/LBP/SVM Nurdle Detection Pipeline

A comprehensive machine learning pipeline for detecting nurdles (plastic pellets) in images using Histogram of Oriented Gradients (HOG) and Local Binary Pattern (LBP) features with Support Vector Machine (SVM) classification.

## 🚀 Quick Start

### 1. Setup

```bash
# Clone the repository
git clone https://github.com/Aqua-Team-Middelburg/HOG-Approach.git
cd HOG-Approach/HOG_Pipeline_App

# One-command setup (installs dependencies and creates directories)
python setup.py
```

### Online Code Editors (Recommended)

This pipeline works great in **GitHub Codespaces**, **Google Colab**, **Replit**, or any online Python environment:

1. **Open in your preferred online editor**
2. **Run setup**: `python setup.py` 
3. **Start training**: `python pipeline.py --config config.yaml`

**Benefits:**
- ✅ No local environment setup required
- ✅ Access to cloud compute resources  
- ✅ Works on any machine with Python
- ✅ Easy sharing and collaboration
- ✅ No virtual environment conflicts

### 2. Prepare Your Data

Place your raw images and corresponding JSON annotation files in the `input/` directory:

```
input/
├── image1.jpg
├── image1.json
├── image2.jpg
├── image2.json
└── ...
```

**JSON Format**: Each JSON file should contain ground truth annotations with this structure:
```json
{
  "image": {
    "width": 1920,
    "height": 1080
  },
  "objects": [
    {
      "center_x": 150,
      "center_y": 200,
      "radius": 20,
      "bbox": {
        "x": 130,
        "y": 180,
        "width": 40,
        "height": 40
      }
    }
  ]
}
```

### 3. Run the Pipeline

**Full Pipeline** (recommended for first run):
```bash
python pipeline.py --config config.yaml
```

**Single Step** (for debugging or partial runs):
```bash
python pipeline.py --config config.yaml --step normalization
python pipeline.py --config config.yaml --step training
```

**Evaluation Only** (after training):
```bash
python pipeline.py --config config.yaml --evaluate-only
```

### 4. View Results

Check the `output/` directory for:
- **Models**: `output/models/` - Trained SVM models
- **Evaluations**: `output/evaluations/` - Performance metrics and reports
- **Visualizations**: `output/visualizations/` - Result images and plots
- **Logs**: `output/logs/` - Detailed execution logs

## 📋 Pipeline Steps

The pipeline automatically executes these steps:

1. **📏 Normalization**: Resize images to consistent dimensions while preserving aspect ratios
2. **✂️ Train/Test Split**: Divide data into training and testing sets
3. **🔍 Sliding Window**: Extract candidate windows using sliding window approach
4. **🎯 Feature Extraction**: Extract HOG, LBP, and combined features
5. **🧠 Model Training**: Train SVM models on extracted features
6. **📊 Evaluation**: Generate performance metrics and visualizations

## ⚙️ Configuration

All pipeline parameters are controlled through `config.yaml`. Key sections:

### Paths Configuration
```yaml
paths:
  input_dir: "input"          # Raw images location
  output_dir: "output"        # Results location
```

### Image Processing
```yaml
preprocessing:
  target_size: [40, 40]       # Window size for processing
  contrast_method: "contrast_stretch"  # Contrast enhancement
  sliding_window:
    stride: 20                # Sliding window step size
```

### Feature Extraction
```yaml
features:
  hog:
    orientations: 9           # HOG orientation bins
    pixels_per_cell: [8, 8]   # HOG cell size
  lbp:
    n_points: 8               # LBP sample points
    radius: 1                 # LBP radius
```

### Model Training
```yaml
training:
  svm:
    kernel: "linear"          # SVM kernel type
    C: 1.0                    # Regularization parameter
```

## 📊 Expected Outputs

### Performance Metrics
- **Precision**: Accuracy of positive detections
- **Recall**: Percentage of nurdles detected
- **F1 Score**: Harmonic mean of precision and recall
- **MAPE**: Mean Absolute Percentage Error for count estimation
- **MAE**: Mean Absolute Error for count estimation

### Generated Files
```
output/
├── models/
│   ├── svm_hog_model.pkl        # HOG-based SVM model
│   ├── svm_lbp_model.pkl        # LBP-based SVM model
│   └── svm_combined_model.pkl   # Combined features model
├── evaluations/
│   ├── performance_report.json  # Detailed metrics
│   └── confusion_matrix.png     # Classification results
├── visualizations/
│   ├── detection_results/       # Images with predicted boxes
│   └── performance_plots/       # Metric visualizations
└── logs/
    ├── pipeline.log            # Execution log
    └── pipeline_results.json   # Complete results summary
```

## 🛠️ Troubleshooting

### Common Issues

**Missing Dependencies**:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Memory Issues**:
- Reduce `batch_size` in config
- Process fewer images at once
- Use smaller window stride

**Poor Performance**:
- Check input image quality
- Verify JSON annotation accuracy
- Adjust SVM parameters (C, kernel)
- Modify feature extraction parameters

**File Not Found Errors**:
- Ensure input images and JSON files are in `input/` directory
- Check file naming consistency (image1.jpg ↔ image1.json)

### Getting Help

1. **Check Logs**: Look at `output/logs/pipeline.log` for detailed error messages
2. **Validate Config**: Ensure `config.yaml` has correct paths and parameters
3. **Test with Small Dataset**: Start with 5-10 images to verify setup
4. **Enable Verbose Mode**: Run with `--verbose` flag for detailed output

## 📈 Performance Tips

### For Better Accuracy
- Use high-quality, well-lit images
- Ensure accurate ground truth annotations
- Experiment with different feature combinations
- Try different SVM kernels (linear, rbf)

### For Faster Processing
- Reduce window stride for speed (increase for accuracy)
- Use smaller target image dimensions
- Process images in batches
- Enable parallel processing in config

## 🔧 Advanced Usage

### Custom Feature Parameters

Edit `config.yaml` to experiment with different feature extraction parameters:

```yaml
features:
  hog:
    orientations: 12          # More orientation bins
    pixels_per_cell: [6, 6]   # Smaller cells for more detail
  lbp:
    n_points: 16              # More sample points
    radius: 2                 # Larger radius
```

### Hyperparameter Tuning

Enable grid search in config:
```yaml
training:
  hyperparameter_tuning:
    enabled: true
    param_grid:
      C: [0.1, 1.0, 10.0]
      gamma: ["scale", "auto"]
```

### Ensemble Methods

Use stacking for improved performance:
```yaml
training:
  ensemble:
    stacking:
      enabled: true
      base_models: ["hog_svm", "lbp_svm"]
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines and feel free to submit issues and enhancement requests.

## 👥 Authors

**Aqua Team Middelburg**
- Comprehensive nurdle detection pipeline
- HOG/LBP feature engineering
- SVM optimization and evaluation

## 🙏 Acknowledgments

- Based on computer vision research in marine plastic detection
- Utilizes scikit-learn and scikit-image libraries
- Inspired by object detection methodologies