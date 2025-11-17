# Implementation Plan: Hybrid Approach for Improved Nurdle Detection

**Project**: HOG/LBP/SVM Nurdle Detection Pipeline  
**Version**: 2.0.0  
**Date**: November 17, 2025  
**Status**: Planning Phase

---

## 🎯 Objectives

Transform the current sliding-window classifier into a precise object detection system by implementing:
1. **Center-point based labeling** to eliminate overlapping detections
2. **Color feature extraction** to improve discriminative power
3. **Hard negative mining** to reduce computational overhead
4. **Bounding box regression** for precise localization
5. **Multi-scale detection** for variable nurdle sizes
6. **Computational optimizations** for reduced RAM usage

---

## 🤖 Copilot Instructions for Implementation

**READ THESE CAREFULLY BEFORE STARTING:**

### Code Quality & Maintainability
- ✅ **Keep code maintainable**: No single file should exceed 500 lines. Break large modules into logical sub-modules.
- ✅ **Modular design**: Each function should have a single, clear responsibility
- ✅ **Documentation**: Add docstrings to all functions/classes with parameters, return values, and examples
- ✅ **Type hints**: Use Python type annotations for all function signatures

### File Management
- ❌ **DO NOT create README files** during implementation (only update existing ones if explicitly needed)
- ❌ **DO NOT create summary/documentation markdown files** after each change
- ✅ **Update existing code**: Modify current implementations rather than creating parallel versions
- ✅ **Remove obsolete code**: If functionality is replaced, DELETE the old implementation completely
- ✅ **No bloatware**: Final codebase should only contain actively used code

### Code Replacement Strategy
When implementing new features:
1. **Identify** what existing code becomes obsolete
2. **Replace** in-place rather than adding alongside
3. **Delete** unused functions, classes, or files immediately
4. **Update** all imports and references
5. **Verify** no dead code remains

### Configuration Management
- ✅ **Use config.yaml**: All new parameters must go in config.yaml
- ✅ **No hardcoded values**: Extract magic numbers to configuration
- ✅ **Backward compatibility**: Ensure existing configs still work with sensible defaults

### Testing Approach
- ✅ **Test incrementally**: After each phase, validate with small dataset before proceeding
- ✅ **Keep existing tests working**: Update tests when replacing functionality
- ✅ **Add new tests**: Create tests for new features in the `tests/` directory

### Git Workflow
- ✅ **Commit after each phase**: Small, focused commits with clear messages
- ✅ **Descriptive messages**: Format: `"Phase X.Y: Brief description of change"`
- ✅ **Test before committing**: Ensure code runs without errors

---

## 📋 Implementation Phases

---

## **PHASE 1: Quick Wins (Minimal Changes)**

**Goal**: Achieve immediate improvements with minimal code complexity  
**Estimated Effort**: 2-3 days  
**Risk Level**: Low

---

### **1.1 Center-Point Labeling for Training Data**

**Objective**: Eliminate overlapping positive detections by labeling windows based on ground truth centers

#### Current Behavior (Problem)
```
Window labeled positive if IoU > 0.5 with ground truth box
→ Multiple overlapping windows can all be positive
→ Results in duplicate detections per nurdle
```

#### Target Behavior (Solution)
```
Window labeled positive ONLY if ground truth CENTER point falls inside window
→ Each nurdle has exactly ONE positive window (the one containing its center)
→ Eliminates overlapping positives at training time
```

#### Files to Modify
- `src/preprocessing/window_processor.py`
  - **Replace** `_calculate_iou()` method with `_contains_center_point()`
  - **Update** `_label_windows()` to use center-point logic
  - **Add** `_calculate_nurdle_center()` helper function
  - **Remove** IoU-based labeling code completely

#### New Configuration Parameters
Add to `config.yaml`:
```yaml
training:
  labeling_method: "center_point"  # "center_point" or "iou" (legacy)
  center_point:
    allow_multiple_centers: false  # If >1 center in window, still label positive?
    center_tolerance_px: 0  # Pixels of tolerance around exact center
```

#### Implementation Steps
1. Add helper function to calculate bounding box center from JSON annotations
2. Modify window labeling to check if center (x, y) falls within window bounds
3. Handle edge case: multiple nurdles with centers in same window (log warning, choose primary)
4. Update logging to report "centers captured" instead of "IoU threshold"
5. **Delete** old IoU calculation code after validation

#### Validation Criteria
- [ ] Training dataset has exactly one positive window per nurdle
- [ ] No overlapping positive windows in training set
- [ ] All ground truth centers are captured by at least one window
- [ ] Test with small dataset (10 images) before full run

#### Expected Impact
- ✅ Overlapping detections: **-70% to -80%**
- ✅ Training positives: **-40% to -60%** (fewer ambiguous windows)
- ✅ Model precision: **+10% to +15%**

---

### **1.2 Add Color Features**

**Objective**: Incorporate RGB color information to improve nurdle/background discrimination

#### Current Behavior (Problem)
```
All images converted to grayscale before feature extraction
→ Loses color information (white nurdles, tan sand, blue water)
→ Reduces discriminative power
```

#### Target Behavior (Solution)
```
Extract HOG features from each RGB channel independently
Keep LBP on grayscale (texture invariant to color)
Concatenate multi-channel HOG + grayscale LBP
```

#### Files to Modify
- `src/feature_extraction/hog_extractor.py`
  - **Update** `extract_features()` to support multi-channel mode
  - **Add** `_extract_channel_hog()` helper for per-channel extraction
  - **Keep** grayscale mode as fallback option
  - **Remove** forced grayscale conversion when multi-channel enabled

- `src/feature_extraction/lbp_extractor.py`
  - **Keep** grayscale conversion (LBP benefits from grayscale)
  - No changes needed (already optimal)

- `src/feature_extraction/combined_extractor.py`
  - **Update** to concatenate multi-channel HOG (3x larger) + LBP
  - **Add** feature dimension validation

#### New Configuration Parameters
Add to `config.yaml`:
```yaml
features:
  hog:
    use_color: true  # Extract from RGB channels vs grayscale
    color_space: "RGB"  # "RGB", "HSV", "LAB" (for future experimentation)
    
  feature_dimensions:
    # Auto-calculated, logged for reference
    # Single-channel HOG: 81 dims (current)
    # Multi-channel HOG: 243 dims (81 × 3 channels)
    # LBP: 10 dims (unchanged)
    # Combined: 253 dims (multi-channel) vs 91 dims (grayscale)
```

#### Implementation Steps
1. Modify HOG extractor to accept `use_color` parameter
2. If `use_color=True`, extract HOG separately for R, G, B channels
3. Concatenate channel features: `[hog_r, hog_g, hog_b]`
4. Validate feature vector dimensions (should be 3× grayscale HOG)
5. Update combined extractor to handle larger feature vectors
6. **Keep** option to disable for A/B testing (config flag)

#### Feature Dimension Breakdown
```
Grayscale Pipeline (Current):
  - HOG: 81 dimensions
  - LBP: 10 dimensions
  - Combined: 91 dimensions

RGB Pipeline (New):
  - HOG-R: 81 dimensions
  - HOG-G: 81 dimensions
  - HOG-B: 81 dimensions
  - LBP: 10 dimensions (grayscale)
  - Combined: 253 dimensions
```

#### Validation Criteria
- [ ] Feature extraction runs on RGB images without grayscale conversion
- [ ] Feature vector dimensions are 3× larger for HOG
- [ ] LBP still operates on grayscale
- [ ] Combined features have correct dimensionality (253D)
- [ ] Compare performance: RGB vs Grayscale on validation set

#### Expected Impact
- ✅ Feature dimensionality: **91D → 253D** (+177%)
- ✅ Nurdle detection recall: **+5% to +10%**
- ✅ False positives (sand/rocks): **-20% to -30%**
- ⚠️ Training time: **+20% to +30%** (more features)
- ⚠️ Memory usage: **+50% to +70%** (larger feature matrices)

---

### **1.3 Hard Negative Mining**

**Objective**: Reduce training set size and improve model focus on difficult examples

#### Current Behavior (Problem)
```
All negative windows included in training set
→ Massive imbalance (thousands of negatives per positive)
→ Many "easy" negatives (blank areas, obvious non-nurdles)
→ High memory usage, slow training
```

#### Target Behavior (Solution)
```
Two-stage training:
1. Train initial model on balanced subset
2. Find false positives (hard negatives)
3. Retrain on positives + hard negatives only
→ Smaller, higher-quality training set
```

#### Files to Modify
- `src/training/svm_trainer.py`
  - **Add** `_perform_hard_negative_mining()` method
  - **Add** `_train_initial_model()` helper
  - **Add** `_collect_hard_negatives()` method
  - **Update** `train_single_model()` to use two-stage approach
  - **Keep** original training as fallback option

#### New Configuration Parameters
Add to `config.yaml`:
```yaml
training:
  hard_negative_mining:
    enabled: true
    
    # Stage 1: Initial training
    initial_negative_ratio: 3.0  # Negatives per positive in initial training
    initial_sample_method: "random"  # "random" or "stratified"
    
    # Stage 2: Hard negative collection
    false_positive_threshold: 0.0  # SVM decision threshold for FP collection
    max_hard_negatives: 10000  # Maximum hard negatives to collect
    hard_negative_ratio: 2.0  # Hard negatives per positive in final training
    
    # Iterations
    mining_iterations: 2  # Number of hard negative mining iterations
```

#### Implementation Steps
1. **Stage 1**: Sample balanced initial training set
   - Use all positives
   - Randomly sample 3× negatives
   - Train initial SVM model
   
2. **Stage 2**: Collect hard negatives
   - Run initial model on ALL negative windows
   - Collect false positives (predicted positive, actually negative)
   - Sort by decision function confidence (most confident mistakes)
   - Keep top N hard negatives
   
3. **Stage 3**: Final training
   - Use all positives
   - Use collected hard negatives (discard easy negatives)
   - Train final SVM model
   
4. Optional: Iterate (use final model to find even harder negatives)

5. **Delete** old single-stage training after validation

#### Implementation Details

```
Algorithm: Bootstrapped Hard Negative Mining

Input: 
  - P = all positive windows (e.g., 500 windows)
  - N = all negative windows (e.g., 50,000 windows)
  - ratio = hard_negative_ratio (e.g., 2.0)

Step 1: Initial Training
  N_easy = random_sample(N, size = len(P) × 3)
  model_init = train_svm(P, N_easy)

Step 2: Hard Negative Collection
  predictions = model_init.predict(N)
  false_positives = N[predictions == 1]  # Model thinks they're positive
  
  # Sort by confidence (most confident mistakes first)
  scores = model_init.decision_function(false_positives)
  N_hard = false_positives[top_k_by_score(scores, k = len(P) × ratio)]

Step 3: Final Training
  model_final = train_svm(P, N_hard)
  
Return: model_final
```

#### Validation Criteria
- [ ] Initial model trains on balanced subset (not all negatives)
- [ ] Hard negatives are collected from initial model's false positives
- [ ] Final model trains on positives + hard negatives only
- [ ] Training set size reduced by 60-80%
- [ ] Model performance maintained or improved on validation set

#### Expected Impact
- ✅ Training set size: **-60% to -80%**
- ✅ Training time: **-50% to -70%**
- ✅ Memory usage: **-50% to -70%**
- ✅ Model precision: **+5% to +10%** (focused on hard cases)
- ✅ Model generalization: **Improved** (less overfitting to easy negatives)

---

### **Phase 1 Completion Checklist**

- [ ] All Phase 1.1 validation criteria met
- [ ] All Phase 1.2 validation criteria met
- [ ] All Phase 1.3 validation criteria met
- [ ] Integration test: Full pipeline runs end-to-end
- [ ] Performance benchmark: Compare to baseline (Phase 0)
- [ ] Documentation: Update DEVELOPER_README.md with new features
- [ ] Git: Commit with message `"Phase 1 Complete: Center-point labeling, color features, hard negative mining"`
- [ ] Clean up: Remove all obsolete code and commented-out sections
- [ ] Config: Ensure config.yaml has all new parameters with sensible defaults

---

## **PHASE 2: Better Localization (More Effort)**

**Goal**: Add precise bounding box prediction capabilities  
**Estimated Effort**: 4-6 days  
**Risk Level**: Medium

---

### **2.1 Bounding Box Regression**

**Objective**: Predict precise bounding box offsets to refine detections

#### Current Behavior (Problem)
```
Model predicts: "Is there a nurdle in this 40×40 window?"
Output: Fixed 40×40 box at window location
→ Box may not tightly fit nurdle
→ No information about nurdle position within window
```

#### Target Behavior (Solution)
```
Two-model approach:
1. Classification SVM: "Is there a nurdle?" (existing)
2. Regression SVM: "Where is it and how big?" (new)

For positive windows, predict:
  - Δx: Offset from window center to nurdle center (X)
  - Δy: Offset from window center to nurdle center (Y)
  - Δw: Scale factor for width (nurdle_width / window_width)
  - Δh: Scale factor for height (nurdle_height / window_height)
```

#### Files to Create
- `src/training/bbox_regressor.py`
  - **New class**: `BoundingBoxRegressor`
  - Methods:
    - `train(positive_windows, ground_truth_boxes, features)`
    - `predict(features) → (Δx, Δy, Δw, Δh)`
    - `refine_boxes(window_boxes, predictions) → refined_boxes`
    - `_calculate_regression_targets()`
    - `_transform_predictions_to_boxes()`

#### Files to Modify
- `src/training/svm_trainer.py`
  - **Add** `train_bbox_regressor()` method
  - **Update** model package to include bbox regressor
  - **Keep** classification training separate

- `src/evaluation/evaluator.py`
  - **Add** `_apply_bbox_refinement()` method
  - **Update** `evaluate()` to refine boxes before NMS
  - **Keep** option to disable refinement for A/B testing

#### New Configuration Parameters
Add to `config.yaml`:
```yaml
training:
  bounding_box_regression:
    enabled: true
    
    # Regression target calculation
    normalize_targets: true  # Normalize offsets by window size
    clip_targets: true  # Clip extreme offsets
    max_offset_ratio: 0.5  # Maximum offset as fraction of window size
    
    # Regression model
    regressor_type: "svr"  # "svr" (Support Vector Regression) or "ridge"
    kernel: "linear"
    C: 1.0
    epsilon: 0.1  # SVR epsilon parameter
    
    # Multi-output regression
    independent_outputs: true  # Train separate regressor per output (Δx, Δy, Δw, Δh)
    
evaluation:
  detection:
    apply_bbox_refinement: true  # Use bbox regression during inference
    min_confidence_for_refinement: 0.5  # Only refine high-confidence detections
```

#### Implementation Steps

1. **Create regression target calculator**
   ```
   For each positive window:
     window_center = (window_x + window_w/2, window_y + window_h/2)
     gt_center = (gt_x + gt_w/2, gt_y + gt_h/2)
     
     Δx = (gt_center_x - window_center_x) / window_w
     Δy = (gt_center_y - window_center_y) / window_h
     Δw = gt_w / window_w
     Δh = gt_h / window_h
     
     target = [Δx, Δy, Δw, Δh]
   ```

2. **Train regression model**
   - Use same features as classification
   - Train on POSITIVE windows only
   - Use multi-output SVR or Ridge regression
   - Save regressor with classification model

3. **Implement box refinement**
   ```
   For each detected window:
     [Δx, Δy, Δw, Δh] = regressor.predict(features)
     
     refined_center_x = window_center_x + Δx × window_w
     refined_center_y = window_center_y + Δy × window_h
     refined_w = window_w × Δw
     refined_h = window_h × Δh
     
     refined_box = convert_to_bbox(refined_center, refined_w, refined_h)
   ```

4. **Integrate into pipeline**
   - Train regressor after classification model
   - Apply refinement before NMS
   - Validate refined boxes (ensure positive dimensions, within image)

5. **Delete** old fixed-box output code

#### Validation Criteria
- [ ] Regressor trains on positive windows only
- [ ] Regression targets correctly calculated from ground truth
- [ ] Refinement applied to detections before NMS
- [ ] Refined boxes have improved IoU with ground truth
- [ ] Edge cases handled (boxes outside image, negative dimensions)
- [ ] Compare: refined vs. non-refined on validation set

#### Expected Impact
- ✅ Average IoU with ground truth: **0.4 → 0.7** (+75%)
- ✅ Tight bounding boxes: **+60% to +80%**
- ✅ Localization accuracy: **Significantly improved**
- ⚠️ Training time: **+10% to +15%** (additional regressor)
- ⚠️ Inference time: **+5%** (regression prediction)

---

### **2.2 Multi-Scale Detection**

**Objective**: Detect nurdles of varying sizes by testing at multiple image scales

#### Current Behavior (Problem)
```
Fixed 40×40 sliding window on original image
→ Can only detect nurdles ~40px in size
→ Misses smaller or larger nurdles
→ No scale invariance
```

#### Target Behavior (Solution)
```
Image pyramid approach:
1. Create scaled versions of image (0.75×, 1.0×, 1.25×)
2. Run sliding window detection at each scale
3. Transform detections back to original image coordinates
4. Merge detections across scales using NMS
```

#### Files to Create
- `src/preprocessing/image_pyramid.py`
  - **New class**: `ImagePyramid`
  - Methods:
    - `build(image, scales) → list[scaled_images]`
    - `_resize_image(image, scale)`
    - `transform_boxes_to_original(boxes, scale) → boxes`

#### Files to Modify
- `src/preprocessing/window_processor.py`
  - **Add** `process_multi_scale()` method
  - **Keep** single-scale processing as default
  - **Update** to tag windows with scale information

- `src/evaluation/evaluator.py`
  - **Update** NMS to handle multi-scale detections
  - **Add** scale-aware box merging
  - **Keep** single-scale evaluation path

#### New Configuration Parameters
Add to `config.yaml`:
```yaml
preprocessing:
  multi_scale:
    enabled: false  # Disabled by default (computational cost)
    scales: [0.75, 1.0, 1.25]  # Image scale factors to test
    
    # Scale selection strategy
    auto_scales: false  # Automatically determine scales from GT box sizes
    min_scale: 0.5
    max_scale: 2.0
    scale_step: 0.25
    
    # Performance optimization
    skip_small_scales_for_large_images: true  # Skip 0.5× for images >2000px
    max_image_dimension_for_scaling: 3000  # Don't scale beyond this

evaluation:
  detection:
    multi_scale_nms: true  # Use scale-aware NMS
    scale_nms_threshold: 0.3  # IoU threshold for cross-scale suppression
```

#### Implementation Steps

1. **Build image pyramid**
   ```
   For each scale in [0.75, 1.0, 1.25]:
     scaled_image = resize(original_image, scale)
     scaled_windows = sliding_window(scaled_image)
     
     For each detection in scaled_windows:
       # Transform back to original coordinates
       original_box = {
         x: detection.x / scale,
         y: detection.y / scale,
         w: detection.w / scale,
         h: detection.h / scale,
         scale: scale,
         confidence: detection.confidence
       }
   ```

2. **Scale-aware detection**
   - Extract features at each scale independently
   - Run classifier on all scales
   - Collect all detections with scale metadata

3. **Multi-scale NMS**
   ```
   Algorithm: Cross-Scale Non-Maximum Suppression
   
   Input: detections from all scales
   
   1. Sort all detections by confidence (descending)
   2. For each detection d:
      - Keep d
      - Suppress all lower-confidence detections with IoU(d, ·) > threshold
      - Suppression is scale-invariant (uses original coordinates)
   
   Output: Non-overlapping detections across scales
   ```

4. **Optimization**
   - Only enable multi-scale when nurdle size varies significantly
   - Use 2-3 scales, not 5+ (computational cost)
   - Consider skipping scales for very large/small images

#### Validation Criteria
- [ ] Image pyramid correctly builds scaled versions
- [ ] Detections at each scale run independently
- [ ] Box coordinates correctly transformed to original scale
- [ ] Multi-scale NMS suppresses redundant detections
- [ ] Small and large nurdles detected better than single-scale
- [ ] Performance impact acceptable (measure runtime)

#### Expected Impact
- ✅ Detection of size-variant nurdles: **+15% to +25%** recall
- ✅ Scale robustness: **Significantly improved**
- ⚠️ Inference time: **+150% to +250%** (3 scales = 3× work)
- ⚠️ Memory usage: **+100% to +200%** (multiple image copies)

**Recommendation**: Only enable if nurdle size varies >30% in dataset

---

### **Phase 2 Completion Checklist**

- [ ] All Phase 2.1 validation criteria met
- [ ] All Phase 2.2 validation criteria met
- [ ] Integration test: Bbox regression + multi-scale work together
- [ ] Performance benchmark: Localization accuracy (IoU) improved
- [ ] Ablation study: Measure contribution of each component
- [ ] Documentation: Update DEVELOPER_README.md with Phase 2 features
- [ ] Git: Commit with message `"Phase 2 Complete: Bounding box regression and multi-scale detection"`
- [ ] Clean up: Remove old fixed-box detection code
- [ ] Config: Multi-scale disabled by default (opt-in feature)

---

## **PHASE 3: Optimization (Reduce Computational Load)**

**Goal**: Reduce RAM usage and improve training/inference speed  
**Estimated Effort**: 3-4 days  
**Risk Level**: Medium

---

### **3.1 Generator-Based Feature Extraction**

**Objective**: Process data in batches using Python generators to reduce memory footprint

#### Current Behavior (Problem)
```
Load ALL windows into memory → extract ALL features → train
→ Memory spike when processing 50,000+ windows
→ Cannot process large datasets on limited RAM
```

#### Target Behavior (Solution)
```
Use generators to stream data through pipeline:
1. Yield windows in batches of 1,000
2. Extract features for batch
3. Discard raw window images immediately
4. Accumulate features or train incrementally
```

#### Files to Modify
- `src/preprocessing/window_processor.py`
  - **Add** `generate_windows_batched()` generator method
  - **Keep** `extract_windows()` for backward compatibility
  - **Update** to support streaming mode

- `src/feature_extraction/combined_extractor.py`
  - **Add** `extract_features_batched()` generator method
  - **Yield** feature batches instead of full matrix
  - **Keep** non-batched mode as default

- `src/training/svm_trainer.py`
  - **Add** support for incremental training (SGDClassifier)
  - **Add** `train_incremental()` method
  - **Keep** batch training (LinearSVC) for smaller datasets

#### New Configuration Parameters
Add to `config.yaml`:
```yaml
system:
  performance:
    # Memory management
    batch_processing:
      enabled: false  # Disabled by default (use for large datasets only)
      batch_size: 1000  # Windows per batch
      
    # Incremental training
    incremental_training:
      enabled: false  # Use SGDClassifier instead of LinearSVC
      partial_fit_batch_size: 5000  # Samples per partial_fit call
      n_epochs: 5  # Number of passes over data
      
    # Feature caching
    cache_features_to_disk: false  # Save features to disk for large datasets
    feature_cache_dir: "temp/feature_cache"
```

#### Implementation Steps

1. **Create window generator**
   ```python
   def generate_windows_batched(self, batch_size=1000):
       """Yield windows in batches without loading all to memory."""
       current_batch = []
       
       for image_path in self.image_paths:
           windows = self._sliding_window_single_image(image_path)
           
           for window in windows:
               current_batch.append(window)
               
               if len(current_batch) >= batch_size:
                   yield current_batch
                   current_batch = []
       
       if current_batch:
           yield current_batch  # Final partial batch
   ```

2. **Create feature generator**
   ```python
   def extract_features_batched(self, window_generator):
       """Extract features in batches from generator."""
       for window_batch in window_generator:
           feature_batch = [self.extract(w) for w in window_batch]
           yield np.array(feature_batch)
           # window_batch goes out of scope → garbage collected
   ```

3. **Implement incremental training**
   ```python
   from sklearn.linear_model import SGDClassifier
   
   def train_incremental(self, feature_generator, label_generator):
       """Train using partial_fit for memory efficiency."""
       model = SGDClassifier(loss='hinge', max_iter=1000)
       
       for epoch in range(self.config.n_epochs):
           for features, labels in zip(feature_generator, label_generator):
               model.partial_fit(features, labels, classes=[0, 1])
       
       return model
   ```

4. **Optional: Disk-based feature caching**
   - Save feature batches to disk using HDF5 or joblib
   - Load on-demand during training
   - Useful for repeated experiments on same dataset

#### Validation Criteria
- [ ] Generator yields windows in batches
- [ ] Peak memory usage reduced (monitor with memory_profiler)
- [ ] Features extracted without loading all windows
- [ ] Incremental training produces comparable model to batch training
- [ ] Pipeline completes on large dataset (>50k windows)

#### Expected Impact
- ✅ Peak RAM usage: **-60% to -80%**
- ✅ Can process datasets 3-5× larger
- ⚠️ Training time: **+10% to +20%** (I/O overhead)
- ⚠️ Code complexity: **+Medium** (generators, streaming)

---

### **3.2 Increase Training Stride**

**Objective**: Reduce number of training windows by using larger stride (less overlap)

#### Current Behavior (Problem)
```
stride = 20px, window_size = 40px
→ 50% overlap between adjacent windows
→ Massive redundancy in training data
→ 4× more windows than necessary
```

#### Target Behavior (Solution)
```
Training: stride = 30-35px (25% overlap)
Inference: stride = 10-15px (higher overlap for better recall)
→ 50% fewer training windows
→ Maintain detection quality during inference
```

#### Files to Modify
- `config.yaml` (no code changes needed!)
  - **Update** `preprocessing.sliding_window.stride` parameter
  - **Add** separate stride values for train vs test

#### New Configuration Parameters
```yaml
preprocessing:
  sliding_window:
    # Separate stride for training vs inference
    stride_train: 30  # Larger stride = fewer windows, faster training
    stride_inference: 15  # Smaller stride = better coverage, more detections
    
    # Or use single value (legacy behavior)
    stride: 20  # Used if stride_train/stride_inference not specified
```

#### Implementation Steps

1. **Update config** with separate stride values
2. **Modify pipeline** to use `stride_train` during training phase
3. **Modify pipeline** to use `stride_inference` during evaluation/inference
4. **No other code changes required**

#### Validation Criteria
- [ ] Training uses larger stride (30-35px)
- [ ] Inference uses smaller stride (10-15px)
- [ ] Training window count reduced by 40-60%
- [ ] Detection recall maintained on validation set

#### Expected Impact
- ✅ Training windows: **-40% to -60%**
- ✅ Training time: **-40% to -60%**
- ✅ Training RAM: **-40% to -60%**
- ✅ Detection quality: **Maintained** (inference stride unchanged)

**This is the EASIEST optimization with high impact!**

---

### **3.3 PCA Dimensionality Reduction**

**Objective**: Reduce feature dimensionality using Principal Component Analysis

#### Current Behavior (Problem)
```
Feature dimensionality:
- Grayscale: 91 dimensions (HOG 81 + LBP 10)
- RGB: 253 dimensions (HOG 243 + LBP 10)

High dimensionality → more memory, slower training
```

#### Target Behavior (Solution)
```
Apply PCA to reduce dimensions:
- Grayscale: 91 → 50 dimensions (~45% reduction)
- RGB: 253 → 80 dimensions (~68% reduction)

Minimal accuracy loss (<2%), major memory savings
```

#### Files to Create
- `src/feature_extraction/dimensionality_reduction.py`
  - **New class**: `FeatureReducer`
  - Methods:
    - `fit(features) → reducer`
    - `transform(features) → reduced_features`
    - `fit_transform(features) → reduced_features`
    - `_select_n_components()` (auto-select based on variance explained)

#### Files to Modify
- `src/training/svm_trainer.py`
  - **Add** optional PCA step before training
  - **Save** PCA transformer with model
  - **Apply** PCA during inference
  - **Keep** non-PCA mode as default

#### New Configuration Parameters
Add to `config.yaml`:
```yaml
features:
  dimensionality_reduction:
    enabled: false  # Disabled by default
    method: "pca"  # "pca" only for now (could add "lda" later)
    
    # PCA configuration
    pca:
      n_components: "auto"  # "auto", integer, or float (variance ratio)
      variance_threshold: 0.95  # If "auto", keep 95% of variance
      whiten: false  # Whether to whiten (normalize) components
      
    # Alternative: manual component specification
    target_dimensions:
      hog: 50  # Reduce HOG from 81/243 to 50
      lbp: 10  # Keep LBP unchanged (already small)
      combined: 80  # Target for combined features
```

#### Implementation Steps

1. **Create PCA reducer**
   ```python
   from sklearn.decomposition import PCA
   
   class FeatureReducer:
       def fit(self, features, variance_threshold=0.95):
           # Auto-select n_components to retain variance_threshold
           pca = PCA(n_components=variance_threshold, whiten=False)
           pca.fit(features)
           
           self.pca = pca
           self.original_dim = features.shape[1]
           self.reduced_dim = pca.n_components_
           
           logger.info(f"PCA: {self.original_dim}D → {self.reduced_dim}D "
                      f"({variance_threshold*100:.1f}% variance)")
   ```

2. **Integrate into training pipeline**
   - Fit PCA on training features
   - Transform training and test features
   - Save PCA transformer with model package
   - Apply during inference

3. **Validate retained information**
   - Log explained variance ratio
   - Ensure >95% variance retained
   - Compare model performance with/without PCA

#### Validation Criteria
- [ ] PCA fitted on training features only (no data leakage)
- [ ] Dimensionality reduced as configured
- [ ] Explained variance ≥95%
- [ ] Model accuracy degradation <2%
- [ ] Memory usage reduced proportionally
- [ ] PCA transformer saved and loaded correctly

#### Expected Impact
- ✅ Feature dimensionality: **-45% to -68%** (depending on original)
- ✅ Memory usage: **-40% to -60%**
- ✅ Training time: **-20% to -30%**
- ⚠️ Model accuracy: **-0% to -2%** (minimal loss)

---

### **Phase 3 Completion Checklist**

- [ ] All Phase 3.1 validation criteria met
- [ ] All Phase 3.2 validation criteria met
- [ ] All Phase 3.3 validation criteria met
- [ ] Integration test: All optimizations work together
- [ ] Benchmark: Memory usage reduced significantly
- [ ] Benchmark: Training time reduced
- [ ] Ablation study: Measure impact of each optimization
- [ ] Documentation: Update DEVELOPER_README.md with optimization guide
- [ ] Git: Commit with message `"Phase 3 Complete: Generator-based extraction, stride optimization, PCA"`
- [ ] Clean up: Remove old memory-intensive code paths
- [ ] Config: Optimizations configurable (enable for large datasets only)

---

## 📊 Expected Cumulative Impact

### After Phase 1 (Quick Wins)
```
Metric                        Baseline → Phase 1      Change
─────────────────────────────────────────────────────────────
Overlapping detections        High     → Low          -70%
Precision                     0.65     → 0.75         +15%
Training time                 10 min   → 5 min        -50%
Memory usage (training)       8 GB     → 3 GB         -62%
Feature dimensions            91       → 253          +177%
```

### After Phase 2 (Better Localization)
```
Metric                        Phase 1  → Phase 2      Change
─────────────────────────────────────────────────────────────
Average IoU                   0.4      → 0.7          +75%
Tight bounding boxes          Poor     → Good         +70%
Detection of varied sizes     Medium   → High         +20%
Inference time                2 sec    → 6 sec        +200%
```

### After Phase 3 (Optimization)
```
Metric                        Phase 2  → Phase 3      Change
─────────────────────────────────────────────────────────────
Peak RAM usage                3 GB     → 1.2 GB       -60%
Training time                 5 min    → 3 min        -40%
Feature dimensions            253      → 80           -68%
Model accuracy                0.85     → 0.84         -1%
Can process images            1000     → 5000         +400%
```

---

## 🧪 Testing & Validation Strategy

### Unit Testing
Each phase should include tests in `tests/` directory:
- `tests/test_center_point_labeling.py`
- `tests/test_color_features.py`
- `tests/test_hard_negative_mining.py`
- `tests/test_bbox_regression.py`
- `tests/test_multi_scale.py`
- `tests/test_generators.py`
- `tests/test_pca_reduction.py`

### Integration Testing
- **Phase 1**: End-to-end pipeline with Phase 1 features
- **Phase 2**: Phase 1 + 2 features working together
- **Phase 3**: Full pipeline with all optimizations

### Performance Benchmarks
Create `benchmarks/` directory with:
- `benchmark_baseline.py` (run before starting)
- `benchmark_phase1.py`
- `benchmark_phase2.py`
- `benchmark_phase3.py`

Track metrics:
- Training time
- Inference time
- Peak memory usage
- Precision, Recall, F1
- Average IoU
- Detection count accuracy

### Validation Dataset
- Reserve 10-20 images that are NEVER used in training
- Use for unbiased evaluation after each phase
- Compare metrics phase-by-phase

---

## 📁 Code Organization Guidelines

### File Size Limits
- ❌ No file >500 lines (break into modules)
- ✅ Each class in its own file if >200 lines
- ✅ Use `__init__.py` to organize sub-packages

### Module Structure After Implementation
```
src/
├── feature_extraction/
│   ├── hog_extractor.py (UPDATED: multi-channel HOG)
│   ├── lbp_extractor.py (unchanged)
│   ├── combined_extractor.py (UPDATED: color features)
│   └── dimensionality_reduction.py (NEW: PCA)
│
├── preprocessing/
│   ├── window_processor.py (UPDATED: center-point, generators)
│   ├── image_pyramid.py (NEW: multi-scale)
│   └── normalizer.py (unchanged)
│
├── training/
│   ├── svm_trainer.py (UPDATED: hard negative mining, incremental)
│   └── bbox_regressor.py (NEW: bounding box refinement)
│
├── evaluation/
│   └── evaluator.py (UPDATED: bbox refinement, multi-scale NMS)
│
└── utils/
    ├── config.py (UPDATED: new parameters)
    └── logger.py (unchanged)
```

### Deprecated Code Removal
After each phase, DELETE:
- Old window labeling code (IoU-based) → REMOVE
- Grayscale-only feature extraction → KEEP as fallback, make opt-in
- Single-scale-only evaluation → KEEP as default, multi-scale opt-in
- Batch-only training → KEEP as default, incremental opt-in

**Rule**: If code is replaced and not used as fallback, DELETE immediately.

---

## 🔄 Git Workflow

### Branch Strategy
```
main (protected)
└── feature/phase-1-quick-wins
    ├── Commit 1: Center-point labeling
    ├── Commit 2: Color features
    └── Commit 3: Hard negative mining

└── feature/phase-2-localization
    ├── Commit 4: Bounding box regression
    └── Commit 5: Multi-scale detection

└── feature/phase-3-optimization
    ├── Commit 6: Generator-based extraction
    ├── Commit 7: Stride optimization
    └── Commit 8: PCA reduction
```

### Commit Message Format
```
Phase X.Y: Brief description

- Detailed change 1
- Detailed change 2
- Files modified: file1.py, file2.py
- Files removed: deprecated_file.py

Validation: [passed/failed]
Benchmark: [metric improvements]
```

### Pull Request Template
```markdown
## Phase [X]: [Name]

### Changes
- [ ] Feature 1 implemented
- [ ] Feature 2 implemented
- [ ] Old code removed

### Validation
- [ ] All unit tests pass
- [ ] Integration test passes
- [ ] Benchmark shows improvement

### Performance Impact
| Metric | Before | After | Change |
|--------|--------|-------|--------|
| ...    | ...    | ...   | ...    |

### Breaking Changes
- None / List any

### Backwards Compatibility
- Config backwards compatible: Yes/No
- Models backwards compatible: Yes/No
```

---

## 📋 Pre-Implementation Checklist

Before starting Phase 1:
- [ ] Git repository synced (✅ COMPLETED)
- [ ] Baseline benchmark run and results recorded
- [ ] Validation dataset set aside (never used in training)
- [ ] Config.yaml backed up
- [ ] Development environment set up (Python, dependencies)
- [ ] Read all Copilot instructions in this document
- [ ] Understand code quality requirements
- [ ] Understand file management strategy

---

## 🎯 Success Criteria

### Phase 1 Success
- Overlapping detections reduced by >70%
- Training time reduced by >40%
- Code remains maintainable (no file >500 lines)
- All obsolete code removed

### Phase 2 Success
- Average IoU improved from 0.4 to >0.65
- Bounding boxes visually tight around nurdles
- Multi-scale optional and functional

### Phase 3 Success
- Peak RAM usage reduced by >50%
- Can process 3× larger datasets
- Model accuracy degradation <2%

### Overall Success
- Detection accuracy significantly improved
- Computational requirements reduced
- Codebase clean and maintainable
- No bloatware or dead code
- All features configurable via config.yaml

---

## 📚 Additional Resources

### Reference Implementations
- Original HOG paper: Dalal & Triggs (2005)
- Hard negative mining: Felzenszwalb et al. (2010) - DPM
- Bounding box regression: Girshick et al. (2014) - R-CNN
- Multi-scale detection: Dollar et al. (2014) - Integral Channel Features

### Documentation to Update
- `README.md`: User-facing documentation
- `DEVELOPER_README.md`: Technical details
- `config.yaml`: Inline comments for new parameters
- Docstrings: All new functions and classes

---

## 🚨 Common Pitfalls to Avoid

1. **Data leakage**: Never fit PCA, scalers, or models on test data
2. **Coordinate transforms**: Carefully track window → image coordinate transforms
3. **Memory leaks**: Always delete large arrays when done
4. **Hardcoded values**: Extract to config.yaml
5. **Dead code**: Remove immediately, don't comment out
6. **Overfitting**: Validate on held-out data, not training data
7. **Config defaults**: Ensure backward compatibility

---

## 📞 Questions During Implementation?

**Before proceeding**:
1. Check config.yaml for relevant parameters
2. Review this implementation plan
3. Check DEVELOPER_README.md for existing patterns
4. Run validation tests to ensure no regression

**If stuck**:
- Commit current progress
- Document the specific issue
- Ask for guidance with context

---

**Document Version**: 1.0  
**Last Updated**: November 17, 2025  
**Status**: Ready for Implementation  
**Estimated Total Timeline**: 9-13 days (conservative estimate)

---

**REMINDER**: Read Copilot instructions at top before starting! 🤖
