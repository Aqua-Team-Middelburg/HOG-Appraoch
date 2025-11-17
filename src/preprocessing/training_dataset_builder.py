"""
Training dataset builder module for batch processing and dataset generation.

This module handles multi-image processing workflows including balanced
dataset generation, window sampling, and persistence of training data.
"""

import numpy as np
import cv2
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Generator
from tqdm import tqdm

from ..utils.config import ConfigLoader

logger = logging.getLogger(__name__)


class TrainingDatasetBuilder:
    """
    Builds balanced training datasets from multiple images.
    
    Responsibilities:
    - Batch processing of multiple images
    - Positive/negative sample balancing
    - Dataset persistence and loading
    - Training statistics tracking
    
    Separates batch workflows from single-image processing.
    """
    
    def __init__(self, config: ConfigLoader, window_processor):
        """
        Initialize training dataset builder.
        
        Args:
            config: Configuration loader instance
            window_processor: SlidingWindowProcessor instance for single-image processing
        """
        self.config = config
        self.window_processor = window_processor
        
        # Get paths from window processor
        self.normalized_dir = window_processor.normalized_dir
        self.candidate_dir = window_processor.candidate_dir
        self.candidate_dir.mkdir(parents=True, exist_ok=True)
        
        # Get labeler reference for metadata
        self.labeler = window_processor.labeler
        
        # Configuration
        self.window_size = window_processor.window_size
        self.stride = window_processor.stride
        self.iou_threshold = window_processor.iou_threshold
        self.gt_bbox_size = window_processor.gt_bbox_size
    
    def generate_balanced_dataset(self, positive_ratio: float = 0.3) -> Dict[str, Any]:
        """
        Generate a balanced training dataset from all normalized images.
        
        Args:
            positive_ratio: Desired ratio of positive to total samples
            
        Returns:
            Dictionary containing balanced training data and statistics
        """
        logger.info("Generating balanced training dataset...")
        
        # Get all normalized image files
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png']:
            image_files.extend(list(self.normalized_dir.glob(f"*{ext}")))
        
        if not image_files:
            raise ValueError("No normalized images found. Run normalization first.")
        
        logger.info(f"Processing {len(image_files)} images for training data")
        
        all_positive_windows = []
        all_negative_windows = []
        
        # Process each image using the window processor
        for image_path in tqdm(image_files, desc="Processing images"):
            result = self.window_processor.process_image_for_training(image_path)
            
            if result['success']:
                all_positive_windows.extend(result['positive_data'])
                all_negative_windows.extend(result['negative_data'])
        
        # Balance the dataset
        num_positive = len(all_positive_windows)
        num_negative_needed = int(num_positive * (1 - positive_ratio) / positive_ratio)
        
        if num_negative_needed > len(all_negative_windows):
            logger.warning(f"Not enough negative samples. Using all {len(all_negative_windows)} available.")
            balanced_negative_windows = all_negative_windows
        else:
            # Randomly sample negative windows
            np.random.seed(self.config.get('system.reproducibility.random_seed', 42))
            indices = np.random.choice(len(all_negative_windows), num_negative_needed, replace=False)
            balanced_negative_windows = [all_negative_windows[i] for i in indices]
        
        # Save processed windows
        self.save_processed_windows(all_positive_windows, balanced_negative_windows)
        
        results = {
            'total_images_processed': len(image_files),
            'total_positive_windows': num_positive,
            'total_negative_windows_available': len(all_negative_windows),
            'balanced_negative_windows': len(balanced_negative_windows),
            'final_positive_ratio': num_positive / (num_positive + len(balanced_negative_windows)),
            'output_directory': str(self.candidate_dir)
        }
        
        logger.info(f"Training dataset generated: {results}")
        return results
    
    def save_processed_windows(self, positive_windows: List[Dict], negative_windows: List[Dict]) -> None:
        """
        Save processed windows to disk for training.
        
        Args:
            positive_windows: List of positive window data dictionaries
            negative_windows: List of negative window data dictionaries
        """
        logger.info("Saving processed windows...")
        
        # Extract window arrays
        positive_arrays = [w['window'] for w in positive_windows]
        negative_arrays = [w['window'] for w in negative_windows]
        
        # Save as numpy arrays
        if positive_arrays:
            np.save(self.candidate_dir / 'processed_windows_positive.npy', positive_arrays)
        if negative_arrays:
            np.save(self.candidate_dir / 'processed_windows_negative.npy', negative_arrays)
        
        # Save metadata
        metadata = {
            'window_size': self.window_size,
            'stride': self.stride,
            'labeling_method': self.labeler.labeling_method,
            'iou_threshold': self.iou_threshold if self.labeler.labeling_method == 'iou' else None,
            'center_tolerance_px': self.labeler.center_tolerance_px if self.labeler.labeling_method == 'center_point' else None,
            'allow_multiple_centers': self.labeler.allow_multiple_centers if self.labeler.labeling_method == 'center_point' else None,
            'gt_bbox_size': self.gt_bbox_size,
            'positive_count': len(positive_windows),
            'negative_count': len(negative_windows)
        }
        
        with open(self.candidate_dir / 'window_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save detailed statistics
        stats = {
            'processing_stats': self.window_processor.stats,
            'positive_samples': [
                {
                    'image_name': w['image_name'],
                    'bbox': w['bbox']
                }
                for w in positive_windows
            ],
            'negative_samples': [
                {
                    'image_name': w['image_name'],
                    'bbox': w['bbox']
                }
                for w in negative_windows
            ]
        }
        
        with open(self.candidate_dir / 'processing_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Saved {len(positive_windows)} positive and {len(negative_windows)} negative windows "
                   f"(labeling method: {self.labeler.labeling_method})")
    
    def load_processed_windows(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load previously processed training windows from disk.
        
        Returns:
            Tuple of (positive_windows, negative_windows, all_windows, labels)
        """
        positive_path = self.candidate_dir / 'processed_windows_positive.npy'
        negative_path = self.candidate_dir / 'processed_windows_negative.npy'
        
        if not positive_path.exists() or not negative_path.exists():
            raise FileNotFoundError("Processed windows not found. Run generate_balanced_dataset first.")
        
        positive_windows = np.load(positive_path)
        negative_windows = np.load(negative_path)
        
        # Combine and create labels
        all_windows = np.concatenate([positive_windows, negative_windows], axis=0)
        labels = np.concatenate([
            np.ones(len(positive_windows)),
            np.zeros(len(negative_windows))
        ])
        
        logger.info(f"Loaded {len(positive_windows)} positive and {len(negative_windows)} negative windows")
        return positive_windows, negative_windows, all_windows, labels
    
    def load_metadata(self) -> Dict[str, Any]:
        """
        Load dataset metadata from disk.
        
        Returns:
            Metadata dictionary
        """
        metadata_path = self.candidate_dir / 'window_metadata.json'
        
        if not metadata_path.exists():
            raise FileNotFoundError("Metadata not found.")
        
        with open(metadata_path, 'r') as f:
            return json.load(f)
    
    def load_statistics(self) -> Dict[str, Any]:
        """
        Load processing statistics from disk.
        
        Returns:
            Statistics dictionary
        """
        stats_path = self.candidate_dir / 'processing_stats.json'
        
        if not stats_path.exists():
            raise FileNotFoundError("Statistics not found.")
        
        with open(stats_path, 'r') as f:
            return json.load(f)
    
    def generate_training_batches(self,
                                 image_list: List[Path],
                                 feature_extractor,
                                 batch_size: int = 1000) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate feature/label batches for incremental training.
        
        This generator yields batches of features and labels for memory-efficient
        incremental training with SGDClassifier.
        
        Args:
            image_list: List of image paths to process
            feature_extractor: Feature extractor instance (HOG, LBP, or Combined)
            batch_size: Number of windows per batch
            
        Yields:
            Tuples of (features, labels) as numpy arrays
        """
        batch_features = []
        batch_labels = []
        
        for image_path in tqdm(image_list, desc="Processing images for incremental training"):
            try:
                # Load image
                image = cv2.imread(str(image_path))
                if image is None:
                    logger.warning(f"Could not load image: {image_path}")
                    continue
                
                # Load ground truth
                json_path = image_path.with_suffix('.json')
                if not json_path.exists():
                    logger.warning(f"No ground truth found for {image_path}")
                    continue
                
                ground_truth = self.window_processor.load_ground_truth_annotations(json_path)
                
                # Extract windows using generator
                for window, bbox in self.window_processor.extract_sliding_windows(image, mode='train'):
                    # Label window
                    label = self.labeler.label_window(bbox, ground_truth, image.shape[:2])
                    
                    # Extract features from window
                    features = feature_extractor.extract_features(window)
                    
                    # Add to batch
                    batch_features.append(features)
                    batch_labels.append(label)
                    
                    # Yield batch when full
                    if len(batch_features) >= batch_size:
                        yield np.array(batch_features), np.array(batch_labels)
                        batch_features = []
                        batch_labels = []
                        
            except Exception as e:
                logger.error(f"Error processing image {image_path}: {e}")
                continue
        
        # Yield remaining samples
        if batch_features:
            yield np.array(batch_features), np.array(batch_labels)
