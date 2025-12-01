"""Features package initialization."""

from .feature_extractor import FeatureExtractor
from .segmentation import (
    build_nurdle_mask,
    detect_background_color,
    segment_blue_bg,
    segment_green_bg,
    segment_white_bg,
    split_touching_particles,
)

__all__ = [
    'FeatureExtractor',
    'build_nurdle_mask',
    'detect_background_color',
    'segment_blue_bg',
    'segment_green_bg',
    'segment_white_bg',
    'split_touching_particles',
]
