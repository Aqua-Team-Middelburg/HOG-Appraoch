"""
Segmentation utilities to isolate nurdles before feature extraction.

Only the segmentation logic is included here; counting/visualization is handled elsewhere.
"""

from typing import Tuple

import cv2
import numpy as np
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage as ndi


def detect_background_color(img_bgr: np.ndarray) -> str:
    """
    Heuristically detect dominant background color from the image border.

    Returns one of: 'white', 'green', 'blue'.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, w = hsv.shape[:2]
    border = max(5, int(0.05 * min(h, w)))
    border_mask = np.zeros((h, w), dtype=np.uint8)
    border_mask[:border, :] = 1
    border_mask[-border:, :] = 1
    border_mask[:, :border] = 1
    border_mask[:, -border:] = 1
    border_pixels = hsv[border_mask == 1]

    mean_h = float(np.mean(border_pixels[:, 0]))
    mean_s = float(np.mean(border_pixels[:, 1]))
    mean_v = float(np.mean(border_pixels[:, 2]))

    if mean_s < 40 and mean_v > 150:
        return "white"
    if 35 <= mean_h <= 85:
        return "green"
    if 90 <= mean_h <= 140:
        return "blue"

    # Fallback based on dominant color channel (green vs blue)
    border_bgr = cv2.cvtColor(border_pixels.reshape(-1, 1, 3), cv2.COLOR_HSV2BGR).reshape(-1, 3)
    channel_means = np.mean(border_bgr, axis=0)
    return "green" if channel_means[1] >= channel_means[0] else "blue"


def _apply_illumination_correction(gray: np.ndarray, blur_sigma: int = 35) -> np.ndarray:
    """
    Apply illumination correction to a grayscale image.
    
    Uses Gaussian blur subtraction to remove uneven lighting, followed by CLAHE
    for contrast enhancement.
    
    Args:
        gray: Grayscale image
        blur_sigma: Sigma for Gaussian blur (higher = larger illumination patterns removed)
        
    Returns:
        Enhanced grayscale image
    """
    bg_blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)
    detrended = cv2.subtract(gray, bg_blur)
    detrended_norm = cv2.normalize(detrended, None, 0, 255, cv2.NORM_MINMAX)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(detrended_norm)
    
    return enhanced


def segment_white_bg(img_bgr: np.ndarray) -> np.ndarray:
    """Segment nurdles on a white/light background using illumination correction and Otsu thresholding."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # Apply illumination correction
    enhanced = _apply_illumination_correction(gray, blur_sigma=35)
    
    # Otsu threshold on enhanced image
    _, mask = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    return mask


def segment_green_bg(img_bgr: np.ndarray) -> np.ndarray:
    """Segment nurdles when the background is predominantly green, with illumination correction."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # Apply illumination correction for better contrast
    enhanced = _apply_illumination_correction(gray, blur_sigma=35)
    
    # Apply HSV-based color detection on original for green background
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 40, 40], dtype=np.uint8)
    upper_green = np.array([85, 255, 255], dtype=np.uint8)
    background = cv2.inRange(hsv, lower_green, upper_green)
    
    # Combine: use enhanced contrast to separate nurdles from potentially weak background
    _, contrast_mask = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Merge both masks (nurdles are where contrast is high OR outside green background)
    mask = cv2.bitwise_or(contrast_mask, cv2.bitwise_not(background))
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask


def segment_blue_bg(img_bgr: np.ndarray) -> np.ndarray:
    """Segment nurdles when the background is predominantly blue, with illumination correction."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # Apply illumination correction for better contrast
    enhanced = _apply_illumination_correction(gray, blur_sigma=35)
    
    # Apply HSV-based color detection on original for blue background
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([90, 40, 40], dtype=np.uint8)
    upper_blue = np.array([140, 255, 255], dtype=np.uint8)
    background = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # Combine: use enhanced contrast to separate nurdles from potentially weak background
    _, contrast_mask = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Merge both masks (nurdles are where contrast is high OR outside blue background)
    mask = cv2.bitwise_or(contrast_mask, cv2.bitwise_not(background))
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask


def split_touching_particles(binary_mask: np.ndarray, min_dist: int = 8) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split touching nurdles using distance transform and watershed.

    Args:
        binary_mask: Binary mask (0/255) with nurdles as foreground.
        min_dist: Minimum distance between local maxima seeds.
    """
    bin_mask = (binary_mask > 0).astype(np.uint8)
    dist = cv2.distanceTransform(bin_mask, cv2.DIST_L2, 5)
    coords = peak_local_max(dist, min_distance=min_dist, labels=bin_mask)
    seed_mask = np.zeros(dist.shape, dtype=bool)
    seed_mask[tuple(coords.T)] = True
    markers, _ = ndi.label(seed_mask)
    labels = watershed(-dist, markers, mask=bin_mask)
    return labels, dist


def segment_contrast_based(
    img_bgr: np.ndarray,
    blur_sigma: int = 35,
    clahe_clip: float = 2.0,
    clahe_tiles: int = 8,
    min_dist: int = 8,
    min_area: int = 50,
    max_area: int = 2000,
) -> np.ndarray:
    """
    Unified contrast-based segmentation using illumination correction.
    
    This method applies:
    1) Grayscale -> illumination correction -> CLAHE
    2) Otsu threshold -> binary mask
    3) Morphological opening
    4) Distance transform + watershed splitting
    5) Size-based filtering
    
    This is background-agnostic and works well across different lighting conditions.
    
    Args:
        img_bgr: Input image (BGR)
        blur_sigma: Sigma for Gaussian blur in illumination correction
        clahe_clip: CLAHE clip limit
        clahe_tiles: CLAHE tile grid size
        min_dist: Minimum distance between watershed seeds
        min_area: Minimum particle area (pixels)
        max_area: Maximum particle area (pixels)
        
    Returns:
        Binary mask with filtered nurdles (0 = background, 255 = nurdle)
    """
    # Step 1: Convert to grayscale and apply illumination correction
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    bg_blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)
    detrended = cv2.subtract(gray, bg_blur)
    detrended_norm = cv2.normalize(detrended, None, 0, 255, cv2.NORM_MINMAX)
    
    # Step 2: CLAHE enhancement
    clahe = cv2.createCLAHE(
        clipLimit=clahe_clip,
        tileGridSize=(clahe_tiles, clahe_tiles),
    )
    enhanced = clahe.apply(detrended_norm)
    
    # Step 3: Otsu threshold
    _, mask = cv2.threshold(
        enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    
    # Step 4: Morphological opening
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_open = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Step 5: Distance transform + watershed
    labels, _ = split_touching_particles(mask_open, min_dist=min_dist)
    
    # Step 6: Size filtering
    filtered_mask = np.zeros_like(mask_open, dtype=np.uint8)
    num_labels = int(np.max(labels))
    for lbl in range(1, num_labels + 1):
        area = np.sum(labels == lbl)
        if min_area <= area <= max_area:
            filtered_mask[labels == lbl] = 255
    
    # Fallback: if filtering removes everything, return base mask
    if np.count_nonzero(filtered_mask) == 0:
        filtered_mask = mask_open
    
    return filtered_mask


def build_nurdle_mask(
    img_bgr: np.ndarray,
    min_dist: int = 8,
    min_area: int = 50,
    max_area: int = 2000,
    use_contrast_method: bool = True
) -> np.ndarray:
    """
    Build a cleaned binary mask isolating nurdles for downstream HOG extraction.
    
    Uses unified contrast-based segmentation by default (background-agnostic),
    with fallback to background-specific methods if needed.
    
    Args:
        img_bgr: Input image (BGR)
        min_dist: Minimum distance between watershed seeds
        min_area: Minimum particle area (pixels)
        max_area: Maximum particle area (pixels)
        use_contrast_method: If True, use unified contrast-based method; 
                            if False, use background-specific methods
    
    Returns:
        Binary mask with filtered nurdles (0 = background, 255 = nurdle)
    """
    if use_contrast_method:
        # Use unified contrast-based segmentation (primary method)
        return segment_contrast_based(
            img_bgr,
            blur_sigma=35,
            clahe_clip=2.0,
            clahe_tiles=8,
            min_dist=min_dist,
            min_area=min_area,
            max_area=max_area
        )
    else:
        # Fallback to background-specific methods
        bg_type = detect_background_color(img_bgr)
        if bg_type == "green":
            base_mask = segment_green_bg(img_bgr)
        elif bg_type == "blue":
            base_mask = segment_blue_bg(img_bgr)
        else:
            base_mask = segment_white_bg(img_bgr)

        labels, _ = split_touching_particles(base_mask, min_dist=min_dist)
        filtered_mask = np.zeros_like(base_mask, dtype=np.uint8)
        num_labels = int(np.max(labels))
        for lbl in range(1, num_labels + 1):
            area = np.sum(labels == lbl)
            if min_area <= area <= max_area:
                filtered_mask[labels == lbl] = 255

        if np.count_nonzero(filtered_mask) == 0:
            filtered_mask = base_mask

        return filtered_mask
