"""
services/vision/us_thyroid/postprocess.py
==========================================
Mask (numpy array / base64 PNG) -> SpatialDerived fields for thyroid.

Identical to us_breast/postprocess.py except:
  - Default organ='thyroid'
  - get_location_quadrant defaults to thyroid logic (left-lobe/right-lobe/isthmus)
  - pixel_spacing_mm default = 0.1 (kept as-is, needs calibration against the real device)

Public API (same as us_breast):
    extract_spatial_features(mask, pixel_spacing_mm) -> dict
    get_location_quadrant(centroid, image_size, organ) -> tuple[str, str]
    postprocess_mask(mask_png_base64, original_size, organ, pixel_spacing_mm) -> dict
"""

import base64
import cv2
import numpy as np
from typing import Tuple


# Computing spatial features from the mask

def extract_spatial_features(
    mask: np.ndarray,
    pixel_spacing_mm: float = 0.1,
) -> dict:
    """
    Computes spatial features from a binary mask.

    Args:
        mask:             binary mask (H, W), values 0 or 255
        pixel_spacing_mm: mm/pixel (default 0.1 - needs calibration against the probe)

    Returns a dict: bbox, area_cm2, centroid, width_px, height_px,
                  aspect_ratio, circularity.
    Returns an empty dict if there is no contour (no nodule detected).
    """
    mask_u8 = mask.astype(np.uint8)
    if mask_u8.max() > 1:
        mask_u8 = (mask_u8 > 127).astype(np.uint8)

    contours, _ = cv2.findContours(
        mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return {}

    largest  = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    area_px  = cv2.contourArea(largest)
    area_cm2 = round(area_px * (pixel_spacing_mm ** 2) / 100, 3)

    M = cv2.moments(largest)
    if M['m00'] > 0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
    else:
        cx, cy = x + w // 2, y + h // 2

    aspect_ratio = round(w / h, 3) if h > 0 else 1.0
    perimeter    = cv2.arcLength(largest, True)
    circularity  = (
        round(4 * np.pi * area_px / (perimeter ** 2), 3)
        if perimeter > 0 else 0.0
    )

    return {
        'bbox':         [x, y, x + w, y + h],
        'area_cm2':     area_cm2,
        'centroid':     [cx, cy],
        'width_px':     w,
        'height_px':    h,
        'aspect_ratio': aspect_ratio,   # > 1.5 -> elongated (suspicious)
        'circularity':  circularity,    # < 0.5 -> irregular margin (suspicious)
    }


# Determining the nodule's quadrant

def get_location_quadrant(
    centroid: list,
    image_size: Tuple[int, int],
    organ: str = 'thyroid',
) -> Tuple[str, str]:
    """
    Determines the nodule's quadrant from the centroid.

    Thyroid convention:
        x < W*0.35  -> left-lobe
        x > W*0.65  -> right-lobe
        else        -> isthmus
    (Breast logic kept as-is for backward compatibility)

    Returns: (quadrant_str, confidence_str)
    """
    H, W   = image_size
    cx, cy = centroid

    dist_to_edge = min(cx, W - cx, cy, H - cy)
    if dist_to_edge < 0.1 * min(H, W):
        confidence = 'low'
    elif dist_to_edge < 0.2 * min(H, W):
        confidence = 'medium'
    else:
        confidence = 'high'

    if organ == 'thyroid':
        if cx < W * 0.35:
            return 'left-lobe', confidence
        elif cx > W * 0.65:
            return 'right-lobe', confidence
        else:
            return 'isthmus', confidence

    elif organ == 'breast':
        mid_x, mid_y = W / 2, H / 2
        if abs(cx - mid_x) < W * 0.1 and abs(cy - mid_y) < H * 0.1:
            return 'central', confidence
        vertical   = 'upper' if cy < mid_y else 'lower'
        horizontal = 'outer' if cx < mid_x else 'inner'
        return f'{vertical}-{horizontal}', confidence

    else:
        return 'unknown', 'low'


# Main postprocess function

def postprocess_mask(
    mask_png_base64: str,
    original_size: Tuple[int, int],
    organ: str = 'thyroid',
    pixel_spacing_mm: float = 0.1,
) -> dict:
    """
    Decodes the mask from a base64 PNG -> SpatialDerived fields.

    Args:
        mask_png_base64:  mask PNG base64-encoded
        original_size:    (H, W) of the original image
        organ:            'thyroid' (default) | 'breast'
        pixel_spacing_mm: mm/pixel

    Raises:
        ValueError: if the base64 cannot be decoded or is not a valid PNG.
    """
    if not mask_png_base64:
        raise ValueError(
            'mask_png_base64 is empty - vision service did not return a mask. '
            'This is an upstream error, not a no-nodule case.'
        )

    try:
        mask_bytes = base64.b64decode(mask_png_base64, validate=True)
    except Exception as e:
        raise ValueError(f'Failed to decode base64 mask: {e}') from e

    mask_array = np.frombuffer(mask_bytes, dtype=np.uint8)
    mask       = cv2.imdecode(mask_array, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(
            'cv2.imdecode returned None - base64 decoded but is not a valid '
            'PNG. Check the encoding step in the vision service.'
        )

    spatial = extract_spatial_features(mask, pixel_spacing_mm)

    if not spatial:
        # Valid case: mask is OK but there is no nodule (model predicts benign/clear)
        return _empty_spatial(original_size)

    quadrant, loc_confidence = get_location_quadrant(
        spatial['centroid'], original_size, organ
    )

    return {
        'bbox':                spatial['bbox'],
        'area_cm2':            spatial['area_cm2'],
        'centroid':            spatial['centroid'],
        'location_quadrant':   quadrant,
        'aspect_ratio':        spatial['aspect_ratio'],
        'circularity':         spatial['circularity'],
        'width_px':            spatial['width_px'],
        'height_px':           spatial['height_px'],
        'location_confidence': loc_confidence,
    }


def _empty_spatial(image_size: Tuple[int, int]) -> dict:
    """Fallback for when no nodule is detected (thyroid clear)."""
    H, W = image_size
    return {
        'bbox':                [0, 0, 0, 0],
        'area_cm2':            0.0,
        'centroid':            [W // 2, H // 2],
        'location_quadrant':   'none',
        'aspect_ratio':        1.0,
        'circularity':         1.0,
        'width_px':            0,
        'height_px':           0,
        'location_confidence': 'low',
    }
