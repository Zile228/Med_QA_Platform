"""
services/vision/us_breast/postprocess.py
==========================================
Mask (numpy array / base64 PNG) -> SpatialDerived fields.

Public API:
    extract_spatial_features(mask, pixel_spacing_mm) -> dict
    get_location_quadrant(centroid, image_size, organ) -> tuple[str, str]
    postprocess_mask(mask_png_base64, original_size, organ, pixel_spacing_mm) -> dict

Called by knowledge/mapper.py - receives the mask as a base64 PNG over the
HTTP body, does NOT read from a path on disk (vision and knowledge are 2
separate containers that do not share a filesystem - see ISSUES_AND_FIXES.md item 1).
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
        pixel_spacing_mm: real-world distance per pixel (mm).
                          0.1mm/px is the default value for the BUSI dataset.
                          1 cm2 = 100 mm2 -> area_px * spacing^2 / 100

    Returns a dict with:
        bbox, area_cm2, centroid, width_px, height_px,
        aspect_ratio, circularity
    Returns an empty dict if no contour is found (normal image/no lesion).
    """
    mask_uint8 = mask.astype(np.uint8)
    if mask_uint8.max() > 1:
        mask_uint8 = (mask_uint8 > 127).astype(np.uint8)

    contours, _ = cv2.findContours(
        mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return {}

    # Take the largest contour, ignore noise
    largest = max(contours, key=cv2.contourArea)

    x, y, w, h = cv2.boundingRect(largest)
    area_px = cv2.contourArea(largest)
    area_cm2 = round(area_px * (pixel_spacing_mm ** 2) / 100, 3)

    M = cv2.moments(largest)
    if M["m00"] > 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        cx, cy = x + w // 2, y + h // 2

    # Compute shape descriptors
    aspect_ratio = round(w / h, 3) if h > 0 else 1.0
    perimeter = cv2.arcLength(largest, True)
    circularity = (
        round(4 * np.pi * area_px / (perimeter ** 2), 3)
        if perimeter > 0 else 0.0
    )

    return {
        "bbox": [x, y, x + w, y + h],
        "area_cm2": area_cm2,
        "centroid": [cx, cy],
        "width_px": w,
        "height_px": h,
        "aspect_ratio": aspect_ratio,   # > 1.5 -> elongated (suspicious)
        "circularity": circularity,     # < 0.5 -> irregular margin (suspicious)
    }


# Determining the lesion's quadrant

def get_location_quadrant(
    centroid: list,
    image_size: Tuple[int, int],
    organ: str = "breast",
) -> Tuple[str, str]:
    """
    Determines the lesion's quadrant from the centroid and image size.

    Args:
        centroid:   [cx, cy] pixel coordinates
        image_size: (H, W) of the original image
        organ:      'breast' | 'thyroid'

    Returns:
        (quadrant_str, confidence_str)

    Breast quadrants (following clinical convention):
        Upper-outer | Upper-inner | Lower-outer | Lower-inner | Central
        Axis: x < W/2 -> outer (if right breast, needs flipping - POC assumes right breast)
              y < H/2 -> upper

    Thyroid:
        Left-lobe | Right-lobe | Isthmus
        Axis: x < W*0.35 -> left-lobe, x > W*0.65 -> right-lobe, else isthmus
    """
    H, W = image_size
    cx, cy = centroid

    # Confidence based on the distance from the centroid to the image edge
    dist_to_edge = min(cx, W - cx, cy, H - cy)
    if dist_to_edge < 0.1 * min(H, W):
        confidence = "low"    # centroid near the edge -> quadrant is uncertain
    elif dist_to_edge < 0.2 * min(H, W):
        confidence = "medium"
    else:
        confidence = "high"

    if organ == "breast":
        mid_x, mid_y = W / 2, H / 2
        # Margin zone around the midline -> central
        if abs(cx - mid_x) < W * 0.1 and abs(cy - mid_y) < H * 0.1:
            return "central", confidence

        vertical   = "upper" if cy < mid_y else "lower"
        horizontal = "outer" if cx < mid_x else "inner"
        # Assumes right breast: outer = left side of the image
        return f"{vertical}-{horizontal}", confidence

    elif organ == "thyroid":
        if cx < W * 0.35:
            return "left-lobe", confidence
        elif cx > W * 0.65:
            return "right-lobe", confidence
        else:
            return "isthmus", confidence

    else:
        return "unknown", "low"


# Main postprocess function

def postprocess_mask(
    mask_png_base64: str,
    original_size: Tuple[int, int],
    organ: str = "breast",
    pixel_spacing_mm: float = 0.1,
) -> dict:
    """
    Decodes the mask from a base64 PNG -> returns all SpatialDerived fields.

    Args:
        mask_png_base64:  mask PNG base64-encoded (received directly over the
                           HTTP body, does NOT read from a path on disk - vision
                           and knowledge are 2 separate containers that don't
                           share a filesystem)
        original_size:    (H, W) of the original image
        organ:            'breast' | 'thyroid'
        pixel_spacing_mm: mm/pixel

    Returns a dict mapping 1-1 into the SpatialDerived schema.

    Raises:
        ValueError: if the base64 cannot be decoded, or the bytes are not a valid PNG.
                    Fails loudly, no silent fallback, to avoid fabricated data.
    """
    if not mask_png_base64:
        raise ValueError(
            "mask_png_base64 is empty - vision service did not return a mask. "
            "This is an upstream error, not a 'normal/no lesion' case."
        )

    try:
        mask_bytes = base64.b64decode(mask_png_base64, validate=True)
    except Exception as e:
        raise ValueError(f"Failed to decode base64 mask: {e}") from e

    mask_array = np.frombuffer(mask_bytes, dtype=np.uint8)
    mask = cv2.imdecode(mask_array, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(
            "cv2.imdecode returned None - base64 decoded but is not a valid PNG. "
            "Check the encoding step in the vision service."
        )

    spatial = extract_spatial_features(mask, pixel_spacing_mm)

    if not spatial:
        # Valid case: mask is OK but has no contour (normal image/no lesion)
        return _empty_spatial(original_size)

    quadrant, loc_confidence = get_location_quadrant(
        spatial["centroid"], original_size, organ
    )

    return {
        "bbox":               spatial["bbox"],
        "area_cm2":           spatial["area_cm2"],
        "centroid":           spatial["centroid"],
        "location_quadrant":  quadrant,
        "aspect_ratio":       spatial["aspect_ratio"],
        "circularity":        spatial["circularity"],
        "width_px":           spatial["width_px"],
        "height_px":          spatial["height_px"],
        "location_confidence": loc_confidence,
    }


def _empty_spatial(image_size: Tuple[int, int]) -> dict:
    """Fallback for when no lesion is detected (normal image)."""
    H, W = image_size
    return {
        "bbox":                [0, 0, 0, 0],
        "area_cm2":            0.0,
        "centroid":            [W // 2, H // 2],
        "location_quadrant":   "none",
        "aspect_ratio":        1.0,
        "circularity":         1.0,
        "width_px":            0,
        "height_px":           0,
        "location_confidence": "low",
    }
