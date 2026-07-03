import base64
import cv2
import numpy as np
from typing import Optional, Tuple


def extract_spatial_features(
    mask: np.ndarray,
    pixel_spacing_mm: Optional[float] = None,
) -> dict:
    """
    Computes spatial features from a binary mask.

    Args:
        mask:             binary mask (H, W), values 0 or 255
        pixel_spacing_mm: real-world distance per pixel (mm).
                          None means no DICOM metadata available; area_cm2 will be None.
    """
    mask_uint8 = mask.astype(np.uint8)
    if mask_uint8.max() > 1:
        mask_uint8 = (mask_uint8 > 127).astype(np.uint8)

    contours, _ = cv2.findContours(
        mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return {}

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    area_px = cv2.contourArea(largest)

    if pixel_spacing_mm is not None:
        area_cm2 = round(area_px * (pixel_spacing_mm ** 2) / 100, 3)
        pixel_spacing_reliable = True
    else:
        area_cm2 = None
        pixel_spacing_reliable = False

    M = cv2.moments(largest)
    if M["m00"] > 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        cx, cy = x + w // 2, y + h // 2

    aspect_ratio = round(w / h, 3) if h > 0 else 1.0
    perimeter = cv2.arcLength(largest, True)
    circularity = (
        round(4 * np.pi * area_px / (perimeter ** 2), 3)
        if perimeter > 0 else 0.0
    )

    if aspect_ratio < 0.8:
        aspect_ratio_interpretation = "taller-than-wide (suspicious per BI-RADS)"
    elif aspect_ratio > 1.8:
        aspect_ratio_interpretation = "markedly wider-than-tall (low suspicion)"
    else:
        aspect_ratio_interpretation = "intermediate"

    return {
        "bbox": [x, y, x + w, y + h],
        "area_cm2": area_cm2,
        "pixel_spacing_reliable": pixel_spacing_reliable,
        "centroid": [cx, cy],
        "width_px": w,
        "height_px": h,
        "aspect_ratio": aspect_ratio,
        "aspect_ratio_interpretation": aspect_ratio_interpretation,
        "circularity": circularity,
    }


def get_location_quadrant(
    centroid: list,
    image_size: Tuple[int, int],
    organ: str = "breast",
    laterality: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Determines the lesion's quadrant from the centroid and image size.

    Breast: upper-outer | upper-inner | lower-outer | lower-inner | central
    Thyroid: left-lobe | right-lobe | isthmus

    For breast, image-left vs image-right alone does not determine
    outer/inner; it depends on which breast was imaged. With laterality
    'right', image-left is medial (inner) and image-right is lateral
    (outer); with laterality 'left' it is reversed. When laterality is
    None, the side cannot be determined and is reported as such.
    """
    H, W = image_size
    cx, cy = centroid

    dist_to_edge = min(cx, W - cx, cy, H - cy)
    if dist_to_edge < 0.1 * min(H, W):
        confidence = "low"
    elif dist_to_edge < 0.2 * min(H, W):
        confidence = "medium"
    else:
        confidence = "high"

    if organ == "breast":
        mid_x, mid_y = W / 2, H / 2
        if abs(cx - mid_x) < W * 0.1 and abs(cy - mid_y) < H * 0.1:
            return "central", confidence
        vertical = "upper" if cy < mid_y else "lower"
        if laterality == "right":
            horizontal = "outer" if cx < mid_x else "inner"
        elif laterality == "left":
            horizontal = "inner" if cx < mid_x else "outer"
        else:
            horizontal = "outer-or-inner"
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


def postprocess_mask(
    mask_png_base64: str,
    original_size: Tuple[int, int],
    organ: str = "breast",
    pixel_spacing_mm: Optional[float] = None,
    laterality: Optional[str] = None,
) -> dict:
    """
    Decodes the mask from a base64 PNG -> returns all SpatialDerived fields.

    Raises:
        ValueError: if the base64 cannot be decoded, or the bytes are not a valid PNG.
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
        return _empty_spatial(original_size)

    quadrant, loc_confidence = get_location_quadrant(
        spatial["centroid"], original_size, organ, laterality
    )

    return {
        "bbox": spatial["bbox"],
        "area_cm2": spatial["area_cm2"],
        "pixel_spacing_reliable": spatial["pixel_spacing_reliable"],
        "centroid": spatial["centroid"],
        "location_quadrant": quadrant,
        "aspect_ratio": spatial["aspect_ratio"],
        "aspect_ratio_interpretation": spatial["aspect_ratio_interpretation"],
        "circularity": spatial["circularity"],
        "width_px": spatial["width_px"],
        "height_px": spatial["height_px"],
        "location_confidence": loc_confidence,
    }


def _empty_spatial(image_size: Tuple[int, int]) -> dict:
    """Fallback for when no lesion is detected (normal image)."""
    H, W = image_size
    return {
        "bbox": [0, 0, 0, 0],
        "area_cm2": None,
        "pixel_spacing_reliable": False,
        "centroid": [W // 2, H // 2],
        "location_quadrant": "none",
        "aspect_ratio": 1.0,
        "aspect_ratio_interpretation": "",
        "circularity": 1.0,
        "width_px": 0,
        "height_px": 0,
        "location_confidence": "low",
    }
