"""
services/vision/us_breast/postprocess.py
==========================================
Mask (numpy array / base64 PNG) -> SpatialDerived fields.

Public API:
    extract_spatial_features(mask, pixel_spacing_mm) -> dict
    get_location_quadrant(centroid, image_size, organ) -> tuple[str, str]
    postprocess_mask(mask_png_base64, original_size, organ, pixel_spacing_mm) -> dict

Được gọi bởi knowledge/mapper.py - nhận mask dưới dạng base64 PNG qua HTTP body,
KHÔNG đọc từ path trên disk (vision và knowledge là 2 container riêng, không
share filesystem - xem ISSUES_AND_FIXES.md mục 1).
"""

import base64
import cv2
import numpy as np
from typing import Tuple


# Tinh spatial features tu mask

def extract_spatial_features(
    mask: np.ndarray,
    pixel_spacing_mm: float = 0.1,
) -> dict:
    """
    Tính toán spatial features từ binary mask.

    Args:
        mask:             binary mask (H, W), values 0 or 255
        pixel_spacing_mm: khoảng cách thực tế mỗi pixel (mm).
                          0.1mm/px là giá trị mặc định cho BUSI dataset.
                          1 cm² = 100 mm² -> area_px * spacing² / 100

    Returns dict với:
        bbox, area_cm2, centroid, width_px, height_px,
        aspect_ratio, circularity
    Trả về empty dict nếu không tìm thấy contour (ảnh normal/no lesion).
    """
    mask_uint8 = mask.astype(np.uint8)
    if mask_uint8.max() > 1:
        mask_uint8 = (mask_uint8 > 127).astype(np.uint8)

    contours, _ = cv2.findContours(
        mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return {}

    # Lay contour lon nhat, bo qua noise
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

    # Tinh toan cac shape descriptor
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


# Xac dinh quadrant cua khoi u

def get_location_quadrant(
    centroid: list,
    image_size: Tuple[int, int],
    organ: str = "breast",
) -> Tuple[str, str]:
    """
    Xác định quadrant của khối u từ centroid và image size.

    Args:
        centroid:   [cx, cy] pixel coordinates
        image_size: (H, W) của ảnh gốc
        organ:      'breast' | 'thyroid'

    Returns:
        (quadrant_str, confidence_str)

    Breast quadrants (theo convention lâm sàng):
        Upper-outer | Upper-inner | Lower-outer | Lower-inner | Central
        Trục: x < W/2 -> outer (nếu right breast, cần flip - POC giả sử right breast)
              y < H/2 -> upper

    Thyroid:
        Left-lobe | Right-lobe | Isthmus
        Trục: x < W*0.35 -> left-lobe, x > W*0.65 -> right-lobe, else isthmus
    """
    H, W = image_size
    cx, cy = centroid

    # Confidence dua theo khoang cach centroid den bien anh
    dist_to_edge = min(cx, W - cx, cy, H - cy)
    if dist_to_edge < 0.1 * min(H, W):
        confidence = "low"    # centroid gần edge -> quadrant không chắc
    elif dist_to_edge < 0.2 * min(H, W):
        confidence = "medium"
    else:
        confidence = "high"

    if organ == "breast":
        mid_x, mid_y = W / 2, H / 2
        # Margin zone quanh midline -> central
        if abs(cx - mid_x) < W * 0.1 and abs(cy - mid_y) < H * 0.1:
            return "central", confidence

        vertical   = "upper" if cy < mid_y else "lower"
        horizontal = "outer" if cx < mid_x else "inner"
        # Gia su right breast: outer = phia trai anh
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


# Ham postprocess tong hop

def postprocess_mask(
    mask_png_base64: str,
    original_size: Tuple[int, int],
    organ: str = "breast",
    pixel_spacing_mm: float = 0.1,
) -> dict:
    """
    Decode mask từ base64 PNG -> trả về toàn bộ SpatialDerived fields.

    Args:
        mask_png_base64:  mask PNG encode base64 (nhận trực tiếp qua HTTP body,
                           KHÔNG đọc từ path trên disk - vision và knowledge là
                           2 container riêng, không share filesystem)
        original_size:    (H, W) ảnh gốc
        organ:            'breast' | 'thyroid'
        pixel_spacing_mm: mm/pixel

    Returns dict map 1-1 vào SpatialDerived schema.

    Raises:
        ValueError: nếu base64 không decode được, hoặc bytes không phải PNG hợp lệ.
                    Fail loud, khong fallback im lang de tranh du lieu gia.
    """
    if not mask_png_base64:
        raise ValueError(
            "mask_png_base64 rỗng - vision service không trả về mask. "
            "Đây là lỗi upstream, không phải case 'normal/no lesion'."
        )

    try:
        mask_bytes = base64.b64decode(mask_png_base64, validate=True)
    except Exception as e:
        raise ValueError(f"Không decode được base64 của mask: {e}") from e

    mask_array = np.frombuffer(mask_bytes, dtype=np.uint8)
    mask = cv2.imdecode(mask_array, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(
            "cv2.imdecode trả về None - base64 decode được nhưng không phải PNG hợp lệ. "
            "Kiểm tra lại bước encode ở vision service."
        )

    spatial = extract_spatial_features(mask, pixel_spacing_mm)

    if not spatial:
        # Case hop le: mask OK nhung khong co contour (anh normal/no lesion)
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
    """Fallback khi không detect được lesion (ảnh normal)."""
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
