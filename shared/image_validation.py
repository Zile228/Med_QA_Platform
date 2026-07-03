"""
shared/image_validation.py
Shared upload validation used by every service that accepts an image
(router, vision, orchestrator). Centralizes size and dimension limits
so they can't drift out of sync between services.
"""

MAX_UPLOAD_BYTES = 15 * 1024 * 1024       # 15 MB
MAX_IMAGE_DIMENSION_PX = 8000              # reject absurd width/height (decompression-bomb guard)


class ImageValidationError(ValueError):
    """Raised when an uploaded image fails size/dimension checks."""


def check_upload_size(image_bytes: bytes) -> None:
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise ImageValidationError(
            f"Image too large ({len(image_bytes)} bytes). "
            f"Maximum allowed is {MAX_UPLOAD_BYTES} bytes."
        )


def check_image_dimensions(width: int, height: int) -> None:
    if width > MAX_IMAGE_DIMENSION_PX or height > MAX_IMAGE_DIMENSION_PX:
        raise ImageValidationError(
            f"Image dimensions {width}x{height} exceed the "
            f"{MAX_IMAGE_DIMENSION_PX}px limit."
        )
