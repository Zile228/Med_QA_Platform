"""
services/vision/us_breast/model.py
Checkpoint loading + inference wrapper for UNet_MTL (breast).

Public API:
    load_model(checkpoint_path, device) -> (UNet_MTL, Config)
    run_inference(model, cfg, image_bytes) -> dict
"""

import os
import base64
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from torchvision import transforms
from PIL import Image

# Set DISABLE_XAI=true to skip Grad-CAM, MC-Dropout, and texture extraction.
# Cuts per-request compute by ~10-12x; output keys are still present but empty.
_DISABLE_XAI = os.getenv("DISABLE_XAI", "false").lower() == "true"

from .arch import Config, UNet_MTL
from shared.image_validation import check_image_dimensions, ImageValidationError
from services.vision._vision_helpers import (
    extract_enriched_bottleneck,
    _compute_gradcam,
    _compute_gradcam_mask_overlap,
    _extract_texture_features,
    _predict_with_uncertainty,
)


def load_model(checkpoint_path: str, device: str = None) -> tuple:
    """
    Load UNet_MTL from a checkpoint .pt file.

    Returns:
        (model, cfg) -- the model in eval mode, cfg holds MEAN/STD/IDX_TO_CLASS
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = Config()
    cfg.BACKBONE = "efficientnet_b4"
    cfg.CLASSIFICATION_HEAD = "fc"
    cfg.USE_Deform = False
    cfg.NUM_CLASSES = 3
    cfg.DEVICE = device

    model = UNet_MTL(cfg).to(device)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Place the mtl_effnet_fc_conv.pt file in models/checkpoints/"
        )

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"[vision/us_breast] Loaded checkpoint from {checkpoint_path} on {device}")
    return model, cfg


def _build_transform(cfg: Config) -> transforms.Compose:
    """Inference transform, uses BUSI mean/std (different from ImageNet)."""
    return transforms.Compose([
        transforms.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.MEAN, std=cfg.STD),
    ])


def _preprocess(image_bytes: bytes, cfg: Config) -> tuple:
    """
    Decode bytes -> PIL RGB -> tensor (1, 3, H, W).

    Returns:
        tensor: preprocessed input for the model
        original_size: (H, W) of the original image
    """
    img_array = np.frombuffer(image_bytes, dtype=np.uint8)
    img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Could not decode the image. Check the format (PNG/JPG).")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    original_size = (img_rgb.shape[0], img_rgb.shape[1])
    check_image_dimensions(width=original_size[1], height=original_size[0])

    pil_img = Image.fromarray(img_rgb)
    transform = _build_transform(cfg)
    tensor = transform(pil_img).unsqueeze(0)
    return tensor, original_size


def run_inference(
    model: UNet_MTL,
    cfg: Config,
    image_bytes: bytes,
) -> dict:
    """
    End-to-end inference: bytes -> mask + classification + enriched features.

    Returns a dict with keys matching the ModelOutput schema.
    """
    device = cfg.DEVICE
    tensor, original_size = _preprocess(image_bytes, cfg)
    tensor = tensor.to(device)

    with torch.no_grad():
        seg_output, cls_output, bottleneck_out = model(tensor)

    probs = F.softmax(cls_output, dim=1).squeeze(0)
    all_scores = {
        cls: round(float(probs[idx]), 4)
        for cls, idx in cfg.CLASS_TO_IDX.items()
    }
    top_idx = int(probs.argmax())
    top_label = cfg.IDX_TO_CLASS[top_idx]
    confidence = round(float(probs[top_idx]), 4)

    mask_upsampled = F.interpolate(
        seg_output,
        size=original_size,
        mode="bilinear",
        align_corners=False,
    ).squeeze(0).squeeze(0)
    mask_np = (mask_upsampled.cpu().numpy() > 0.5).astype(np.uint8) * 255

    ok, mask_png_bytes = cv2.imencode(".png", mask_np)
    if not ok:
        raise RuntimeError("Could not encode mask as PNG.")
    mask_png_base64 = base64.b64encode(mask_png_bytes.tobytes()).decode("ascii")

    bottleneck_enriched = extract_enriched_bottleneck(bottleneck_out.cpu())

    if _DISABLE_XAI:
        gradcam_base64 = ""
        gradcam_mask_overlap = {}
        texture_features = {}
        uncertainty = {}
    else:
        gradcam_base64, gradcam_np = _compute_gradcam(model, tensor, top_idx, original_size)
        gradcam_mask_overlap = _compute_gradcam_mask_overlap(gradcam_np, seg_output, original_size)
        texture_features = _extract_texture_features(bottleneck_out, seg_output)
        uncertainty = _predict_with_uncertainty(model, tensor, cfg)

    filtered_findings = [
        {"label": cls, "confidence": score, "reason": "below threshold 0.1"}
        for cls, score in all_scores.items()
        if score < 0.1
    ]

    return {
        "top_label": top_label,
        "confidence": confidence,
        "all_scores": all_scores,
        "mask_png_base64": mask_png_base64,
        "original_size": list(original_size),
        "bottleneck_enriched": bottleneck_enriched,
        "gradcam_png_base64": gradcam_base64,
        "gradcam_mask_overlap": gradcam_mask_overlap,
        "texture_features": texture_features,
        "uncertainty": uncertainty,
        "filtered_findings": filtered_findings,
    }
