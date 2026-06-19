"""
services/vision/us_thyroid/model.py
======================================
Load checkpoint + inference wrapper cho UNet_MTL (thyroid).

Public API:
    load_model(checkpoint_path, device)  -> (UNet_MTL, Config)
    run_inference(model, cfg, image_bytes) -> dict

Output dict có cùng schema với us_breast - main.py có thể dùng
cùng ModelOutput Pydantic schema mà không cần thay đổi.
"""

import os
import base64
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from torchvision import transforms
from PIL import Image
import io

from .arch import Config, UNet_MTL


# Load model tu checkpoint

def load_model(checkpoint_path: str, device: str = None):
    """
    Load UNet_MTL (thyroid) từ checkpoint .pt file.

    Args:
        checkpoint_path: path tới mtl_effnet_fc_conv_thyroid.pt
        device:          'cuda' | 'cpu' | None (auto-detect)

    Returns:
        (model, cfg) - model ở eval mode, cfg chứa MEAN/STD/IDX_TO_CLASS
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    cfg                      = Config()
    cfg.BACKBONE             = 'efficientnet_b4'
    cfg.CLASSIFICATION_HEAD  = 'fc'
    cfg.USE_Deform           = False
    cfg.NUM_CLASSES          = 2
    cfg.DEVICE               = device

    model = UNet_MTL(cfg).to(device)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint không tìm thấy: {checkpoint_path}\n"
            "Đặt file mtl_effnet_fc_conv_thyroid.pt vào models/checkpoints/"
        )

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"[vision/us_thyroid] Loaded checkpoint từ {checkpoint_path} on {device}")
    return model, cfg


# Tien xu ly anh dau vao

def _build_transform(cfg: Config) -> transforms.Compose:
    """Transform inference, dung TN3K mean/std (khac ImageNet)."""
    return transforms.Compose([
        transforms.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.MEAN, std=cfg.STD),
    ])


def _preprocess(image_bytes: bytes, cfg: Config) -> tuple:
    """
    Decode bytes -> PIL RGB -> tensor (1, 3, H, W).

    Returns:
        tensor:        preprocessed input cho model
        original_size: (H, W) ảnh gốc để upsample mask về
    """
    img_array    = np.frombuffer(image_bytes, dtype=np.uint8)
    img_bgr      = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    img_rgb      = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    original_size = (img_rgb.shape[0], img_rgb.shape[1])   # (H, W)

    pil_img   = Image.fromarray(img_rgb)
    transform = _build_transform(cfg)
    tensor    = transform(pil_img).unsqueeze(0)             # (1, 3, H, W)
    return tensor, original_size


# Bottleneck feature extraction (giống us_breast)

def extract_bottleneck_summary(bottleneck_tensor: torch.Tensor) -> dict:
    """
    Serialize bottleneck tensor (1, 448, H, W) thành text-readable dict.

    Returns:
        activation_energy:       float
        top_channel_activations: list[float] - top-10 channel means
        attention_hotspot_grid:  [row, col] - vị trí model tập trung nhất
    """
    with torch.no_grad():
        feat         = bottleneck_tensor.squeeze(0)          # (448, H, W)
        channel_mean = feat.mean(dim=(1, 2))                 # (448,)
        top_channels = channel_mean.topk(10).values.tolist()
        energy       = float(feat.pow(2).mean())
        spatial_max  = feat.max(dim=0).values                # (H, W)
        hotspot      = spatial_max.argmax().item()
        h, w         = spatial_max.shape
        hotspot_pos  = [hotspot // w, hotspot % w]

    return {
        'activation_energy':       round(energy, 4),
        'top_channel_activations': [round(v, 4) for v in top_channels],
        'attention_hotspot_grid':  hotspot_pos,
    }


# Ham inference chinh

def run_inference(
    model: UNet_MTL,
    cfg: Config,
    image_bytes: bytes,
) -> dict:
    """
    End-to-end inference: bytes -> mask (base64 PNG) + classification + bottleneck.

    Output dict có cùng key schema với us_breast - tương thích ModelOutput schema.

    Args:
        model:       loaded UNet_MTL (eval mode)
        cfg:         Config instance từ load_model
        image_bytes: raw bytes của ảnh upload

    Returns dict:
        top_label, confidence, all_scores,
        mask_png_base64, bottleneck_features, filtered_findings, original_size
    """
    device = cfg.DEVICE

    tensor, original_size = _preprocess(image_bytes, cfg)
    tensor = tensor.to(device)

    with torch.no_grad():
        seg_output, cls_output, bottleneck_out = model(tensor)

    # Softmax lay top label
    probs      = F.softmax(cls_output, dim=1).squeeze(0)    # (2,)
    all_scores = {
        cls: round(float(probs[idx]), 4)
        for cls, idx in cfg.CLASS_TO_IDX.items()
    }
    top_idx    = int(probs.argmax())
    top_label  = cfg.IDX_TO_CLASS[top_idx]
    confidence = round(float(probs[top_idx]), 4)

    # Upsample mask ve kich thuoc anh goc
    mask_upsampled = F.interpolate(
        seg_output,
        size=original_size,
        mode='bilinear',
        align_corners=False,
    ).squeeze(0).squeeze(0)                                  # (H_orig, W_orig)
    mask_np = (mask_upsampled.cpu().numpy() > 0.5).astype(np.uint8) * 255

    # Encode mask thanh PNG base64 (khong ghi file)
    ok, mask_png_bytes = cv2.imencode('.png', mask_np)
    if not ok:
        raise RuntimeError('Không encode được mask thành PNG.')
    mask_png_base64 = base64.b64encode(mask_png_bytes.tobytes()).decode('ascii')

    # Trich xuat bottleneck features
    bottleneck_features = extract_bottleneck_summary(bottleneck_out.cpu())

    # Labels co confidence < 0.1 duoc filter ra
    filtered_findings = [
        {'label': cls, 'confidence': score, 'reason': 'below threshold 0.1'}
        for cls, score in all_scores.items()
        if score < 0.1
    ]

    return {
        'top_label':          top_label,
        'confidence':         confidence,
        'all_scores':         all_scores,
        'mask_png_base64':    mask_png_base64,
        'bottleneck_features': bottleneck_features,
        'filtered_findings':  filtered_findings,
        'original_size':      original_size,
    }
