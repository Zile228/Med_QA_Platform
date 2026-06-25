"""
services/vision/_vision_helpers.py
====================================
Shared bottleneck enrichment, Grad-CAM, texture, and uncertainty helpers
used by both us_breast/model.py and us_thyroid/model.py.

No imports from arch.py or model.py -- safe for any container that copies
this file alongside model.py.
"""

import base64
import threading
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2

_dropout_locks: dict = {}
_dropout_locks_guard = threading.Lock()


def _get_dropout_lock(model) -> threading.Lock:
    """
    Returns a lock unique to this model instance, so concurrent requests
    sharing the same model cannot toggle each other's Dropout train/eval
    state mid-inference.
    """
    key = id(model)
    with _dropout_locks_guard:
        if key not in _dropout_locks:
            _dropout_locks[key] = threading.Lock()
        return _dropout_locks[key]


def extract_enriched_bottleneck(bottleneck_tensor: torch.Tensor) -> dict:
    """
    Compute clinically meaningful spatial statistics from the encoder bottleneck.

    Args:
        bottleneck_tensor: (1, 448, Hf, Wf) -- typically (1, 448, 7, 7).
    """
    with torch.no_grad():
        feat = bottleneck_tensor.squeeze(0)
        _, Hf, Wf = feat.shape

        activation_energy = round(float(feat.pow(2).mean()), 4)

        spatial_map = feat.mean(dim=0)

        ch, cw = Hf // 2, Wf // 2
        r = max(1, min(Hf, Wf) // 4)
        h0, h1 = max(0, ch - r), min(Hf, ch + r + 1)
        w0, w1 = max(0, cw - r), min(Wf, cw + r + 1)
        centre_mean = float(spatial_map[h0:h1, w0:w1].mean())

        border_mask = torch.ones(Hf, Wf, dtype=torch.bool)
        border_mask[h0:h1, w0:w1] = False
        border_vals = spatial_map[border_mask]
        border_mean = float(border_vals.mean()) if border_vals.numel() > 0 else 0.0

        activation_scale = float(spatial_map.abs().mean()) + 1e-6
        denom_floor = 0.05 * activation_scale
        center_periphery_ratio = round(
            abs(centre_mean) / (abs(border_mean) + denom_floor), 4
        )

        flat = spatial_map.flatten()
        p = torch.softmax(flat, dim=0)
        entropy = round(float(-(p * (p + 1e-8).log()).sum()), 4)

        hm, wm = Hf // 2, Wf // 2
        q_nw = round(float(spatial_map[:hm, :wm].mean()), 4)
        q_ne = round(float(spatial_map[:hm, wm:].mean()), 4)
        q_sw = round(float(spatial_map[hm:, :wm].mean()), 4)
        q_se = round(float(spatial_map[hm:, wm:].mean()), 4)

        channel_mean = feat.mean(dim=(1, 2))
        top_channels = [round(v, 4) for v in channel_mean.topk(10).values.tolist()]

    return {
        "activation_energy": activation_energy,
        "center_periphery_ratio": center_periphery_ratio,
        "spatial_entropy": entropy,
        "quadrant_activations": {"nw": q_nw, "ne": q_ne, "sw": q_sw, "se": q_se},
        "top_channel_activations": top_channels,
    }


def _compute_gradcam(model, tensor: torch.Tensor, target_class_idx: int, original_size: tuple) -> tuple:
    """
    Compute Grad-CAM heatmap for target_class_idx.

    Uses a gradient hook on the live tensor from the forward hook -- not a
    detached clone -- so cls_output.backward() actually reaches the hook.

    Returns:
        cam_base64: str - base64-encoded PNG of the upsampled heatmap
        cam_np:     np.ndarray - raw (H_orig, W_orig) float32 in [0, 1]
    """
    activations: list = []
    gradients: list = []

    def fwd_hook(module, input, output):
        live_act = output[-1]
        activations.append(live_act.detach().clone())
        live_act.register_hook(lambda g: gradients.append(g.detach().clone()))

    handle = model.backbone.register_forward_hook(fwd_hook)

    with torch.enable_grad():
        seg_output, cls_output, _ = model(tensor)
        grad_out = torch.zeros_like(cls_output)
        grad_out[0, target_class_idx] = 1.0
        cls_output.backward(gradient=grad_out, retain_graph=False)

    handle.remove()

    act = activations[0] if activations else None
    grad = gradients[0] if gradients else None

    if act is None or grad is None:
        cam_np = np.zeros((original_size[0], original_size[1]), dtype=np.float32)
    else:
        weights = grad.squeeze(0).mean(dim=(1, 2))
        cam = (weights[:, None, None] * act.squeeze(0)).sum(dim=0)
        cam = torch.relu(cam)
        cam_max = cam.max()
        if cam_max > 1e-8:
            cam = cam / cam_max
        cam_np = cam.detach().cpu().numpy().astype(np.float32)
        H_orig, W_orig = original_size
        cam_np = cv2.resize(cam_np, (W_orig, H_orig))

    cam_uint8 = (np.clip(cam_np, 0.0, 1.0) * 255).astype(np.uint8)
    ok, buf = cv2.imencode(".png", cam_uint8)
    cam_base64 = base64.b64encode(buf.tobytes()).decode("ascii") if ok else ""

    return cam_base64, cam_np


def _compute_gradcam_mask_overlap(
    cam_np: np.ndarray,
    seg_output: torch.Tensor,
    original_size: tuple,
    cam_threshold: float = 0.5,
    mask_threshold: float = 0.5,
) -> dict:
    """
    Compare the Grad-CAM attention region vs the segmentation mask.

    iou < 0.3 means model attends to a very different region than it segmented
    (strong uncertainty signal).
    """
    cam_bin = (cam_np > cam_threshold).astype(np.uint8)

    mask_up = F.interpolate(
        seg_output.detach(),
        size=original_size,
        mode="bilinear",
        align_corners=False,
    )
    mask_bin = (mask_up.squeeze().cpu().numpy() > mask_threshold).astype(np.uint8)

    inter = int((cam_bin & mask_bin).sum())
    union = int((cam_bin | mask_bin).sum())
    iou = round(inter / (union + 1e-8), 3)

    return {
        "iou": iou,
        "interpretation": "high" if iou > 0.5 else ("medium" if iou > 0.3 else "low"),
    }


def _extract_texture_features(
    bottleneck_out: torch.Tensor,
    seg_output: torch.Tensor,
) -> dict:
    """
    Compare bottleneck activations inside vs outside the segmentation mask.

    internal_heterogeneity:     std of activations inside the mask per channel,
                                averaged over channels. High = heterogeneous lesion.
    lesion_background_contrast: mean inside - mean outside. High = lesion stands out.
    """
    feat = bottleneck_out.detach()
    while feat.dim() > 3:
        feat = feat.squeeze(0)
    C, Hf, Wf = feat.shape

    seg = seg_output.detach()
    while seg.dim() > 3:
        seg = seg.squeeze(0)
    if seg.dim() == 2:
        seg = seg.unsqueeze(0)

    mask_small = F.interpolate(
        seg.unsqueeze(0),
        size=(Hf, Wf),
        mode="nearest",
    ).squeeze()
    mask_bool = mask_small > 0.5

    if mask_bool.sum() < 2 or (~mask_bool).sum() < 1:
        return {"internal_heterogeneity": 0.0, "lesion_background_contrast": 0.0}

    feat_in = feat[:, mask_bool]
    feat_out = feat[:, ~mask_bool]

    return {
        "internal_heterogeneity": round(float(feat_in.std(dim=1).mean()), 4),
        "lesion_background_contrast": round(float(feat_in.mean() - feat_out.mean()), 4),
    }


def _enable_dropout_only(model) -> list:
    """
    Switches only nn.Dropout submodules to train mode, leaving BatchNorm
    and everything else in eval mode. Returns the list of toggled modules
    so the caller can restore them afterwards.

    model.train() is NOT used here -- it would also switch BatchNorm2d
    (in the decoder ConvBlocks and the EfficientNet-B4 backbone) into
    training mode, which computes single-sample batch statistics at
    batch size 1 and permanently corrupts the shared model's running
    mean/var buffers across requests.
    """
    toggled = []
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()
            toggled.append(module)
    return toggled


def _predict_with_uncertainty(model, tensor: torch.Tensor, cfg, n_passes: int = 10) -> dict:
    """
    Run n_passes stochastic forward passes with dropout active.

    Only Dropout submodules are switched to train mode (see _enable_dropout_only).
    BatchNorm stays in eval mode throughout, so running statistics on the shared
    model instance are never touched. A per-model lock serializes the toggled
    window so concurrent requests cannot race on shared dropout state.
    """
    lock = _get_dropout_lock(model)
    with lock:
        toggled = _enable_dropout_only(model)
        preds = []
        try:
            with torch.no_grad():
                for _ in range(n_passes):
                    _, cls_out, _ = model(tensor.detach())
                    preds.append(
                        F.softmax(cls_out, dim=1).squeeze(0).cpu().numpy()
                    )
        finally:
            for module in toggled:
                module.eval()

    preds_np = np.stack(preds)
    mean = preds_np.mean(axis=0)
    std = preds_np.std(axis=0)
    entropy = float(-np.sum(mean * np.log(mean + 1e-8)))

    return {
        "mean_confidence": [round(float(v), 4) for v in mean],
        "uncertainty": [round(float(v), 4) for v in std],
        "predictive_entropy": round(entropy, 4),
    }
