"""
services/router/model.py

Modality Router - EfficientNet-B0 classifier.

Classifies the input image into one of these modalities:
    - us_breast   (Breast Ultrasound)
    - us_thyroid  (Thyroid Ultrasound)
    - ood         (Out-of-distribution - reject)

Phase 2: add an xray class without changing the code, only add it to IDX_TO_CLASS.

Public API:
    load_router(checkpoint_path, device)   -> (model, transform)
    run_routing(model, transform, image_bytes, ood_threshold) -> dict
        -> maps 1-1 into the RoutingResult schema
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import timm
import torch.nn as nn
from torchvision import transforms
from PIL import Image

from shared.image_validation import check_image_dimensions


# Router configuration

ROUTER_CLASSES = {
    0: "us_breast",
    1: "us_thyroid",
    }
CLASS_TO_IDX = {v: k for k, v in ROUTER_CLASSES.items()}

# Maps module_key to modality and organ
MODALITY_MAP = {
    "us_breast":  {"modality": "ultrasound", "organ": "breast"},
    "us_thyroid": {"modality": "ultrasound", "organ": "thyroid"},
    "xray":       {"modality": "xray",       "organ": "chest"},
    "ood":        {"modality": "unknown",    "organ": "unknown"},
}

MODULE_KEY_MAP = {
    "us_breast":  "us_breast",
    "us_thyroid": "us_thyroid",
    "xray":       "xray",
    "ood":        "ood",
}

OOD_THRESHOLD = float(os.getenv("OOD_THRESHOLD", "0.6"))

# Router priority weight when combined with the user hint
ROUTER_HINT_ROUTER_WEIGHT = float(os.getenv("ROUTER_HINT_ROUTER_WEIGHT", "0.7"))

# Maps the user's organ hint to a module key
HINT_ORGAN_TO_MODULE_KEY = {
    "breast":  "us_breast",
    "thyroid": "us_thyroid",
}

# ImageNet mean/std (suitable for classifying image type, not anatomy)
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]
IMG_SIZE = 224   # EfficientNet-B0 native input


# Combines router confidence with the user hint

def resolve_with_hint(
    router_probs: dict,
    hint_module_key: str = None,
    router_weight: float = ROUTER_HINT_ROUTER_WEIGHT,
) -> dict:
    """
    Combines router_probs and the user hint using a linear weighting.

    score(class) = router_weight * router_probs[class]
                  + (1 - router_weight) * (1.0 if class == hint else 0.0)

    hint_conflict reflects whether the hint differs from the router's
    original top-1, independent of the final result.
    """
    router_top_key = max(router_probs, key=router_probs.get)

    if hint_module_key is None:
        return {
            "final_module_key": router_top_key,
            "hint_conflict": False,
            "hint_resolution_note": None,
            "final_decision_source": "router",
            "weighted_scores": dict(router_probs),
        }

    weighted_scores = {
        cls: round(
            router_weight * prob + (1 - router_weight) * (1.0 if cls == hint_module_key else 0.0),
            4,
        )
        for cls, prob in router_probs.items()
    }
    final_key = max(weighted_scores, key=weighted_scores.get)
    hint_conflict = hint_module_key != router_top_key

    if not hint_conflict:
        note = None
        source = "router"
    else:
        note = (
            f"Router predicted '{router_top_key}' (confidence {router_probs[router_top_key]:.0%}), "
            f"user selected hint '{hint_module_key}'. Combined using weight router_weight="
            f"{router_weight} to reach the final decision '{final_key}'."
        )
        source = "user_hint" if final_key == hint_module_key else "weighted"

    return {
        "final_module_key": final_key,
        "hint_conflict": hint_conflict,
        "hint_resolution_note": note,
        "final_decision_source": source,
        "weighted_scores": weighted_scores,
    }


# Model definition

class ModalityRouter(nn.Module):
    """
    EfficientNet-B0 + custom classifier head.
    Lightweight - only classifies modality, doesn't need a heavy backbone.
    """
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=False,
            num_classes=0,       # remove default head
        )
        in_features = self.backbone.num_features   # 1280 for eff-b0

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone.forward_features(x)   # (B, 1280, H, W)
        return self.classifier(feat)               # (B, num_classes)


# Load model and transform

def build_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])


def load_router(
    checkpoint_path: str,
    device: str = None,
):
    """
    Loads ModalityRouter from a checkpoint.

    If the checkpoint doesn't exist -> returns a model with random weights
    plus a warning (dev mode - the service can still start), and sets
    `degraded=True` so every response from run_routing() carries this flag.

    Returns:
        model:     ModalityRouter in eval mode
        transform: torchvision transform
        device:    str
        degraded:  bool - True if running with random weights
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    num_classes = len(ROUTER_CLASSES)
    model = ModalityRouter(num_classes=num_classes).to(device)

    degraded = not os.path.exists(checkpoint_path)
    if not degraded:
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"[router] Loaded checkpoint: {checkpoint_path}")
    else:
        print(
            f"[router] WARNING: checkpoint not found at {checkpoint_path}. "
            "Running with random weights - for dev/testing only."
        )

    model.eval()
    transform = build_transform()
    return model, transform, device, degraded


# Run inference

def run_routing(
    model: ModalityRouter,
    transform: transforms.Compose,
    device: str,
    image_bytes: bytes,
    ood_threshold: float = OOD_THRESHOLD,
    degraded: bool = False,
    hint_module_key: str = None,
) -> dict:
    """
    Classify image bytes -> RoutingResult fields.

    OOD logic:
        If max(softmax) < ood_threshold -> is_ood=True, module_key='ood'.
        The orchestrator rejects the request before forwarding it to the
        Vision service. The hint does not apply when is_ood=True -- the
        image doesn't belong to any class the model knows, so the hint has
        no basis to override.

    `hint_module_key`: 'us_breast' | 'us_thyroid' | None, already validated
    in main.py. If present, combined with router_probs via
    resolve_with_hint() to produce the final module_key, along with
    hint_conflict/hint_resolution_note/final_decision_source.

    `degraded=True` is attached to every response when the router runs with
    random weights.

    Returns a dict that maps 1-1 into the RoutingResult schema.
    """
    if degraded:
        print(
            "[router] WARNING: serving request with random weights - "
            "the routing decision has no statistical meaning."
        )

    # Decode bytes -> PIL
    img_array = np.frombuffer(image_bytes, dtype=np.uint8)
    img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Failed to decode image. Check the format (PNG/JPG).")
    check_image_dimensions(width=img_bgr.shape[1], height=img_bgr.shape[0])
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)

    # Preprocess
    tensor = transform(pil_img).unsqueeze(0).to(device)  # (1, 3, 224, 224)

    # Inference
    with torch.no_grad():
        logits = model(tensor)                            # (1, num_classes)
        probs = F.softmax(logits, dim=1).squeeze(0)      # (num_classes,)

    # Build all_scores
    all_scores = {
        ROUTER_CLASSES[i]: round(float(probs[i]), 4)
        for i in range(len(ROUTER_CLASSES))
    }

    top_idx = int(probs.argmax())
    top_key = ROUTER_CLASSES[top_idx]
    confidence = round(float(probs[top_idx]), 4)

    # OOD check
    is_ood = confidence < ood_threshold
    if is_ood:
        top_key = "ood"
        all_scores["ood"] = round(1 - confidence, 4)

    hint_result = {
        "hint_conflict": False,
        "hint_resolution_note": None,
        "final_decision_source": "router",
    }
    final_key = top_key

    if hint_module_key is not None and not is_ood:
        hint_result = resolve_with_hint(all_scores, hint_module_key)
        final_key = hint_result["final_module_key"]

    meta = MODALITY_MAP.get(final_key, MODALITY_MAP["ood"])

    return {
        "modality":         meta["modality"],
        "organ":            meta["organ"],
        "confidence":       confidence,
        "all_scores":       all_scores,
        "is_ood":           is_ood,
        "module_key":       MODULE_KEY_MAP.get(final_key, "ood"),
        "router_degraded":  degraded,
        "user_hint_modality": MODALITY_MAP.get(hint_module_key, {}).get("modality") if hint_module_key else None,
        "user_hint_organ":    MODALITY_MAP.get(hint_module_key, {}).get("organ") if hint_module_key else None,
        "hint_conflict":           hint_result["hint_conflict"],
        "hint_resolution_note":    hint_result["hint_resolution_note"],
        "final_decision_source":   hint_result["final_decision_source"],
    }
