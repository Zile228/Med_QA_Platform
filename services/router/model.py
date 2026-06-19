"""
services/router/model.py
=========================
Modality Router - EfficientNet-B0 classifier.

Phân loại ảnh đầu vào vào 1 trong các modality:
    - us_breast   (Breast Ultrasound)
    - us_thyroid  (Thyroid Ultrasound)
    - ood         (Out-of-distribution - reject)

Phase 2: thêm xray class mà không sửa code, chỉ thêm class vào IDX_TO_CLASS.

Public API:
    load_router(checkpoint_path, device)   -> (model, transform)
    run_routing(model, transform, image_bytes, ood_threshold) -> dict
        -> map 1-1 vào RoutingResult schema
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


# Cau hinh cho router

ROUTER_CLASSES = {
    0: "us_breast",
    1: "us_thyroid",
    }
CLASS_TO_IDX = {v: k for k, v in ROUTER_CLASSES.items()}

# Map module_key sang modality va organ
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

# Trong so uu tien router khi ket hop voi user hint
ROUTER_HINT_ROUTER_WEIGHT = float(os.getenv("ROUTER_HINT_ROUTER_WEIGHT", "0.7"))

# Map organ hint cua user sang module key
HINT_ORGAN_TO_MODULE_KEY = {
    "breast":  "us_breast",
    "thyroid": "us_thyroid",
}

# ImageNet mean/std (phu hop cho classify loai anh, khong phai anatomy)
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]
IMG_SIZE = 224   # EfficientNet-B0 native input


# Kết hợp router confidence với user hint

def resolve_with_hint(
    router_probs: dict,
    hint_module_key: str = None,
    router_weight: float = ROUTER_HINT_ROUTER_WEIGHT,
) -> dict:
    """
    Ket hop router_probs va user hint theo trong so tuyen tinh.

    score(class) = router_weight * router_probs[class]
                  + (1 - router_weight) * (1.0 neu class == hint else 0.0)

    hint_conflict phan anh hint co khac top-1 goc cua router khong,
    doc lap voi ket qua cuoi cung.
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
            f"Router dự đoán '{router_top_key}' (confidence {router_probs[router_top_key]:.0%}), "
            f"user chọn hint '{hint_module_key}'. Kết hợp theo trọng số router_weight="
            f"{router_weight} ra quyết định cuối '{final_key}'."
        )
        source = "user_hint" if final_key == hint_module_key else "weighted"

    return {
        "final_module_key": final_key,
        "hint_conflict": hint_conflict,
        "hint_resolution_note": note,
        "final_decision_source": source,
        "weighted_scores": weighted_scores,
    }


# Dinh nghia model

class ModalityRouter(nn.Module):
    """
    EfficientNet-B0 + custom classifier head.
    Lightweight - chỉ phân loại modality, không cần heavy backbone.
    """
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=False,
            num_classes=0,       # remove default head
        )
        in_features = self.backbone.num_features   # 1280 với eff-b0

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


# Load model va transform

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
    Load ModalityRouter từ checkpoint.

    Nếu checkpoint không tồn tại -> trả về model với random weights
    kèm warning (dev mode - service vẫn start được), và đánh dấu
    `degraded=True` để mọi response từ run_routing() mang theo flag này.

    Returns:
        model:     ModalityRouter ở eval mode
        transform: torchvision transform
        device:    str
        degraded:  bool - True nếu đang chạy random weights
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
            f"[router] WARNING: checkpoint không tìm thấy tại {checkpoint_path}. "
            "Chạy với random weights - chỉ dùng cho dev/testing."
        )

    model.eval()
    transform = build_transform()
    return model, transform, device, degraded


# Chay inference

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
        Nếu max(softmax) < ood_threshold -> is_ood=True, module_key='ood'
        Orchestrator sẽ reject request trước khi gửi sang Vision service.
        Hint không áp dụng khi is_ood=True - ảnh không thuộc class nào model
        biết, hint không có cơ sở để override.

    `hint_module_key`: 'us_breast' | 'us_thyroid' | None, đã validate ở main.py.
    Nếu có, kết hợp với router_probs qua resolve_with_hint() để ra module_key
    cuối cùng, kèm hint_conflict/hint_resolution_note/final_decision_source.

    `degraded=True` duoc gan vao moi response khi router chay voi random weights.

    Tra ve dict map 1-1 vao RoutingResult schema.
    """
    if degraded:
        print(
            "[router] WARNING: serving request với random weights - "
            "routing decision không có ý nghĩa thống kê."
        )

    # Decode bytes -> PIL
    img_array = np.frombuffer(image_bytes, dtype=np.uint8)
    img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Không decode được ảnh. Kiểm tra format (PNG/JPG).")
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
