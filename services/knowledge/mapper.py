"""
services/knowledge/mapper.py
==============================
Layer 3 - Knowledge Mapper.

Nhận ModelOutput + RoutingResult -> trả về KnowledgeMapped + SpatialDerived.

Không cần ML - thuần rule-based + hardcode clinical knowledge.
Design: dễ audit, dễ sửa bởi clinician, không có black box.

Public API:
    map_knowledge(modality, organ, top_label, confidence, all_scores) -> dict
        -> map 1-1 vào KnowledgeMapped schema

    derive_spatial(mask_png_base64, original_size, organ) -> dict
        -> map 1-1 vào SpatialDerived schema
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from services.vision.us_breast.postprocess import postprocess_mask as _postprocess_breast
from services.vision.us_thyroid.postprocess import postprocess_mask as _postprocess_thyroid

_POSTPROCESS_BY_ORGAN = {
    "breast": _postprocess_breast,
    "thyroid": _postprocess_thyroid,
}


# Bang tra cuu kien thuc lam sang

# Breast: anh xa BI-RADS theo label
BIRADS_MAP = {
    "normal":    {"birads": "BI-RADS 1", "risk_category": "Negative (BI-RADS 1)"},
    "benign":    {"birads": "BI-RADS 3", "risk_category": "Probably benign (BI-RADS 3)"},
    "malignant": {"birads": "BI-RADS 4C–5", "risk_category": "High suspicion (BI-RADS 4C–5)"},
}

# Thyroid: anh xa TI-RADS theo label
TIRADS_MAP = {
    "normal":    {"tirads": "TI-RADS 1", "risk_category": "Normal thyroid (TI-RADS 1)"},
    "benign":    {"tirads": "TI-RADS 3", "risk_category": "Mildly suspicious (TI-RADS 3)"},
    "malignant": {"tirads": "TI-RADS 5", "risk_category": "Highly suspicious (TI-RADS 5)"},
}

# Muc do nghiem trong dung chung cho moi modality (1=nhe, 4=nguy cap)
SEVERITY_MAP = {
    "normal":    {"severity": "incidental",   "severity_level": 1},
    "benign":    {"severity": "significant",  "severity_level": 2},
    "malignant": {"severity": "urgent",       "severity_level": 3},
}

# Ma ICD-10 tuong ung theo organ va label
ICD10_MAP = {
    "breast": {
        "normal":    "Z12.31",   # Encounter for screening mammogram
        "benign":    "N63.0",    # Unspecified lump in breast
        "malignant": "C50.9",    # Malignant neoplasm of breast, unspecified
    },
    "thyroid": {
        "normal":    "E04.9",    # Nontoxic goiter, unspecified
        "benign":    "E04.1",    # Nontoxic single thyroid nodule
        "malignant": "C73",      # Malignant neoplasm of thyroid gland
    },
}

# Mo ta lam sang ngan gon cho LLM
DESCRIPTION_MAP = {
    ("breast", "normal"):    (
        "No sonographic evidence of focal lesion. Breast parenchyma appears within normal limits."
    ),
    ("breast", "benign"):    (
        "Hypoechoic or isoechoic lesion with well-defined margins and oval/round shape. "
        "Features are consistent with a benign process such as fibroadenoma or cyst. "
        "Short-interval follow-up recommended per BI-RADS 3 guidelines."
    ),
    ("breast", "malignant"): (
        "Irregular hypoechoic mass with ill-defined or spiculated margins. "
        "Posterior acoustic shadowing may be present. "
        "Findings are highly suspicious for malignancy (BI-RADS 4C–5). "
        "Tissue sampling is strongly recommended."
    ),
    ("thyroid", "normal"):   (
        "Thyroid gland appears homogeneous with no discrete nodule identified."
    ),
    ("thyroid", "benign"):   (
        "Mildly hypoechoic nodule with smooth margins. "
        "Low to intermediate suspicion per ACR TI-RADS 3. "
        "Follow-up ultrasound in 1–2 years recommended."
    ),
    ("thyroid", "malignant"): (
        "Solid hypoechoic nodule with irregular margins, micro-calcifications, or taller-than-wide shape. "
        "Highly suspicious for malignancy per ACR TI-RADS 5. "
        "Fine-needle aspiration biopsy (FNA) is indicated."
    ),
}


# Nguong canh bao khi confidence qua cao (co the chua calibrate)
CONFIDENCE_CALIBRATION_THRESHOLD = float(
    os.getenv("CONFIDENCE_CALIBRATION_THRESHOLD", "0.999")
)


def _maybe_calibration_note(confidence: float) -> str | None:
    if confidence >= CONFIDENCE_CALIBRATION_THRESHOLD:
        return (
            f"Confidence {confidence:.2%} is unusually high for a model trained on a "
            "small dataset (BUSI, 780 images) without verified calibration. Treat this "
            "number as a relative ranking signal, not a calibrated probability."
        )
    return None

def _maybe_escalate_severity(
    base_severity: str,
    base_level: int,
    label: str,
    confidence: float,
) -> tuple:
    """
    Escalate len "critical" neu malignant + confidence >= 0.9.
    Downgrade xuong "significant" neu malignant + confidence < 0.5.
    """
    if label == "malignant":
        if confidence >= 0.9:
            return "critical", 4
        elif confidence < 0.5:
            return "significant", 2
    return base_severity, base_level



def map_knowledge(
    modality: str,
    organ: str,
    top_label: str,
    confidence: float,
    all_scores: dict,
) -> dict:
    """
    Rule-based clinical knowledge mapping.

    Args:
        modality:   'ultrasound' | 'xray'
        organ:      'breast' | 'thyroid'
        top_label:  'benign' | 'malignant' | 'normal'
        confidence: float [0, 1]
        all_scores: dict từ ModelOutput

    Returns dict map 1-1 vào KnowledgeMapped schema.
    """
    label = top_label.lower()
    organ = organ.lower()

    # Severity
    sev = SEVERITY_MAP.get(label, {"severity": "incidental", "severity_level": 1})
    severity, severity_level = _maybe_escalate_severity(
        sev["severity"], sev["severity_level"], label, confidence
    )

    # Risk category - organ specific
    if organ == "breast":
        risk_info = BIRADS_MAP.get(label, BIRADS_MAP["benign"])
        risk_category = risk_info["risk_category"]
    elif organ == "thyroid":
        risk_info = TIRADS_MAP.get(label, TIRADS_MAP["benign"])
        risk_category = risk_info["risk_category"]
    else:
        risk_category = f"Unknown organ: {organ}"

    # ICD-10
    icd10 = (
        ICD10_MAP
        .get(organ, ICD10_MAP["breast"])
        .get(label, "R93.8")   # R93.8: Other abnormal findings on diagnostic imaging
    )

    # Description
    description = DESCRIPTION_MAP.get(
        (organ, label),
        f"No clinical description available for {organ}/{label}."
    )

    return {
        "description":    description,
        "severity":       severity,
        "severity_level": severity_level,
        "icd10_hint":     icd10,
        "risk_category":  risk_category,
        "confidence_calibration_note": _maybe_calibration_note(confidence),
    }


def derive_spatial(
    mask_png_base64: str,
    original_size: tuple,
    organ: str = "breast",
    pixel_spacing_mm: float = 0.1,
) -> dict:
    """
    Wrapper sang postprocess_mask từ Vision service.

    Mask nhan qua HTTP body dang base64 PNG, khong dung path tren disk.

    Args:
        mask_png_base64:  mask PNG encode base64 (từ ModelOutput.mask_png_base64)
        original_size:    (H, W) ảnh gốc
        organ:            'breast' | 'thyroid'
        pixel_spacing_mm: mm/pixel

    Returns dict map 1-1 vào SpatialDerived schema.

    Raises:
        ValueError: neu mask khong decode duoc hoac organ khong hop le.
    """
    organ_key = organ.lower()
    fn = _POSTPROCESS_BY_ORGAN.get(organ_key)
    if fn is None:
        raise ValueError(
            f"Không có postprocess cho organ='{organ}'. Hỗ trợ: "
            f"{list(_POSTPROCESS_BY_ORGAN.keys())}"
        )
    return fn(
        mask_png_base64=mask_png_base64,
        original_size=tuple(original_size),
        organ=organ_key,
        pixel_spacing_mm=pixel_spacing_mm,
    )
