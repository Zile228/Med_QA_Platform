"""
shared/schemas.py - Unified Output Schema
==========================================
File quan trọng nhất của project. Mọi service đều import từ đây.
Không sửa schema mà không báo - thay đổi ở đây ảnh hưởng toàn bộ pipeline.
"""

from pydantic import BaseModel, Field
from typing import Optional


# Layer 1: Router Output

class RoutingResult(BaseModel):
    """Output của Router Service (Layer 1)."""

    modality: str = Field(..., description="'ultrasound' | 'xray' | 'ood'")
    organ: str = Field(..., description="'breast' | 'thyroid' | 'heart' | 'chest' | 'unknown'")
    confidence: float = Field(..., description="Confidence của routing decision [0, 1]")
    all_scores: dict = Field(
        ...,
        description="Softmax scores tất cả class, vd: {'us_breast': 0.91, 'us_thyroid': 0.07}"
    )
    is_ood: bool = Field(
        default=False,
        description="True nếu confidence < threshold -> reject trước khi vào vision"
    )
    module_key: str = Field(
        ...,
        description="Key trong module_registry.yaml, vd: 'us_breast_v1'"
    )
    router_degraded: bool = Field(
        default=False,
        description=(
            "True nếu router đang chạy với random weights (checkpoint không tồn tại). "
            "Khi True, routing decision không có ý nghĩa thống kê - orchestrator PHẢI "
            "chặn pipeline thay vì tiếp tục chạy với routing ngẫu nhiên."
        ),
    )
    user_hint_modality: Optional[str] = Field(
        default=None,
        description="Modality user chọn thủ công, None nếu chọn 'Tự động'",
    )
    user_hint_organ: Optional[str] = Field(
        default=None,
        description="Organ user chọn thủ công, None nếu chọn 'Tự động'",
    )
    hint_conflict: bool = Field(
        default=False,
        description="True nếu hint của user khác top-1 prediction gốc của router",
    )
    hint_resolution_note: Optional[str] = Field(
        default=None,
        description="Giải thích cách router_confidence và user hint được kết hợp để ra quyết định cuối",
    )
    final_decision_source: str = Field(
        default="router",
        description="'router' | 'user_hint' | 'weighted' - nguồn quyết định cuối cùng cho module_key",
    )


# Layer 2: Vision Module Output

class ModelOutput(BaseModel):
    """Raw output từ UNet_MTL inference (Layer 2)."""

    top_label: str = Field(
        ...,
        description="'benign' | 'malignant' | 'normal'"
    )
    confidence: float = Field(..., description="Confidence của top_label [0, 1]")
    all_scores: dict = Field(
        ...,
        description="{'benign': 0.1, 'malignant': 0.87, 'normal': 0.03}"
    )
    mask_png_base64: str = Field(
        ...,
        description=(
            "Binary mask PNG, encoded base64. Truyền trực tiếp qua HTTP body - "
            "KHÔNG dùng path trên disk vì vision/knowledge là 2 container riêng, "
            "không share filesystem. Đây là cơ chế stateless duy nhất được hỗ trợ."
        ),
    )
    bottleneck_features: dict = Field(
        ...,
        description=(
            "Summary statistics từ encoder bottleneck (7×7×448). "
            "Keys: activation_energy, top_channel_activations, attention_hotspot_grid"
        )
    )
    original_size: list = Field(
        default=[512, 512],
        description="[H, W] của ảnh gốc - dùng để knowledge service tính spatial features"
    )


# Layer 3: Knowledge Mapper Output

class KnowledgeMapped(BaseModel):
    """Clinical knowledge enrichment (Layer 3)."""

    description: str = Field(
        ...,
        description="Text mô tả từ label + RAG context"
    )
    severity: str = Field(
        ...,
        description="'incidental' | 'significant' | 'urgent' | 'critical'"
    )
    severity_level: int = Field(
        ...,
        ge=1, le=4,
        description="1=incidental, 2=significant, 3=urgent, 4=critical"
    )
    icd10_hint: str = Field(
        ...,
        description="Breast: 'N63.0' (benign) / 'C50.9' (malignant). Thyroid: 'E04.9' / 'C73'"
    )
    risk_category: str = Field(
        ...,
        description=(
            "Breast: 'Low risk (BI-RADS 2-3)' | 'High risk (BI-RADS 4C-5)'. "
            "Thyroid: 'TI-RADS 3' | 'TI-RADS 5'. "
            "Không lookup JSON - derive từ label + confidence"
        )
    )
    confidence_calibration_note: Optional[str] = Field(
        default=None,
        description=(
            "Set khi confidence vượt ngưỡng nghi ngờ overfitting (vd >= 0.999) trên dataset "
            "nhỏ chưa qua calibration. Cảnh báo này KHÔNG phải bug - chỉ để LLM/clinician "
            "không đọc số % như độ tin cậy tuyệt đối."
        ),
    )


class SpatialDerived(BaseModel):
    """Spatial features từ segmentation mask (Layer 3, cv2.boundingRect)."""

    bbox: list = Field(..., description="[x1, y1, x2, y2] pixel coordinates")
    area_cm2: float = Field(..., description="Diện tích khối u tính bằng cm² (dùng pixel_spacing)")
    centroid: list = Field(..., description="[cx, cy] pixel coordinates")
    location_quadrant: str = Field(
        ...,
        description=(
            "Breast: 'upper-inner' | 'upper-outer' | 'lower-inner' | 'lower-outer' | 'central'. "
            "Thyroid: 'left-lobe' | 'right-lobe' | 'isthmus'"
        )
    )
    aspect_ratio: float = Field(
        ...,
        description="width/height. >1.5 -> elongated (suspicious for malignancy)"
    )
    circularity: float = Field(
        ...,
        description="4π·area/perimeter². 1.0=perfect circle, <0.5=irregular margin (suspicious)"
    )
    width_px: int
    height_px: int
    location_confidence: str = Field(
        ...,
        description="'low' | 'medium' | 'high' - dựa trên mask quality"
    )


# Schema thong nhat qua toan bo pipeline

class UnifiedOutput(BaseModel):
    """
    Schema chuẩn hóa output qua mọi layer.
    QA Agent chỉ đọc schema này - không biết ảnh từ đâu.
    Thêm modality mới = thêm pipeline, không sửa schema.
    """

    modality: str = Field(..., description="'ultrasound' | 'xray'")
    organ: str = Field(..., description="'breast' | 'thyroid' | 'heart' | 'chest'")
    image_id: str = Field(..., description="Unique ID cho request (UUID hoặc filename)")

    model_output: ModelOutput
    knowledge_mapped: KnowledgeMapped
    spatial_derived: SpatialDerived

    filtered_findings: list = Field(
        default_factory=list,
        description=(
            "Findings bị drop do confidence thấp - vẫn giữ lại để LLM biết. "
            "Vd: [{'label': 'normal', 'confidence': 0.03, 'reason': 'below threshold 0.1'}]"
        )
    )
    coverage_note: str = Field(
        default="Model trained on BUSI dataset (benign/malignant/normal only).",
        description="Cảnh báo về training data scope để LLM không over-generalize"
    )


# Layer 4: Final Report Output

class Tier1Structured(BaseModel):
    """Tier 1: Structured fields - dễ parse, hiển thị UI."""

    modality: str
    organ: str
    label: str
    confidence: float
    risk_category: str
    severity: str
    severity_level: int
    icd10_hint: str
    location_quadrant: str
    bbox: list = Field(
        default_factory=lambda: [0, 0, 0, 0],
        description="[x1, y1, x2, y2] pixel coordinates - dùng để vẽ overlay trên UI",
    )
    area_cm2: float
    aspect_ratio: float
    circularity: float
    confidence_calibration_note: Optional[str] = Field(
        default=None,
        description="Copy từ KnowledgeMapped - hiển thị trực tiếp ở Tier 1 để UI dễ render banner.",
    )
    hint_conflict: bool = Field(
        default=False,
        description="Copy từ RoutingResult - True nếu user hint khác router top-1 prediction",
    )
    hint_resolution_note: Optional[str] = Field(
        default=None,
        description="Copy từ RoutingResult - giải thích cách hint và router confidence được kết hợp",
    )


class ReportOutput(BaseModel):
    """Final output từ Orchestrator - LLM-generated 3-tier report."""

    image_id: str
    tier_1_structured: Tier1Structured = Field(
        ...,
        description="Structured data fields - parse từ UnifiedOutput, không cần LLM"
    )
    tier_2_radiological_description: str = Field(
        ...,
        description=(
            "LLM-generated: mô tả radiological bằng ngôn ngữ tự nhiên. "
            "Vd: 'A 1.24 cm² hypoechoic lesion with irregular margins (circularity: 0.42)...'"
        )
    )
    tier_3_diagnostic_suggestion: str = Field(
        ...,
        description=(
            "LLM-generated: gợi ý chẩn đoán + follow-up. "
            "KHÔNG phải chẩn đoán cuối cùng - AI assist only."
        )
    )
    disclaimer: str = Field(
        default=(
            "This AI-generated report is for screening assistance only and does not constitute "
            "a medical diagnosis. All findings must be reviewed and confirmed by a qualified "
            "radiologist or physician."
        )
    )
    rag_sources: list = Field(
        default_factory=list,
        description="Danh sách PDF sources dùng trong RAG retrieval"
    )
    rag_disabled_warning: Optional[str] = Field(
        default=None,
        description=(
            "Set khi FAISS index chưa build / chưa có PDF nào được index. "
            "Nếu không None, report được sinh ra KHÔNG có clinical guideline retrieval - "
            "chỉ dựa trên classification label + hardcode mapping. UI phải hiển thị banner cảnh báo."
        ),
    )
