"""
shared/schemas.py - Unified Output Schema
==========================================
File quan trong nhat cua project. Moi service deu import tu day.
Khong sua schema ma khong bao - thay doi o day anh huong toan bo pipeline.
"""

from pydantic import BaseModel, Field
from typing import Optional, List


# Layer 1: Router Output

class RoutingResult(BaseModel):
    """Output cua Router Service (Layer 1)."""

    modality: str = Field(..., description="'ultrasound' | 'xray' | 'ood'")
    organ: str = Field(..., description="'breast' | 'thyroid' | 'heart' | 'chest' | 'unknown'")
    confidence: float = Field(..., description="Confidence cua routing decision [0, 1]")
    all_scores: dict = Field(
        ...,
        description="Softmax scores tat ca class, vd: {'us_breast': 0.91, 'us_thyroid': 0.07}"
    )
    is_ood: bool = Field(
        default=False,
        description="True neu confidence < threshold -> reject truoc khi vao vision"
    )
    module_key: str = Field(
        ...,
        description="Key trong module_registry.yaml, vd: 'us_breast_v1'"
    )
    router_degraded: bool = Field(
        default=False,
        description=(
            "True neu router dang chay voi random weights (checkpoint khong ton tai). "
            "Khi True, routing decision khong co y nghia thong ke - orchestrator PHAI "
            "chan pipeline thay vi tiep tuc chay voi routing ngau nhien."
        ),
    )
    user_hint_modality: Optional[str] = Field(
        default=None,
        description="Modality user chon thu cong, None neu chon 'Tu dong'",
    )
    user_hint_organ: Optional[str] = Field(
        default=None,
        description="Organ user chon thu cong, None neu chon 'Tu dong'",
    )
    hint_conflict: bool = Field(
        default=False,
        description="True neu hint cua user khac top-1 prediction goc cua router",
    )
    hint_resolution_note: Optional[str] = Field(
        default=None,
        description="Giai thich cach router_confidence va user hint duoc ket hop de ra quyet dinh cuoi",
    )
    final_decision_source: str = Field(
        default="router",
        description="'router' | 'user_hint' | 'weighted' - nguon quyet dinh cuoi cung cho module_key",
    )


# Layer 2: Vision Module Output

class ModelOutput(BaseModel):
    """Raw output tu UNet_MTL inference (Layer 2)."""

    top_label: str = Field(
        ...,
        description="'benign' | 'malignant' | 'normal'"
    )
    confidence: float = Field(..., description="Confidence cua top_label [0, 1]")
    all_scores: dict = Field(
        ...,
        description="{'benign': 0.1, 'malignant': 0.87, 'normal': 0.03}"
    )
    mask_png_base64: str = Field(
        ...,
        description=(
            "Binary mask PNG, encoded base64. Truyen truc tiep qua HTTP body - "
            "KHONG dung path tren disk vi vision/knowledge la 2 container rieng, "
            "khong share filesystem. Day la co che stateless duy nhat duoc ho tro."
        ),
    )
    bottleneck_features: dict = Field(
        ...,
        description=(
            "Summary statistics tu encoder bottleneck (7x7x448). "
            "Keys: activation_energy, top_channel_activations, attention_hotspot_grid"
        )
    )
    original_size: list = Field(
        default=[512, 512],
        description="[H, W] cua anh goc - dung de knowledge service tinh spatial features"
    )


# Layer 3: Knowledge Mapper Output

class KnowledgeMapped(BaseModel):
    """Clinical knowledge enrichment (Layer 3)."""

    description: str = Field(
        ...,
        description="Text mo ta tu label + RAG context"
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
            "Khong lookup JSON - derive tu label + confidence"
        )
    )
    confidence_calibration_note: Optional[str] = Field(
        default=None,
        description=(
            "Set khi confidence vuot nguong nghi ngo overfitting (vd >= 0.999) tren dataset "
            "nho chua qua calibration. Canh bao nay KHONG phai bug - chi de LLM/clinician "
            "khong doc so % nhu do tin cay tuyet doi."
        ),
    )


class SpatialDerived(BaseModel):
    """Spatial features tu segmentation mask (Layer 3, cv2.boundingRect)."""

    bbox: list = Field(..., description="[x1, y1, x2, y2] pixel coordinates")
    area_cm2: float = Field(..., description="Dien tich khoi u tinh bang cm2 (dung pixel_spacing)")
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
        description="4pi*area/perimeter^2. 1.0=perfect circle, <0.5=irregular margin (suspicious)"
    )
    width_px: int
    height_px: int
    location_confidence: str = Field(
        ...,
        description="'low' | 'medium' | 'high' - dua tren mask quality"
    )


# Schema thong nhat qua toan bo pipeline

class UnifiedOutput(BaseModel):
    """
    Schema chuan hoa output qua moi layer.
    QA Agent chi doc schema nay - khong biet anh tu dau.
    Them modality moi = them pipeline, khong sua schema.
    """

    modality: str = Field(..., description="'ultrasound' | 'xray'")
    organ: str = Field(..., description="'breast' | 'thyroid' | 'heart' | 'chest'")
    image_id: str = Field(..., description="Unique ID cho request (UUID hoac filename)")

    model_output: ModelOutput
    knowledge_mapped: KnowledgeMapped
    spatial_derived: SpatialDerived

    filtered_findings: list = Field(
        default_factory=list,
        description=(
            "Findings bi drop do confidence thap - van giu lai de LLM biet. "
            "Vd: [{'label': 'normal', 'confidence': 0.03, 'reason': 'below threshold 0.1'}]"
        )
    )
    coverage_note: str = Field(
        default="Model trained on BUSI dataset (benign/malignant/normal only).",
        description="Canh bao ve training data scope de LLM khong over-generalize"
    )


# Layer 4: Final Report Output

class Tier1Structured(BaseModel):
    """Tier 1: Structured fields - de parse, hien thi UI."""

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
        description="[x1, y1, x2, y2] pixel coordinates - dung de ve overlay tren UI",
    )
    area_cm2: float
    aspect_ratio: float
    circularity: float
    confidence_calibration_note: Optional[str] = Field(
        default=None,
        description="Copy tu KnowledgeMapped - hien thi truc tiep o Tier 1 de UI de render banner.",
    )
    hint_conflict: bool = Field(
        default=False,
        description="Copy tu RoutingResult - True neu user hint khac router top-1 prediction",
    )
    hint_resolution_note: Optional[str] = Field(
        default=None,
        description="Copy tu RoutingResult - giai thich cach hint va router confidence duoc ket hop",
    )


class RagSource(BaseModel):
    """Citation cu the cho 1 chunk RAG da duoc su dung."""

    file: str = Field(..., description="Ten file PDF nguon")
    page: int = Field(..., description="So trang trong PDF nguon")


class CoTResult(BaseModel):
    """
    Ket qua suy luan Chain-of-Thought tu cot_reasoning node.
    Cung dinh dang voi output cua mapper de so sanh duoc.
    """

    severity: str
    severity_level: int
    icd10_hint: str
    risk_category: str
    reasoning: str = Field(
        ...,
        description="Audit trail day du tung buoc suy luan: label -> spatial -> RAG -> bottleneck -> ket luan"
    )


class ReportOutput(BaseModel):
    """Final output tu Orchestrator - LLM-generated 3-tier report."""

    image_id: str
    tier_1_structured: Tier1Structured = Field(
        ...,
        description="Structured data fields - parse tu UnifiedOutput, khong can LLM"
    )
    tier_2_radiological_description: str = Field(
        ...,
        description=(
            "LLM-generated: mo ta radiological bang ngon ngu tu nhien. "
            "Vd: 'A 1.24 cm2 hypoechoic lesion with irregular margins (circularity: 0.42)...'"
        )
    )
    tier_3_diagnostic_suggestion: str = Field(
        ...,
        description=(
            "LLM-generated: goi y chan doan + follow-up. "
            "KHONG phai chan doan cuoi cung - AI assist only."
        )
    )
    disclaimer: str = Field(
        default=(
            "This AI-generated report is for screening assistance only and does not constitute "
            "a medical diagnosis. All findings must be reviewed and confirmed by a qualified "
            "radiologist or physician."
        )
    )
    rag_sources: List[RagSource] = Field(
        default_factory=list,
        description="Danh sach citation cu the: file PDF + so trang duoc dung trong RAG"
    )
    rag_disabled_warning: Optional[str] = Field(
        default=None,
        description=(
            "Set khi FAISS index chua build / chua co PDF nao duoc index. "
            "Neu khong None, report duoc sinh ra KHONG co clinical guideline retrieval - "
            "chi dua tren classification label + hardcode mapping. UI phai hien thi banner canh bao."
        ),
    )
    mapper_result: Optional[dict] = Field(
        default=None,
        description="Ket qua tu rule-based mapper (KnowledgeMapped dict)"
    )
    cot_result: Optional[CoTResult] = Field(
        default=None,
        description="Ket qua tu CoT reasoning engine (LLM-based)"
    )
    consensus: Optional[bool] = Field(
        default=None,
        description=(
            "True neu mapper va CoT dong thuan (severity_level chenh lech <= 1). "
            "False neu bat dong nhieu hon 1 muc - UI hien thi banner yeu cau radiologist xac nhan. "
            "None khi CoT chua chay (vi du: loi hoac chua duoc bat)."
        )
    )
