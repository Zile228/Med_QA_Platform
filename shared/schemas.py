from pydantic import BaseModel, Field
from typing import Optional, List


class RoutingResult(BaseModel):
    modality: str = Field(..., description="'ultrasound' | 'xray' | 'ood'")
    organ: str = Field(..., description="'breast' | 'thyroid' | 'heart' | 'chest' | 'unknown'")
    confidence: float = Field(..., description="Confidence of the routing decision [0, 1]")
    all_scores: dict = Field(...)
    is_ood: bool = Field(default=False)
    module_key: str = Field(...)
    router_degraded: bool = Field(default=False)
    user_hint_modality: Optional[str] = Field(default=None)
    user_hint_organ: Optional[str] = Field(default=None)
    hint_conflict: bool = Field(default=False)
    hint_resolution_note: Optional[str] = Field(default=None)
    final_decision_source: str = Field(default="router")


class ModelOutput(BaseModel):
    top_label: str = Field(..., description="'benign' | 'malignant' | 'normal'")
    confidence: float = Field(...)
    all_scores: dict = Field(...)
    mask_png_base64: str = Field(...)
    original_size: list = Field(default=[512, 512])
    bottleneck_enriched: dict = Field(
        default_factory=dict,
        description=(
            "Spatial statistics from the encoder bottleneck (448 ch, 7x7 grid). "
            "Keys: activation_energy, center_periphery_ratio, spatial_entropy, "
            "quadrant_activations {nw,ne,sw,se}, top_channel_activations."
        ),
    )
    gradcam_png_base64: str = Field(
        default="",
        description="Grad-CAM heatmap PNG, base64-encoded. Empty string if unavailable.",
    )
    gradcam_mask_overlap: dict = Field(
        default_factory=dict,
        description="Keys: iou (float), interpretation ('high'|'medium'|'low').",
    )
    texture_features: dict = Field(
        default_factory=dict,
        description="Keys: internal_heterogeneity (float), lesion_background_contrast (float).",
    )
    uncertainty: dict = Field(
        default_factory=dict,
        description=(
            "Keys: mean_confidence (list[float]), uncertainty (list[float]), "
            "predictive_entropy (float)."
        ),
    )
    filtered_findings: list = Field(default_factory=list)


class KnowledgeMapped(BaseModel):
    description: str = Field(...)
    severity: str = Field(...)
    severity_level: int = Field(..., ge=1, le=4)
    icd10_hint: str = Field(...)
    risk_category: str = Field(...)
    confidence_calibration_note: Optional[str] = Field(default=None)


class SpatialDerived(BaseModel):
    bbox: list = Field(..., description="[x1, y1, x2, y2] pixel coordinates")
    area_cm2: Optional[float] = Field(
        default=None,
        description=(
            "Lesion area in cm2. None when pixel_spacing_mm is unknown "
            "(no DICOM metadata). Always None for 'normal' label (no contour)."
        ),
    )
    pixel_spacing_reliable: bool = Field(
        default=False,
        description="True only when pixel_spacing_mm came from DICOM metadata.",
    )
    centroid: list = Field(...)
    location_quadrant: str = Field(...)
    aspect_ratio: float = Field(
        ...,
        description=(
            "width/height ratio. "
            "Breast: <0.8 = taller-than-wide (suspicious per BI-RADS). "
            "Thyroid: <1.0 = taller-than-wide (suspicious per TI-RADS). "
            ">1.8 = markedly wider-than-tall (low suspicion)."
        ),
    )
    aspect_ratio_interpretation: str = Field(
        default="",
        description="Human-readable interpretation of aspect_ratio.",
    )
    circularity: float = Field(
        ...,
        description="4pi*area/perimeter^2. 1.0=perfect circle, <0.5=irregular margin (suspicious)",
    )
    width_px: int
    height_px: int
    location_confidence: str = Field(...)


class UnifiedOutput(BaseModel):
    modality: str = Field(...)
    organ: str = Field(...)
    image_id: str = Field(...)
    model_output: ModelOutput
    knowledge_mapped: KnowledgeMapped
    spatial_derived: SpatialDerived
    filtered_findings: list = Field(default_factory=list)
    coverage_note: str = Field(default="Model trained on BUSI dataset (benign/malignant/normal only).")


class Tier1Structured(BaseModel):
    modality: str
    organ: str
    label: str
    confidence: float
    risk_category: str
    severity: str
    severity_level: int
    icd10_hint: str
    location_quadrant: str
    bbox: list = Field(default_factory=lambda: [0, 0, 0, 0])
    area_cm2: Optional[float] = None
    pixel_spacing_reliable: bool = False
    aspect_ratio: float
    aspect_ratio_interpretation: str = ""
    circularity: float
    confidence_calibration_note: Optional[str] = Field(default=None)
    hint_conflict: bool = Field(default=False)
    hint_resolution_note: Optional[str] = Field(default=None)
    icd10_agreement: Optional[bool] = Field(default=None)
    gradcam_png_base64: str = ""
    visual_flags: list = Field(default_factory=list)
    risk_modifier: int = Field(default=0)
    label_agreement: Optional[bool] = Field(default=None)
    hard_conflict: Optional[bool] = Field(default=None)


class RagSource(BaseModel):
    file: str = Field(...)
    page: int = Field(...)
    # Optional vi UI khong can hien thi noi dung chunk, chi can citation
    # (file, page). eval/eval_ragas.py va eval/eval_qa.py dung field nay de
    # danh gia Faithfulness/Consistency tren noi dung RAG thuc te, vi
    # _rag_chunks_internal trong graph.py khong duoc serialize ra HTTP response.
    text: Optional[str] = Field(default=None)


class CoTResult(BaseModel):
    severity: str
    severity_level: int
    icd10_hint: str
    risk_category: str
    reasoning: str = Field(...)
    cot_label: str = Field(
        default="unknown",
        description=(
            "CoT's independent classification: 'benign'|'malignant'|'normal'|'unknown'. "
            "Compared against CNN top_label in consistency_guard for label_agreement."
        ),
    )


class ReportOutput(BaseModel):
    image_id: str
    tier_1_structured: Tier1Structured = Field(...)
    tier_2_radiological_description: str = Field(...)
    tier_3_diagnostic_suggestion: str = Field(...)
    disclaimer: str = Field(
        default=(
            "This AI-generated report is for screening assistance only and does not constitute "
            "a medical diagnosis. All findings must be reviewed and confirmed by a qualified "
            "radiologist or physician."
        )
    )
    rag_sources: List[RagSource] = Field(default_factory=list)
    rag_disabled_warning: Optional[str] = Field(default=None)
    mapper_result: Optional[dict] = Field(default=None)
    cot_result: Optional[CoTResult] = Field(default=None)
    consensus: Optional[bool] = Field(default=None)
    icd10_agreement: Optional[bool] = Field(default=None)
    hard_conflict: Optional[bool] = Field(
        default=None,
        description=(
            "True when: CNN label disagrees with CoT label (label_agreement=False) "
            "AND at least one of: (a) severity levels differ by more than 1, "
            "OR (b) CoT says malignant but CNN says benign or normal. "
            "When True, the UI must show a mandatory radiologist-review banner. "
            "None if CoT has not run or returned undetermined."
        ),
    )