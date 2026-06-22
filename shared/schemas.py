"""
shared/schemas.py - Unified Output Schema
==========================================
Most important file of the project. Every service imports from here.
Do not change the schema without notice - changes here affect the entire pipeline.
"""

from pydantic import BaseModel, Field
from typing import Optional, List


# Layer 1: Router Output

class RoutingResult(BaseModel):
    """Output of the Router Service (Layer 1)."""

    modality: str = Field(..., description="'ultrasound' | 'xray' | 'ood'")
    organ: str = Field(..., description="'breast' | 'thyroid' | 'heart' | 'chest' | 'unknown'")
    confidence: float = Field(..., description="Confidence of the routing decision [0, 1]")
    all_scores: dict = Field(
        ...,
        description="Softmax scores for all classes, e.g.: {'us_breast': 0.91, 'us_thyroid': 0.07}"
    )
    is_ood: bool = Field(
        default=False,
        description="True if confidence < threshold -> reject before entering vision"
    )
    module_key: str = Field(
        ...,
        description="Key in module_registry.yaml, e.g.: 'us_breast_v1'"
    )
    router_degraded: bool = Field(
        default=False,
        description=(
            "True if the router is running with random weights (checkpoint does not exist). "
            "When True, the routing decision is not statistically meaningful - the orchestrator "
            "MUST block the pipeline instead of continuing with random routing."
        ),
    )
    user_hint_modality: Optional[str] = Field(
        default=None,
        description="Modality manually chosen by the user, None if 'Auto-detect' is chosen",
    )
    user_hint_organ: Optional[str] = Field(
        default=None,
        description="Organ manually chosen by the user, None if 'Auto-detect' is chosen",
    )
    hint_conflict: bool = Field(
        default=False,
        description="True if the user's hint differs from the router's original top-1 prediction",
    )
    hint_resolution_note: Optional[str] = Field(
        default=None,
        description="Explains how router_confidence and the user hint were combined to reach the final decision",
    )
    final_decision_source: str = Field(
        default="router",
        description="'router' | 'user_hint' | 'weighted' - source of the final decision for module_key",
    )


# Layer 2: Vision Module Output

class ModelOutput(BaseModel):
    """Raw output from UNet_MTL inference (Layer 2)."""

    top_label: str = Field(
        ...,
        description="'benign' | 'malignant' | 'normal'"
    )
    confidence: float = Field(..., description="Confidence of top_label [0, 1]")
    all_scores: dict = Field(
        ...,
        description="{'benign': 0.1, 'malignant': 0.87, 'normal': 0.03}"
    )
    mask_png_base64: str = Field(
        ...,
        description=(
            "Binary mask PNG, base64-encoded. Passed directly over the HTTP body - "
            "does NOT use a path on disk since vision/knowledge are 2 separate containers "
            "that do not share a filesystem. This is the only supported stateless mechanism."
        ),
    )
    bottleneck_features: dict = Field(
        ...,
        description=(
            "Summary statistics from the encoder bottleneck (7x7x448). "
            "Keys: activation_energy, top_channel_activations, attention_hotspot_grid"
        )
    )
    original_size: list = Field(
        default=[512, 512],
        description="[H, W] of the original image - used by the knowledge service to compute spatial features"
    )


# Layer 3: Knowledge Mapper Output

class KnowledgeMapped(BaseModel):
    """Clinical knowledge enrichment (Layer 3)."""

    description: str = Field(
        ...,
        description="Descriptive text from label + RAG context"
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
            "Not a JSON lookup - derived from label + confidence"
        )
    )
    confidence_calibration_note: Optional[str] = Field(
        default=None,
        description=(
            "Set when confidence exceeds a threshold suggestive of overfitting (e.g. >= 0.999) on a "
            "small, uncalibrated dataset. This warning is NOT a bug - it just keeps the LLM/clinician "
            "from reading the percentage as an absolute confidence."
        ),
    )


class SpatialDerived(BaseModel):
    """Spatial features from the segmentation mask (Layer 3, cv2.boundingRect)."""

    bbox: list = Field(..., description="[x1, y1, x2, y2] pixel coordinates")
    area_cm2: float = Field(..., description="Lesion area in cm2 (computed using pixel_spacing)")
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
        description="'low' | 'medium' | 'high' - based on mask quality"
    )


# Unified schema across the whole pipeline

class UnifiedOutput(BaseModel):
    """
    Standardized output schema across every layer.
    QA Agent only reads this schema - does not know where the image came from.
    Adding a new modality = adding a new pipeline, not editing the schema.
    """

    modality: str = Field(..., description="'ultrasound' | 'xray'")
    organ: str = Field(..., description="'breast' | 'thyroid' | 'heart' | 'chest'")
    image_id: str = Field(..., description="Unique ID for the request (UUID or filename)")

    model_output: ModelOutput
    knowledge_mapped: KnowledgeMapped
    spatial_derived: SpatialDerived

    filtered_findings: list = Field(
        default_factory=list,
        description=(
            "Findings dropped due to low confidence - still kept so the LLM is aware of them. "
            "E.g.: [{'label': 'normal', 'confidence': 0.03, 'reason': 'below threshold 0.1'}]"
        )
    )
    coverage_note: str = Field(
        default="Model trained on BUSI dataset (benign/malignant/normal only).",
        description="Warning about the training data scope so the LLM does not over-generalize"
    )


# Layer 4: Final Report Output

class Tier1Structured(BaseModel):
    """Tier 1: Structured fields - easy to parse, display in the UI."""

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
        description="[x1, y1, x2, y2] pixel coordinates - used to draw the overlay in the UI",
    )
    area_cm2: float
    aspect_ratio: float
    circularity: float
    confidence_calibration_note: Optional[str] = Field(
        default=None,
        description="Copied from KnowledgeMapped - shown directly in Tier 1 so the UI can render the banner.",
    )
    hint_conflict: bool = Field(
        default=False,
        description="Copied from RoutingResult - True if the user hint differs from the router's top-1 prediction",
    )
    hint_resolution_note: Optional[str] = Field(
        default=None,
        description="Copied from RoutingResult - explains how the hint and router confidence were combined",
    )
    icd10_agreement: Optional[bool] = Field(
        default=None,
        description=(
            "True if the mapper's and CoT's icd10_hint match, False if they differ, "
            "None if CoT has not run. Independent of consensus (severity) - when False, "
            "the icd10_hint above is a combined string of both codes, e.g. 'N63.0 / C50.9'."
        ),
    )


class RagSource(BaseModel):
    """Specific citation for one used RAG chunk."""

    file: str = Field(..., description="Source PDF file name")
    page: int = Field(..., description="Page number in the source PDF")


class CoTResult(BaseModel):
    """
    Chain-of-Thought reasoning result from the cot_reasoning node.
    Uses the same format as the mapper's output so they can be compared.
    """

    severity: str
    severity_level: int
    icd10_hint: str
    risk_category: str
    reasoning: str = Field(
        ...,
        description="Full audit trail of each reasoning step: label -> spatial -> RAG -> bottleneck -> conclusion"
    )


class ReportOutput(BaseModel):
    """Final output from the Orchestrator - LLM-generated 3-tier report."""

    image_id: str
    tier_1_structured: Tier1Structured = Field(
        ...,
        description="Structured data fields - parsed from UnifiedOutput, no LLM needed"
    )
    tier_2_radiological_description: str = Field(
        ...,
        description=(
            "LLM-generated: radiological description in natural language. "
            "E.g.: 'A 1.24 cm2 hypoechoic lesion with irregular margins (circularity: 0.42)...'"
        )
    )
    tier_3_diagnostic_suggestion: str = Field(
        ...,
        description=(
            "LLM-generated: diagnostic suggestion + follow-up. "
            "NOT a final diagnosis - AI assist only."
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
        description="List of specific citations: PDF file + page number used in RAG"
    )
    rag_disabled_warning: Optional[str] = Field(
        default=None,
        description=(
            "Set when the FAISS index has not been built / no PDF has been indexed yet. "
            "If not None, the report was generated WITHOUT clinical guideline retrieval - "
            "based only on the classification label + hardcoded mapping. The UI must show a warning banner."
        ),
    )
    mapper_result: Optional[dict] = Field(
        default=None,
        description="Result from the rule-based mapper (KnowledgeMapped dict)"
    )
    cot_result: Optional[CoTResult] = Field(
        default=None,
        description="Result from the CoT reasoning engine (LLM-based)"
    )
    consensus: Optional[bool] = Field(
        default=None,
        description=(
            "True if the mapper and CoT agree (severity_level differs by <= 1). "
            "False if they disagree by more than 1 level - the UI shows a banner requiring radiologist confirmation. "
            "None if CoT has not run (e.g.: error or not enabled)."
        )
    )
    icd10_agreement: Optional[bool] = Field(
        default=None,
        description=(
            "True if the mapper's and CoT's icd10_hint match, False if they differ. "
            "This is a clinical question independent of consensus (severity) - the two ICD-10 "
            "codes can differ even when severity_level agrees. None if CoT has not run."
        )
    )
