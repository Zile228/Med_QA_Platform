"""
services/orchestrator/graph.py
================================
LangGraph pipeline - orchestrates the entire flow.

Graph topology:
    route -> vision -> spatial -> knowledge                      --|
                               -> rag_retrieve -> cot_reasoning  --|-> merge -> consistency_guard -> qa_agent

Public API:
    build_graph(services_cfg, llm_client, rag_store, registry) -> CompiledGraph
    run_pipeline(graph, image_bytes, image_id, modality_hint, organ_hint) -> ReportOutput dict
    run_pipeline_async(...) -> ReportOutput dict
"""

import asyncio
import os
import json
import uuid
import httpx
from typing import TypedDict, Optional, Annotated

try:
    from langgraph.graph import StateGraph, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    print("[orchestrator] langgraph not installed, using sequential fallback")

from shared.schemas import (
    RoutingResult, ModelOutput, KnowledgeMapped,
    SpatialDerived, UnifiedOutput, ReportOutput, Tier1Structured,
    RagSource, CoTResult,
)
from shared.telemetry import get_tracer
from services.orchestrator.visual_interpreter import interpret_visual_features


def _keep_last(a, b):
    """Reducer: keep the newest non-None value."""
    return b if b is not None else a


def _merge_list(a, b):
    """Reducer: keep the newest list if non-empty."""
    return b if b else a


class OrchestratorState(TypedDict):
    image_bytes:    bytes
    image_id:       str
    modality_hint:  Optional[str]
    organ_hint:     Optional[str]
    pixel_spacing_mm: Optional[float]
    laterality:     Optional[str]

    routing:        Annotated[Optional[dict], _keep_last]
    model_output:   Annotated[Optional[dict], _keep_last]

    knowledge:      Annotated[Optional[dict], _keep_last]
    spatial:        Annotated[Optional[dict], _keep_last]

    cot_result:     Annotated[Optional[dict], _keep_last]

    rag_chunks:     Annotated[list, _merge_list]
    rag_meta:       Annotated[list, _merge_list]

    consensus:      Annotated[Optional[bool], _keep_last]
    icd10_agreement: Annotated[Optional[bool], _keep_last]
    label_agreement: Annotated[Optional[bool], _keep_last]
    hard_conflict:  Annotated[Optional[bool], _keep_last]

    visual_flags:   Annotated[list, _merge_list]
    risk_modifier:  Annotated[Optional[int], _keep_last]

    report:         Annotated[Optional[dict], _keep_last]
    error:          Annotated[Optional[str], _keep_last]


def _post_json(url: str, json_body: dict, timeout: int = 60) -> dict:
    with httpx.Client(timeout=timeout) as client:
        resp = client.post(url, json=json_body)
        resp.raise_for_status()
        return resp.json()


def _post_multipart(
    url: str,
    image_bytes: bytes,
    fields: dict = None,
    timeout: int = 120,
) -> dict:
    files = {"image": ("image.png", image_bytes, "image/png")}
    data = fields or {}
    with httpx.Client(timeout=timeout) as client:
        resp = client.post(url, files=files, data=data)
        resp.raise_for_status()
        return resp.json()


async def _post_multipart_async(
    url: str,
    image_bytes: bytes,
    fields: dict = None,
    timeout: int = 120,
) -> dict:
    files = {"image": ("image.png", image_bytes, "image/png")}
    data = fields or {}
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(url, files=files, data=data)
        resp.raise_for_status()
        return resp.json()


async def _post_json_async(
    url: str,
    json_body: dict,
    timeout: int = 60,
) -> dict:
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(url, json=json_body)
        resp.raise_for_status()
        return resp.json()


SYSTEM_PROMPT = """You are an AI radiology assistant helping generate structured clinical reports.
You receive structured image analysis data and must produce clear, professional radiological descriptions.
Always include the disclaimer that findings must be confirmed by a qualified radiologist.
Never make a definitive diagnosis -- use language like 'findings are suggestive of' or 'cannot exclude'.
Be concise, factual, and use standard radiological terminology."""

CHAT_SYSTEM_PROMPT = """You are an AI radiology assistant answering follow-up questions about a previously
analyzed medical image. You have access to the full analysis report including classification, spatial
features, and retrieved clinical guidelines. Answer concisely and accurately based on the provided context.
Never make a definitive diagnosis -- always recommend confirmation by a qualified radiologist."""

COT_SYSTEM_PROMPT = """You are a clinical reasoning engine for medical imaging analysis.
Reason step by step, showing your work explicitly.
Your output MUST be valid JSON matching the schema provided. Output ONLY the JSON object, no preamble."""


def _build_report_prompt(
    unified: dict,
    rag_chunks: list,
    cot_result: Optional[dict] = None,
    consensus: Optional[bool] = None,
    icd10_agreement: Optional[bool] = None,
    visual_flags: list = None,
    risk_modifier: int = 0,
    hard_conflict: Optional[bool] = None,
) -> str:
    """Build the LLM prompt from UnifiedOutput + RAG context + CoT."""
    rag_context = (
        "\n\n".join(rag_chunks) if rag_chunks
        else "No additional clinical guidelines retrieved."
    )
    km = unified.get("knowledge_mapped", {})
    sd = unified.get("spatial_derived", {})
    mo = unified.get("model_output", {})

    area_val = sd.get("area_cm2")
    area_str = f"{area_val:.3f} cm2" if area_val is not None else "unavailable (no DICOM metadata)"

    hard_conflict_block = ""
    if hard_conflict is True:
        cnn_label = mo.get("top_label", "unknown")
        cot_label = (cot_result or {}).get("cot_label", "unknown")
        cot_sev = (cot_result or {}).get("severity", "unknown")
        mapper_sev = km.get("severity", "unknown")
        hard_conflict_block = f"""
### HARD CONFLICT -- Radiologist Review Mandatory
The CNN model classified this as: {cnn_label}
Independent CoT reasoning classified this as: {cot_label}
Rule engine severity: {mapper_sev} | CoT severity: {cot_sev}

IMPORTANT: Do NOT pick a side. Present BOTH interpretations in Tier 3.
State explicitly: "Mandatory radiologist review required -- AI assessments are in conflict."
"""

    consensus_block = ""
    if cot_result is not None and hard_conflict is not True:
        if consensus is False:
            cot_sev = cot_result.get("severity", "unknown")
            cot_icd = cot_result.get("icd10_hint", "unknown")
            cot_risk = cot_result.get("risk_category", "unknown")
            cot_reason = cot_result.get("reasoning", "")
            mapper_sev = km.get("severity", "unknown")
            mapper_icd = km.get("icd10_hint", "unknown")
            consensus_block = f"""
### Disagreement between Rule Engine and AI Reasoning (consensus: false)
Rule engine (mapper): severity={mapper_sev}, icd10={mapper_icd}
CoT reasoning:        severity={cot_sev},   icd10={cot_icd}, risk={cot_risk}
CoT audit trail: {cot_reason}

IMPORTANT: In Tier 3, CLEARLY present BOTH perspectives and state:
"Radiologist confirmation required due to disagreement between rule-based and AI reasoning."
"""
        else:
            cot_reason = cot_result.get("reasoning", "")
            consensus_block = f"""
### AI Reasoning (agrees with the rule engine)
CoT audit trail: {cot_reason}
"""

    icd10_block = ""
    if icd10_agreement is False and consensus is not False:
        mapper_icd = km.get("icd10_hint", "unknown")
        cot_icd = (cot_result or {}).get("icd10_hint", "unknown")
        icd10_block = f"""
### ICD-10 Code Disagreement (icd10_agreement: false)
Rule engine (mapper) suggests icd10={mapper_icd}, CoT reasoning suggests icd10={cot_icd}.
Severity agrees but the ICD-10 codes differ -- this is an independent clinical
disagreement, you MUST state BOTH codes in Tier 2 and Tier 3.
"""

    # Visual flags now contain only spatial-derived flags (aspect ratio,
    # circularity). Uncertainty and Grad-CAM overlap flags are disabled
    # pending more grounded proxy metrics.
    flags_text = "\n".join(f"  - {f}" for f in (visual_flags or []))
    visual_flags_block = ""
    if visual_flags:
        visual_flags_block = f"""
### Visual Feature Flags (spatial features only)
{flags_text}
Cumulative risk modifier: {risk_modifier:+d}
"""

    prompt = f"""
## Clinical Image Analysis -- Structured Report Generation

### Image Analysis Results
- Modality: {unified.get('modality', 'unknown')} | Organ: {unified.get('organ', 'unknown')}
- Classification: {mo.get('top_label', 'unknown')} (confidence: {mo.get('confidence', 0):.0%})
- All scores: {json.dumps(mo.get('all_scores', {}), indent=2)}
- Severity: {km.get('severity', 'unknown')} (level {km.get('severity_level', 0)}/4)
- Risk category: {km.get('risk_category', 'unknown')}
- ICD-10 hint: {km.get('icd10_hint', 'unknown')}

### Spatial Features (from segmentation mask)
- Location: {sd.get('location_quadrant', 'unknown')}
- Area: {area_str}
- Aspect ratio: {sd.get('aspect_ratio', 0):.3f} -- {sd.get('aspect_ratio_interpretation', '')}
- Circularity: {sd.get('circularity', 0)} (<0.5 = irregular margin / suspicious)
- Bounding box: {sd.get('bbox', [])}

### Clinical Coverage Note
{unified.get('coverage_note', '')}
{hard_conflict_block}{consensus_block}{icd10_block}{visual_flags_block}
### Retrieved Clinical Guidelines
{rag_context}

---

Please generate:

**TIER 2 -- Radiological Description** (2-3 sentences, professional radiological language):
Describe the findings using the spatial and classification data above.
{"Both ICD-10 codes above must be mentioned explicitly since they disagree." if icd10_block else f"Reference the ICD-10 code {km.get('icd10_hint', '')} and severity {km.get('severity', '')} explicitly."}

**TIER 3 -- Diagnostic Suggestion** (2-3 sentences, include follow-up recommendation):
Based on the findings and clinical guidelines, suggest next steps. Do NOT make a definitive diagnosis.
Start with: "AI-assisted suggestion (must be confirmed by radiologist):"
{"Include both rule-based and AI reasoning perspectives, ending with the disagreement note above." if consensus is False else ""}
{"Mention both ICD-10 codes and note the discrepancy requires radiologist review." if icd10_block else ""}
"""
    return prompt.strip()


def _build_cot_prompt(
    spatial: dict,
    visual_features: dict,
    rag_chunks: list,
    organ: str,
) -> str:
    """
    Build the Chain-of-Thought prompt without CNN label.
    CoT reasons ONLY from spatial features and clinical guidelines.
    Uncertainty and Grad-CAM overlap flags are disabled -- Step 2 reflects
    spatial-only flags when those metrics are inactive.
    """
    rag_text = "\n\n".join(rag_chunks) if rag_chunks else "No clinical guidelines available."
    flags_text = "\n".join(f"  - {f}" for f in visual_features.get("clinical_flags", []))
    if not flags_text:
        flags_text = "  (no significant flags)"

    area_val = spatial.get("area_cm2")
    area_str = f"{area_val:.3f} cm2" if area_val is not None else "unavailable (no DICOM metadata)"

    # TN3K has only benign/malignant labels; "normal" is not a valid class for thyroid.
    if organ == "thyroid":
        cot_label_options = '"benign" | "malignant"'
    else:
        cot_label_options = '"benign" | "malignant" | "normal"'

    return f"""You are analyzing a {organ} ultrasound image.
You have NOT been told the CNN model's classification label.
Reason ONLY from the imaging features and clinical guidelines below.

## Step 1: Spatial Features (from segmentation mask)
- Location: {spatial.get('location_quadrant', 'unknown')}
- Area: {area_str}
- Aspect ratio: {spatial.get('aspect_ratio', 0):.3f} -- {spatial.get('aspect_ratio_interpretation', '')}
- Circularity: {spatial.get('circularity', 0):.3f} (< 0.5 = irregular margin, suspicious)

## Step 2: Spatial Flags (from segmentation mask)
Clinical flags derived from spatial features:
{flags_text}

Risk modifier from flags: {visual_features.get('risk_modifier', 0):+d}
(positive = additional suspicion; negative = reassuring)

## Step 3: Clinical Guidelines (RAG)
{rag_text}

## Step 4: Conclude
Based ONLY on the above (do not assume any CNN label), determine:
- cot_label: your independent classification -- {cot_label_options}
- severity: "incidental" | "significant" | "urgent" | "critical"
- severity_level: integer 1-4 matching severity
- icd10_hint: appropriate ICD-10 code for {organ} + your cot_label
- risk_category: clinical risk description
- reasoning: full audit trail covering each step above

Output ONLY valid JSON, no markdown, no text outside the JSON object:
{{
  "cot_label": "...",
  "severity": "...",
  "severity_level": 0,
  "icd10_hint": "...",
  "risk_category": "...",
  "reasoning": "Step 1: ... Step 2: ... Step 3: ... Step 4: ..."
}}"""


def _build_chat_prompt(
    unified_context: dict,
    rag_chunks: list,
    history: list,
    message: str,
) -> str:
    """Build the multi-turn chatbot prompt from existing context + history."""
    rag_context = (
        "\n\n".join(rag_chunks) if rag_chunks
        else "No additional clinical guidelines retrieved."
    )

    report = unified_context.get("report", {})
    t1 = report.get("tier_1_structured", {})

    area_val = t1.get("area_cm2")
    area_str = f"{area_val:.3f} cm2" if area_val is not None else "unavailable"

    context_block = f"""
## Context: Previously Analyzed Image

Image ID: {unified_context.get('image_id', 'unknown')}
Modality/Organ: {t1.get('modality', '?')} / {t1.get('organ', '?')}
Classification: {t1.get('label', '?')} ({t1.get('confidence', 0):.0%} confidence)
Risk category: {t1.get('risk_category', '?')}
Severity: {t1.get('severity', '?')} (level {t1.get('severity_level', 0)}/4)
Location: {t1.get('location_quadrant', '?')}
Area: {area_str}
Aspect ratio: {t1.get('aspect_ratio', 1.0):.3f}
Circularity: {t1.get('circularity', 1.0):.3f}
ICD-10: {t1.get('icd10_hint', '?')}

Tier 2 (Radiological description):
{report.get('tier_2_radiological_description', '')}

Tier 3 (Diagnostic suggestion):
{report.get('tier_3_diagnostic_suggestion', '')}

Retrieved clinical guidelines:
{rag_context}
""".strip()

    history_block = ""
    for turn in history:
        role = turn.get("role", "user")
        content = turn.get("content", "")
        history_block += f"\n{role.upper()}: {content}"

    return f"{context_block}\n\n## Conversation History{history_block}\n\nUSER: {message}"


ALLOW_DEGRADED_ROUTER = os.getenv("ALLOW_DEGRADED_ROUTER", "false").lower() == "true"


def make_route_node(router_url: str):
    async def route_node(state: OrchestratorState) -> dict:
        with get_tracer().start_as_current_span("graph.route") as span:
            span.set_attribute("image_id", state.get("image_id", ""))
            try:
                fields = {}
                if state.get("modality_hint"):
                    fields["modality_hint"] = state["modality_hint"]
                if state.get("organ_hint"):
                    fields["organ_hint"] = state["organ_hint"]

                result = await _post_multipart_async(
                    f"{router_url}/route",
                    state["image_bytes"],
                    fields=fields or None,
                )

                if result.get("router_degraded") and not ALLOW_DEGRADED_ROUTER:
                    span.set_attribute("route.degraded", True)
                    return {
                        "routing": result,
                        "error": (
                            "Router is running with random weights (no trained checkpoint loaded). "
                            "The routing decision is meaningless and could route the wrong modality "
                            "without warning. Pipeline blocked to avoid generating an incorrect report. "
                            "Place a checkpoint at models/checkpoints/router_effnet_b0.pt, or set "
                            "ALLOW_DEGRADED_ROUTER=true in .env if this is a dev/demo environment."
                        ),
                    }

                if result.get("is_ood"):
                    span.set_attribute("route.is_ood", True)
                    return {
                        "routing": result,
                        "error": (
                            f"Image rejected as out-of-distribution "
                            f"(confidence {result.get('confidence', 0):.0%} < threshold). "
                            "Please upload a breast or thyroid ultrasound image."
                        ),
                    }

                span.set_attribute("route.module_key", result.get("module_key", ""))
                span.set_attribute("route.organ", result.get("organ", ""))
                span.set_attribute("route.confidence", result.get("confidence", 0.0))
                return {"routing": result}

            except Exception as e:
                span.record_exception(e)
                return {"error": f"[route_node] {e}"}
    return route_node


def make_vision_node(vision_url: str, registry=None):
    async def vision_node(state: OrchestratorState) -> dict:
        if state.get("error"):
            return {"model_output": state.get("model_output")}
        routing = state.get("routing") or {}
        organ = routing.get("organ", "breast")
        module_key = routing.get("module_key", "us_breast")

        with get_tracer().start_as_current_span("graph.vision") as span:
            span.set_attribute("image_id", state.get("image_id", ""))
            span.set_attribute("vision.organ", organ)
            try:
                if registry is not None:
                    endpoint = registry.vision_endpoint_for(module_key)
                else:
                    endpoint_map = {
                        "us_breast":  "/analyze/us_breast",
                        "us_thyroid": "/analyze/us_thyroid",
                        "xray":       "/analyze/xray",
                    }
                    endpoint = endpoint_map.get(module_key, "/analyze/us_breast")
            except Exception as e:
                return {"error": f"[vision_node] {e}"}

            try:
                result = await _post_multipart_async(
                    f"{vision_url}{endpoint}",
                    state["image_bytes"],
                    fields={"organ": organ},
                )
                span.set_attribute("vision.top_label", result.get("top_label", ""))
                span.set_attribute("vision.confidence", result.get("confidence", 0.0))
                return {"model_output": result}
            except Exception as e:
                span.record_exception(e)
                return {"error": f"[vision_node] {e}"}
    return vision_node


def make_spatial_node(knowledge_url: str, registry=None):
    """
    Computes spatial features from the segmentation mask.

    registry: ModuleRegistry instance. When provided, uses
    registry.knowledge_spatial_url; otherwise falls back to a constructed URL.
    """
    async def spatial_node(state: OrchestratorState) -> dict:
        if state.get("error"):
            return {"spatial": state.get("spatial")}

        routing = state.get("routing") or {}
        mo = state.get("model_output") or {}

        payload = {
            "organ": routing.get("organ", "breast"),
            "mask_png_base64": mo.get("mask_png_base64", ""),
            "original_size": list(mo.get("original_size", [512, 512])),
            "pixel_spacing_mm": state.get("pixel_spacing_mm"),
            "laterality": state.get("laterality"),
        }

        url = (
            registry.knowledge_spatial_url
            if registry is not None
            else f"{knowledge_url}/map/spatial"
        )

        with get_tracer().start_as_current_span("graph.spatial") as span:
            span.set_attribute("image_id", state.get("image_id", ""))
            try:
                result = await _post_json_async(url, payload)
                spatial = result.get("spatial_derived", {})
                area_val = spatial.get("area_cm2")
                span.set_attribute("spatial.area_cm2", area_val or 0.0)
                return {"spatial": spatial}
            except Exception as e:
                span.record_exception(e)
                return {"error": f"[spatial_node] {e}"}
    return spatial_node


def make_knowledge_node(knowledge_url: str, registry=None):
    """
    registry: ModuleRegistry instance. When provided, uses
    registry.knowledge_knowledge_url; otherwise falls back to a constructed URL.
    """
    async def knowledge_node(state: OrchestratorState) -> dict:
        if state.get("error"):
            return {"knowledge": state.get("knowledge"), "spatial": state.get("spatial")}

        routing = state.get("routing") or {}
        mo = state.get("model_output") or {}

        payload = {
            "modality": routing.get("modality", "ultrasound"),
            "organ": routing.get("organ", "breast"),
            "top_label": mo.get("top_label", "benign"),
            "confidence": mo.get("confidence", 0.0),
            "all_scores": mo.get("all_scores", {}),
        }

        url = (
            registry.knowledge_knowledge_url
            if registry is not None
            else f"{knowledge_url}/map/knowledge"
        )

        with get_tracer().start_as_current_span("graph.knowledge") as span:
            span.set_attribute("image_id", state.get("image_id", ""))
            span.set_attribute("knowledge.organ", routing.get("organ", ""))
            span.set_attribute("knowledge.label", mo.get("top_label", ""))
            try:
                result = await _post_json_async(url, payload)
                km = result.get("knowledge_mapped", {})
                span.set_attribute("knowledge.severity", km.get("severity", ""))
                span.set_attribute("knowledge.severity_level", km.get("severity_level", 0))
                return {"knowledge": km}
            except Exception as e:
                span.record_exception(e)
                return {"error": f"[knowledge_node] {e}"}
    return knowledge_node


def make_cot_node(llm_client):
    """
    CoT reasoning node - runs in parallel with the knowledge node.
    Does NOT receive the CNN classification label -- reasons independently
    from spatial features only (uncertainty and Grad-CAM overlap flags are
    disabled pending more grounded proxy metrics).
    """
    async def cot_reasoning_node(state: OrchestratorState) -> dict:
        if state.get("error"):
            return {"cot_result": state.get("cot_result")}

        mo = state.get("model_output") or {}
        sd = state.get("spatial") or {}
        routing = state.get("routing") or {}
        rag_chunks = state.get("rag_chunks") or []

        if not mo:
            return {"cot_result": None}

        visual_features = interpret_visual_features(
            bottleneck=mo.get("bottleneck_enriched", {}),
            texture=mo.get("texture_features", {}),
            uncertainty=mo.get("uncertainty", {}),
            gradcam_overlap=mo.get("gradcam_mask_overlap", {}),
            spatial=sd,
            organ=routing.get("organ", "breast"),
        )

        prompt = _build_cot_prompt(
            spatial=sd,
            visual_features=visual_features,
            rag_chunks=rag_chunks,
            organ=routing.get("organ", "breast"),
        )

        with get_tracer().start_as_current_span("graph.cot_reasoning") as span:
            span.set_attribute("image_id", state.get("image_id", ""))
            span.set_attribute("cot.organ", routing.get("organ", ""))
            raw = None
            try:
                raw = await asyncio.to_thread(
                    llm_client.generate, prompt, COT_SYSTEM_PROMPT
                )
                clean = raw.strip().replace("```json", "").replace("```", "").strip()
                parsed = json.loads(clean)
                cot = {
                    "cot_label":      str(parsed.get("cot_label", "unknown")),
                    "severity":       str(parsed.get("severity", "incidental")),
                    "severity_level": int(parsed.get("severity_level", 1)),
                    "icd10_hint":     str(parsed.get("icd10_hint", "R93.8")),
                    "risk_category":  str(parsed.get("risk_category", "undetermined")),
                    "reasoning":      str(parsed.get("reasoning", "")),
                }
                span.set_attribute("cot.severity_level", cot["severity_level"])
                span.set_attribute("cot.cot_label", cot["cot_label"])
            except Exception as e:
                span.record_exception(e)
                raw_preview = raw[:200] if raw else ""
                cot = {
                    "cot_label":      "unknown",
                    "severity":       "undetermined",
                    "severity_level": 0,
                    "icd10_hint":     "R93.8",
                    "risk_category":  "undetermined",
                    "reasoning":      f"Parse error: {e}. Raw: {raw_preview}",
                }

        return {
            "cot_result":    cot,
            "visual_flags":  visual_features.get("clinical_flags", []),
            "risk_modifier": visual_features.get("risk_modifier", 0),
        }

    return cot_reasoning_node


def make_rag_node(rag_store, rag_mode: str = "two_stage"):
    """
    RAG node. In two_stage mode, runs after spatial with query "{modality} {organ}".
    In single_stage mode, returns empty chunks immediately (retrieve happens in
    consistency_guard instead).
    """
    async def rag_node(state: OrchestratorState) -> OrchestratorState:
        if state.get("error"):
            return state

        if rag_mode == "single_stage":
            state["rag_chunks"] = []
            state["rag_meta"] = []
            return state

        routing = state.get("routing") or {}
        modality = routing.get("modality", "ultrasound")
        organ = routing.get("organ")
        query = f"{modality} {organ}".strip() if organ else modality

        try:
            if rag_store and rag_store.is_ready():
                meta_list = await asyncio.to_thread(
                    rag_store.retrieve_with_meta, query, 100, organ
                )
            else:
                meta_list = []

            state["rag_chunks"] = [m["chunk"] for m in meta_list]
            state["rag_meta"] = meta_list
        except Exception as e:
            print(f"[rag_node] Retrieve error: {e}")
            state["rag_chunks"] = []
            state["rag_meta"] = []
        return state
    return rag_node


def make_merge_node():
    """Waits for all three branches (knowledge, cot, rag) before consistency_guard."""
    async def merge_node(state: OrchestratorState) -> OrchestratorState:
        if state.get("error"):
            return state
        if not state.get("knowledge"):
            state["error"] = "[merge_node] knowledge output is missing -- vision/knowledge branch failed."
            return state
        return state
    return merge_node


def make_second_rag_retrieval(rag_store, rag_mode: str = "two_stage"):
    """
    Second retrieve after the knowledge result is available.
    In two_stage mode: enriched query reranks candidates from first retrieve.
    In single_stage mode: single retrieve + rerank using enriched query directly.
    """
    async def second_retrieve(state: OrchestratorState) -> tuple:
        """
        Returns (all_chunks: list[str], all_meta: list[dict]) after merging and reranking.
        Called from consistency_guard_node, not a standalone LangGraph node.
        """
        if not (rag_store and rag_store.is_ready()):
            return state.get("rag_chunks") or [], state.get("rag_meta") or []

        km = state.get("knowledge") or {}
        routing = state.get("routing") or {}
        organ = routing.get("organ")
        mo = state.get("model_output") or {}
        top_label = mo.get("top_label", "")
        icd10 = km.get("icd10_hint", "")

        enriched_query = f"{top_label} {organ or ''} ultrasound findings {icd10}".strip()

        if rag_mode == "single_stage":
            try:
                meta_list = await asyncio.to_thread(
                    rag_store.retrieve_with_meta, enriched_query, 3, organ
                )
            except Exception as e:
                print(f"[second_rag single_stage] Retrieve error: {e}")
                meta_list = []
            reranked = rag_store.rerank(enriched_query, meta_list, top_n=3)
            chunks = [m["chunk"] for m in reranked]
            return chunks, reranked

        try:
            meta2 = await asyncio.to_thread(
                rag_store.retrieve_with_meta, enriched_query, 5, organ
            )
        except Exception as e:
            print(f"[second_rag] Second retrieve error: {e}")
            meta2 = []

        existing_meta = state.get("rag_meta") or []
        existing_texts = {m["chunk"] for m in existing_meta}
        combined = list(existing_meta)
        for m in meta2:
            if m["chunk"] not in existing_texts:
                combined.append(m)
                existing_texts.add(m["chunk"])

        reranked = rag_store.rerank(enriched_query, combined, top_n=3)
        chunks = [m["chunk"] for m in reranked]
        return chunks, reranked

    return second_retrieve


def make_consistency_guard_node(rag_store, rag_mode: str = "two_stage"):
    """
    1. Performs the second RAG retrieve with enriched context.
    2. Compares mapper and CoT severity_level -> sets consensus flag.
    3. Compares mapper and CoT icd10_hint -> sets icd10_agreement flag.
    4. Compares CNN label vs CoT label -> sets label_agreement and hard_conflict.
    """
    second_retrieve = make_second_rag_retrieval(rag_store, rag_mode)

    async def consistency_guard_node(state: OrchestratorState) -> OrchestratorState:
        if state.get("error"):
            return state

        with get_tracer().start_as_current_span("graph.consistency_guard") as span:
            span.set_attribute("image_id", state.get("image_id", ""))

            final_chunks, final_meta = await second_retrieve(state)
            state["rag_chunks"] = final_chunks
            state["rag_meta"] = final_meta

            mapper = state.get("knowledge") or {}
            mapper_level = mapper.get("severity_level", 0)
            mapper_icd10 = mapper.get("icd10_hint")

            cot = state.get("cot_result") or {}
            cot_level = cot.get("severity_level", 0)
            cot_icd10 = cot.get("icd10_hint")

            cot_undetermined = cot_level == 0 or cot.get("severity") == "undetermined"

            if cot_undetermined:
                state["consensus"] = None
            elif abs(mapper_level - cot_level) <= 1:
                state["consensus"] = True
            else:
                state["consensus"] = False

            if cot_undetermined or mapper_icd10 is None or cot_icd10 is None:
                state["icd10_agreement"] = None
            else:
                state["icd10_agreement"] = mapper_icd10 == cot_icd10

            cnn_label = (state.get("model_output") or {}).get("top_label", "unknown")
            cot_label = cot.get("cot_label", "unknown")

            if cot_undetermined or cot_label == "unknown":
                state["label_agreement"] = None
                state["hard_conflict"] = None
            else:
                state["label_agreement"] = (cnn_label.lower() == cot_label.lower())
                state["hard_conflict"] = (
                    not state["label_agreement"]  # label disagrees...
                    and (
                        abs(mapper_level - cot_level) > 1  # ...AND severity diverges
                        or (cot_label in ("malignant",) and cnn_label in ("benign", "normal"))  # ...OR benign↔malignant flip
                    )
                )

            span.set_attribute("guard.mapper_level", mapper_level)
            span.set_attribute("guard.cot_level", cot_level)
            span.set_attribute("guard.mapper_icd10", mapper_icd10 or "")
            span.set_attribute("guard.cot_icd10", cot_icd10 or "")
            span.set_attribute("guard.consensus", str(state.get("consensus")))
            span.set_attribute("guard.icd10_agreement", str(state.get("icd10_agreement")))
            span.set_attribute("guard.label_agreement", str(state.get("label_agreement")))
            span.set_attribute("guard.hard_conflict", str(state.get("hard_conflict")))

        return state
    return consistency_guard_node


def make_qa_agent_node(llm_client, rag_store):
    """Generates the 3-tier report from vision, knowledge, CoT, and RAG results."""
    async def qa_agent_node(state: OrchestratorState) -> OrchestratorState:
        if state.get("error"):
            return state

        routing = state.get("routing") or {}
        mo = state.get("model_output") or {}
        km = state.get("knowledge") or {}
        sd = state.get("spatial") or {}
        rag_chunks = state.get("rag_chunks") or []
        rag_meta = state.get("rag_meta") or []
        cot_result = state.get("cot_result")
        consensus = state.get("consensus")
        icd10_agreement = state.get("icd10_agreement")
        hard_conflict = state.get("hard_conflict")
        visual_flags = state.get("visual_flags") or []
        risk_modifier = state.get("risk_modifier") or 0

        unified = {
            "modality": routing.get("modality", "ultrasound"),
            "organ": routing.get("organ", "breast"),
            "image_id": state["image_id"],
            "model_output": mo,
            "knowledge_mapped": km,
            "spatial_derived": sd,
            "coverage_note": "Model trained on BUSI dataset (benign/malignant/normal only).",
        }

        prompt = _build_report_prompt(
            unified, rag_chunks, cot_result, consensus, icd10_agreement,
            visual_flags=visual_flags,
            risk_modifier=risk_modifier,
            hard_conflict=hard_conflict,
        )

        with get_tracer().start_as_current_span("graph.qa_agent") as span:
            span.set_attribute("image_id", state.get("image_id", ""))
            span.set_attribute("qa.consensus", str(consensus))
            span.set_attribute("qa.icd10_agreement", str(icd10_agreement))
            span.set_attribute("qa.hard_conflict", str(hard_conflict))
            span.set_attribute("qa.rag_chunks_used", len(rag_chunks))
            try:
                llm_response = await asyncio.to_thread(
                    llm_client.generate, prompt, SYSTEM_PROMPT
                )
            except Exception as e:
                span.record_exception(e)
                llm_response = f"[LLM unavailable: {e}]"

        tier2, tier3 = _parse_tiers(llm_response)

        mapper_icd10 = km.get("icd10_hint", "unknown")
        if icd10_agreement is False:
            cot_icd10 = (cot_result or {}).get("icd10_hint", "unknown")
            tier1_icd10_hint = f"{mapper_icd10} / {cot_icd10}"
        else:
            tier1_icd10_hint = mapper_icd10

        tier1 = {
            "modality": routing.get("modality", "ultrasound"),
            "organ": routing.get("organ", "breast"),
            "label": mo.get("top_label", "unknown"),
            "confidence": mo.get("confidence", 0.0),
            "risk_category": km.get("risk_category", "unknown"),
            "severity": km.get("severity", "unknown"),
            "severity_level": km.get("severity_level", 1),
            "icd10_hint": tier1_icd10_hint,
            "location_quadrant": sd.get("location_quadrant", "unknown"),
            "bbox": sd.get("bbox", [0, 0, 0, 0]),
            "area_cm2": sd.get("area_cm2"),
            "pixel_spacing_reliable": sd.get("pixel_spacing_reliable", False),
            "aspect_ratio": sd.get("aspect_ratio", 1.0),
            "aspect_ratio_interpretation": sd.get("aspect_ratio_interpretation", ""),
            "circularity": sd.get("circularity", 1.0),
            "confidence_calibration_note": km.get("confidence_calibration_note"),
            "hint_conflict": routing.get("hint_conflict", False),
            "hint_resolution_note": routing.get("hint_resolution_note"),
            "icd10_agreement": icd10_agreement,
            "gradcam_png_base64": mo.get("gradcam_png_base64", ""),
            "visual_flags": visual_flags,
            "risk_modifier": risk_modifier,
            "label_agreement": state.get("label_agreement"),
            "hard_conflict": hard_conflict,
        }

        rag_sources = [
            {"file": m.get("source_file", "unknown"), "page": m.get("page_number", 0)}
            for m in rag_meta
        ]

        state["report"] = {
            "image_id": state["image_id"],
            "tier_1_structured": tier1,
            "tier_2_radiological_description": tier2,
            "tier_3_diagnostic_suggestion": tier3,
            "rag_sources": rag_sources,
            "rag_disabled_warning": (
                None if (rag_store and rag_store.is_ready()) else
                "RAG context not available -- report generated from classification "
                "label and hardcoded mapping only, without clinical guideline retrieval."
            ),
            "mapper_result": km,
            "cot_result": cot_result,
            "consensus": consensus,
            "icd10_agreement": icd10_agreement,
            "hard_conflict": hard_conflict,
            "label_agreement": state.get("label_agreement"),
            "_rag_chunks_internal": rag_chunks,
        }
        return state
    return qa_agent_node


def _parse_tiers(llm_text: str) -> tuple:
    """Parse the LLM response -> (tier2_text, tier3_text)."""
    text = llm_text.strip()
    t2_start = -1
    t3_start = -1

    for marker in ["**TIER 2", "TIER 2", "Tier 2"]:
        idx = text.find(marker)
        if idx != -1:
            t2_start = idx
            break

    for marker in ["**TIER 3", "TIER 3", "Tier 3"]:
        idx = text.find(marker)
        if idx != -1:
            t3_start = idx
            break

    if t2_start != -1 and t3_start != -1:
        tier2 = text[t2_start:t3_start].strip()
        tier3 = text[t3_start:].strip()
    elif t3_start != -1:
        tier2 = text[:t3_start].strip()
        tier3 = text[t3_start:].strip()
    else:
        mid = len(text) // 2
        tier2 = text[:mid].strip()
        tier3 = text[mid:].strip()

    for marker in [
        "**TIER 2 -- Radiological Description**",
        "TIER 2 -- Radiological Description",
        "**TIER 3 -- Diagnostic Suggestion**",
        "TIER 3 -- Diagnostic Suggestion",
    ]:
        tier2 = tier2.replace(marker, "").strip()
        tier3 = tier3.replace(marker, "").strip()

    return (
        tier2 or "(No radiological description generated)",
        tier3 or "(No diagnostic suggestion generated)",
    )


def build_graph(services_cfg: dict, llm_client, rag_store, registry=None):
    """
    Build and compile the LangGraph pipeline.

    Args:
        services_cfg: dict with router_url, vision_url, knowledge_url
        llm_client:   BaseLLMClient instance
        rag_store:    FAISSStore instance
        registry:     ModuleRegistry instance
    """
    router_url = services_cfg.get("router_url", "http://router:8001")
    vision_url = services_cfg.get("vision_url", "http://vision:8002")
    knowledge_url = services_cfg.get("knowledge_url", "http://knowledge:8003")

    rag_mode = os.getenv("RAG_MODE", "two_stage")

    route_node = make_route_node(router_url)
    vision_node = make_vision_node(vision_url, registry=registry)
    spatial_node = make_spatial_node(knowledge_url, registry=registry)
    knowledge_node = make_knowledge_node(knowledge_url, registry=registry)
    cot_node = make_cot_node(llm_client)
    rag_node = make_rag_node(rag_store, rag_mode=rag_mode)
    merge_node = make_merge_node()
    consistency_guard_node = make_consistency_guard_node(rag_store, rag_mode=rag_mode)
    qa_agent_node = make_qa_agent_node(llm_client, rag_store)

    if LANGGRAPH_AVAILABLE:
        g = StateGraph(OrchestratorState)
        g.add_node("route", route_node)
        g.add_node("vision", vision_node)
        g.add_node("spatial", spatial_node)
        g.add_node("knowledge", knowledge_node)
        g.add_node("cot_reasoning", cot_node)
        g.add_node("rag_retrieve", rag_node)
        g.add_node("merge", merge_node)
        g.add_node("consistency_guard", consistency_guard_node)
        g.add_node("qa_agent", qa_agent_node)

        g.set_entry_point("route")
        g.add_edge("route", "vision")
        g.add_edge("vision", "spatial")
        g.add_edge("spatial", "knowledge")
        g.add_edge("spatial", "rag_retrieve")
        g.add_edge("rag_retrieve", "cot_reasoning")

        g.add_edge(["knowledge", "cot_reasoning", "rag_retrieve"], "merge")
        g.add_edge("merge", "consistency_guard")
        g.add_edge("consistency_guard", "qa_agent")
        g.add_edge("qa_agent", END)

        return g.compile()
    else:
        return AsyncSequentialFallback(
            image_nodes=[route_node, vision_node, spatial_node, knowledge_node],
            cot_node=cot_node,
            rag_node=rag_node,
            merge_node=merge_node,
            consistency_guard_node=consistency_guard_node,
            qa_agent_node=qa_agent_node,
        )


class AsyncSequentialFallback:
    """
    Fallback when langgraph is not installed.
    Mirrors build_graph edges: route -> vision -> spatial ->
    {knowledge, rag_retrieve} in parallel -> cot_reasoning -> merge ->
    consistency_guard -> qa_agent.

    cot_node runs after rag_retrieve (so its prompt's clinical-guidelines
    step is populated) but on a snapshot with knowledge set to None, so
    CoT never sees the CNN-driven mapper result before reasoning.
    """

    def __init__(
        self,
        image_nodes,
        cot_node,
        rag_node,
        merge_node,
        consistency_guard_node,
        qa_agent_node,
    ):
        self.image_nodes = image_nodes
        self.cot_node = cot_node
        self.rag_node = rag_node
        self.merge_node = merge_node
        self.consistency_guard_node = consistency_guard_node
        self.qa_agent_node = qa_agent_node

    async def ainvoke(self, state: dict) -> dict:
        import copy

        route_node = self.image_nodes[0]
        state.update(await route_node(state))
        if state.get("error"):
            return state

        vision_node, spatial_node, knowledge_node = self.image_nodes[1:]

        state.update(await vision_node(state))
        if state.get("error"):
            return state

        state.update(await spatial_node(state))
        if state.get("error"):
            return state

        state_for_knowledge = copy.deepcopy(state)
        state_for_rag = copy.deepcopy(state)
        knowledge_partial, rag_result = await asyncio.gather(
            knowledge_node(state_for_knowledge),
            self.rag_node(state_for_rag),
        )
        state.update(knowledge_partial)
        state["rag_chunks"] = rag_result.get("rag_chunks", [])
        state["rag_meta"] = rag_result.get("rag_meta", [])

        state_for_cot = copy.deepcopy(state)
        state_for_cot["knowledge"] = None
        cot_partial = await self.cot_node(state_for_cot)
        if not state.get("error"):
            state["cot_result"] = cot_partial.get("cot_result")
            state["visual_flags"] = cot_partial.get("visual_flags", [])
            state["risk_modifier"] = cot_partial.get("risk_modifier", 0)

        if state.get("error"):
            return state

        state = await self.merge_node(state)
        if state.get("error"):
            return state

        state = await self.consistency_guard_node(state)
        if state.get("error"):
            return state

        state = await self.qa_agent_node(state)
        return state


async def run_pipeline_async(
    graph,
    image_bytes: bytes,
    image_id: str = None,
    modality_hint: str = None,
    organ_hint: str = None,
    pixel_spacing_mm: float = None,
    laterality: str = None,
) -> dict:
    """
    Run the entire pipeline from bytes -> ReportOutput dict (async).

    Used by the FastAPI async handler.
    """
    if image_id is None:
        image_id = uuid.uuid4().hex[:12]

    initial_state: OrchestratorState = {
        "image_bytes": image_bytes,
        "image_id": image_id,
        "modality_hint": modality_hint,
        "organ_hint": organ_hint,
        "pixel_spacing_mm": pixel_spacing_mm,
        "laterality": laterality,
        "routing": None,
        "model_output": None,
        "knowledge": None,
        "spatial": None,
        "cot_result": None,
        "rag_chunks": [],
        "rag_meta": [],
        "consensus": None,
        "icd10_agreement": None,
        "label_agreement": None,
        "hard_conflict": None,
        "visual_flags": [],
        "risk_modifier": 0,
        "report": None,
        "error": None,
    }

    final_state = await graph.ainvoke(initial_state)

    if final_state.get("error"):
        raise RuntimeError(final_state["error"])

    return final_state["report"]


def run_pipeline(
    graph,
    image_bytes: bytes,
    image_id: str = None,
    modality_hint: str = None,
    organ_hint: str = None,
    pixel_spacing_mm: float = None,
    laterality: str = None,
) -> dict:
    """Sync wrapper of run_pipeline_async -- only for use outside FastAPI."""
    return asyncio.run(run_pipeline_async(
        graph, image_bytes, image_id, modality_hint, organ_hint,
        pixel_spacing_mm, laterality,
    ))