"""
services/orchestrator/graph.py
================================
LangGraph pipeline - orchestrates the entire flow.

Graph (fan-out/fan-in after adding CoT):
    route -> vision -> knowledge  --|
          -> cot_reasoning        --|-> merge -> consistency_guard -> qa_agent
          -> rag_retrieve         --|

    The three branches after route run in parallel (knowledge + cot_reasoning + rag).
    merge waits for ALL THREE branches before running consistency_guard.
    consistency_guard compares severity_level (consensus) and icd10_hint (icd10_agreement)
    of mapper vs CoT, two separate flags since they are two different clinical questions.

Each node = 1 HTTP call to the corresponding service:
    route_node            -> POST router:8001/route
    vision_node           -> POST vision:8002/analyze/{modality}
    knowledge_node        -> POST knowledge:8003/map
    cot_reasoning_node    -> LLM Chain-of-Thought (runs in parallel with knowledge)
    rag_node              -> FAISS retrieve (local, no HTTP)
    merge_node            -> checks state, short-circuits on error
    consistency_guard_node -> compares mapper vs CoT, sets consensus flag
    qa_agent_node         -> LLM generates the 3-tier report

Public API:
    build_graph(services_cfg, llm_client, rag_store, registry) -> CompiledGraph
    run_pipeline(graph, image_bytes, question, image_id,
                 modality_hint, organ_hint) -> ReportOutput dict
    run_pipeline_async(...) -> ReportOutput dict  (for FastAPI async handler)
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


# Reducers for LangGraph state

def _keep_last(a, b):
    """Reducer: keep the newest non-None value."""
    return b if b is not None else a


def _merge_list(a, b):
    """Reducer: keep the newest list if non-empty."""
    return b if b else a


class OrchestratorState(TypedDict):
    # Input - unchanged after init
    image_bytes:    bytes
    question:       str
    image_id:       str
    modality_hint:  Optional[str]
    organ_hint:     Optional[str]

    # Layer 1 output
    routing:        Annotated[Optional[dict], _keep_last]

    # Layer 2 output
    model_output:   Annotated[Optional[dict], _keep_last]

    # Layer 3 output from the rule-based mapper
    knowledge:      Annotated[Optional[dict], _keep_last]
    spatial:        Annotated[Optional[dict], _keep_last]

    # Layer 3 output from the CoT engine (runs in parallel with knowledge)
    cot_result:     Annotated[Optional[dict], _keep_last]

    # RAG context from the branch running in parallel with vision
    rag_chunks:     Annotated[list, _merge_list]
    rag_meta:       Annotated[list, _merge_list]

    # Result of comparing mapper vs CoT
    consensus:      Annotated[Optional[bool], _keep_last]
    icd10_agreement: Annotated[Optional[bool], _keep_last]

    # Final output
    report:         Annotated[Optional[dict], _keep_last]

    # Error tracking
    error:          Annotated[Optional[str], _keep_last]


# HTTP call helpers

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
    """Async version of _post_multipart - used by async nodes."""
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


# Building prompts for the LLM

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


def _format_bottleneck(bottleneck: dict) -> str:
    """Format bottleneck_features into descriptive text for the prompt."""
    if not bottleneck:
        return "Bottleneck features: no data available."

    energy = bottleneck.get("activation_energy", "N/A")
    hotspot = bottleneck.get("attention_hotspot_grid", [])
    top_ch = bottleneck.get("top_channel_activations", [])

    # hotspot is a single [row, col] coordinate pair (where the model focuses most),
    # NOT a 2D matrix -- see services/vision/us_breast/model.py: hotspot_pos = [row, col]
    hotspot_str = ""
    if hotspot and len(hotspot) == 2:
        hotspot_str = f"region [{hotspot[0]},{hotspot[1]}] (7x7 grid)"

    return (
        f"Bottleneck activation energy: {energy}. "
        f"Strongest attention focus: {hotspot_str or 'undetermined'}. "
        f"Top channels: {top_ch[:3] if top_ch else 'N/A'}."
    )


def _build_report_prompt(
    unified: dict,
    question: str,
    rag_chunks: list,
    cot_result: Optional[dict] = None,
    consensus: Optional[bool] = None,
    icd10_agreement: Optional[bool] = None,
) -> str:
    """Build the LLM prompt from UnifiedOutput + user question + RAG context + CoT."""
    rag_context = (
        "\n\n".join(rag_chunks) if rag_chunks
        else "No additional clinical guidelines retrieved."
    )
    km = unified.get("knowledge_mapped", {})
    sd = unified.get("spatial_derived", {})
    mo = unified.get("model_output", {})
    bottleneck_text = _format_bottleneck(mo.get("bottleneck_features", {}))

    consensus_block = ""
    if cot_result is not None:
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

    # icd10_agreement is a question separate from consensus -- severity can
    # agree while the ICD-10 codes still differ, requiring an independent
    # warning to the LLM.
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

    prompt = f"""
## Clinical Image Analysis -- Structured Report Generation

### User Question
{question}

### Image Analysis Results
- Modality: {unified.get('modality', 'unknown')} | Organ: {unified.get('organ', 'unknown')}
- Classification: {mo.get('top_label', 'unknown')} (confidence: {mo.get('confidence', 0):.0%})
- All scores: {json.dumps(mo.get('all_scores', {}), indent=2)}
- Severity: {km.get('severity', 'unknown')} (level {km.get('severity_level', 0)}/4)
- Risk category: {km.get('risk_category', 'unknown')}
- ICD-10 hint: {km.get('icd10_hint', 'unknown')}

### Spatial Features (from segmentation mask)
- Location: {sd.get('location_quadrant', 'unknown')}
- Area: {sd.get('area_cm2', 0)} cm2
- Aspect ratio: {sd.get('aspect_ratio', 0)} (>1.5 = elongated / suspicious)
- Circularity: {sd.get('circularity', 0)} (<0.5 = irregular margin / suspicious)
- Bounding box: {sd.get('bbox', [])}

### Model Attention (Bottleneck Features)
{bottleneck_text}
Note: if the attention region diverges significantly from the segmentation bbox, this is a signal the model is uncertain.

### Clinical Coverage Note
{unified.get('coverage_note', '')}
{consensus_block}{icd10_block}
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

Answer the user's question: "{question}"
"""
    return prompt.strip()


def _build_cot_prompt(
    top_label: str,
    confidence: float,
    all_scores: dict,
    spatial: dict,
    bottleneck: dict,
    rag_chunks: list,
    organ: str,
) -> str:
    """
    Build the Chain-of-Thought prompt to reason independently of the mapper.
    CoT MUST NOT know the mapper's result before reasoning.
    Output is a JSON object matching the CoTResult schema.
    """
    rag_text = "\n\n".join(rag_chunks) if rag_chunks else "No clinical guidelines available."
    bottleneck_text = _format_bottleneck(bottleneck)

    return f"""You are analyzing a {organ} ultrasound image.

Reason through the following data step by step, then output your conclusion as JSON.

## Step 1: Classification Result
- Top label: {top_label} (confidence: {confidence:.2%})
- All scores: {json.dumps(all_scores)}

## Step 2: Spatial Features
- Location: {spatial.get('location_quadrant', 'unknown')}
- Area: {spatial.get('area_cm2', 0):.3f} cm2
- Aspect ratio: {spatial.get('aspect_ratio', 0):.3f} (>1.5 suspicious)
- Circularity: {spatial.get('circularity', 0):.3f} (<0.5 irregular margin)

## Step 3: Model Attention
{bottleneck_text}
If attention hotspot differs significantly from the bbox, note model uncertainty.

## Step 4: Clinical Guidelines (RAG)
{rag_text}

## Step 5: Conclude
Based on all steps above, determine:
- severity: one of "incidental" | "significant" | "urgent" | "critical"
- severity_level: integer 1-4 matching severity
- icd10_hint: appropriate ICD-10 code for {organ} + {top_label}
- risk_category: clinical risk description (e.g. "High suspicion (BI-RADS 4C-5)")
- reasoning: full audit trail as a single string covering all 5 steps

Output ONLY valid JSON, no markdown, no explanation outside JSON:
{{
  "severity": "...",
  "severity_level": 0,
  "icd10_hint": "...",
  "risk_category": "...",
  "reasoning": "Step 1: ... Step 2: ... Step 3: ... Step 4: ... Step 5: ..."
}}"""


def _build_chat_prompt(
    unified_context: dict,
    rag_chunks: list,
    history: list,
    message: str,
) -> str:
    """
    Build the multi-turn chatbot prompt from existing context + history.

    unified_context: dict containing the unified output + report from the
                      original analyze call.
    history: list[dict] with keys 'role' and 'content'.
    """
    rag_context = (
        "\n\n".join(rag_chunks) if rag_chunks
        else "No additional clinical guidelines retrieved."
    )

    report = unified_context.get("report", {})
    t1 = report.get("tier_1_structured", {})

    context_block = f"""
## Context: Previously Analyzed Image

Image ID: {unified_context.get('image_id', 'unknown')}
Modality/Organ: {t1.get('modality', '?')} / {t1.get('organ', '?')}
Classification: {t1.get('label', '?')} ({t1.get('confidence', 0):.0%} confidence)
Risk category: {t1.get('risk_category', '?')}
Severity: {t1.get('severity', '?')} (level {t1.get('severity_level', 0)}/4)
Location: {t1.get('location_quadrant', '?')}
Area: {t1.get('area_cm2', 0):.3f} cm2
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


# LangGraph nodes

ALLOW_DEGRADED_ROUTER = os.getenv("ALLOW_DEGRADED_ROUTER", "false").lower() == "true"


def make_route_node(router_url: str):
    """
    Async node - calls /route with a hint if provided.
    hint_conflict/hint_resolution_note/final_decision_source are copied into
    state so report_node can use them in tier_1_structured.
    """
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

                span.set_attribute("route.module_key",  result.get("module_key", ""))
                span.set_attribute("route.organ",       result.get("organ", ""))
                span.set_attribute("route.confidence",  result.get("confidence", 0.0))
                return {"routing": result}

            except Exception as e:
                span.record_exception(e)
                return {"error": f"[route_node] {e}"}
    return route_node


def make_vision_node(vision_url: str, registry=None):
    """
    registry: ModuleRegistry instance - reads facts from module_registry.yaml.
    If None, falls back to a hardcoded dict (test/dev use only).
    """
    async def vision_node(state: OrchestratorState) -> dict:
        if state.get("error"):
            # Return an empty dict so LangGraph keeps the current state unchanged
            return {"model_output": state.get("model_output")}
        routing = state.get("routing") or {}
        organ = routing.get("organ", "breast")
        module_key = routing.get("module_key", "us_breast")

        with get_tracer().start_as_current_span("graph.vision") as span:
            span.set_attribute("image_id",   state.get("image_id", ""))
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
                span.set_attribute("vision.top_label",  result.get("top_label", ""))
                span.set_attribute("vision.confidence", result.get("confidence", 0.0))
                return {"model_output": result}
            except Exception as e:
                span.record_exception(e)
                return {"error": f"[vision_node] {e}"}
    return vision_node


def make_spatial_node(knowledge_url: str):
    """
    Separates the spatial derive step (from the segmentation mask) out of
    knowledge_node.

    Reason: spatial_derived (bbox, area, aspect_ratio, circularity) is computed
    PURELY from the segmentation mask -- it has nothing to do with severity
    mapping (rule-based). It used to only be available after knowledge_node
    (running in parallel with cot_reasoning_node) finished, so cot_reasoning_node
    always read spatial=None/empty (race condition). This node runs right
    after vision, before fanning out to knowledge/cot_reasoning, so both see
    the real spatial data instead of an empty {}.
    """
    async def spatial_node(state: OrchestratorState) -> dict:
        if state.get("error"):
            return {"spatial": state.get("spatial")}

        routing = state.get("routing") or {}
        mo = state.get("model_output") or {}

        payload = {
            "modality":            routing.get("modality", "ultrasound"),
            "organ":               routing.get("organ", "breast"),
            "top_label":           mo.get("top_label", "benign"),
            "confidence":          mo.get("confidence", 0.0),
            "all_scores":          mo.get("all_scores", {}),
            "mask_png_base64":     mo.get("mask_png_base64", ""),
            "original_size":       list(mo.get("original_size", [512, 512])),
            "pixel_spacing_mm":    0.1,
            "bottleneck_features": mo.get("bottleneck_features", {}),
        }

        with get_tracer().start_as_current_span("graph.spatial") as span:
            span.set_attribute("image_id", state.get("image_id", ""))
            try:
                result = await _post_json_async(f"{knowledge_url}/map", payload)
                spatial = result.get("spatial_derived", {})
                span.set_attribute("spatial.area_cm2", spatial.get("area_cm2", 0.0))
                return {"spatial": spatial}
            except Exception as e:
                span.record_exception(e)
                return {"error": f"[spatial_node] {e}"}
    return spatial_node


def make_knowledge_node(knowledge_url: str):
    async def knowledge_node(state: OrchestratorState) -> dict:
        if state.get("error"):
            return {"knowledge": state.get("knowledge"), "spatial": state.get("spatial")}

        routing = state.get("routing") or {}
        mo = state.get("model_output") or {}

        payload = {
            "modality":            routing.get("modality", "ultrasound"),
            "organ":               routing.get("organ", "breast"),
            "top_label":           mo.get("top_label", "benign"),
            "confidence":          mo.get("confidence", 0.0),
            "all_scores":          mo.get("all_scores", {}),
            "mask_png_base64":     mo.get("mask_png_base64", ""),
            "original_size":       list(mo.get("original_size", [512, 512])),
            "pixel_spacing_mm":    0.1,
            # Pass-through bottleneck_features - the knowledge service doesn't
            # process it, just forwards it so the orchestrator can use it in
            # the CoT and qa_agent prompts
            "bottleneck_features": mo.get("bottleneck_features", {}),
        }

        with get_tracer().start_as_current_span("graph.knowledge") as span:
            span.set_attribute("image_id",       state.get("image_id", ""))
            span.set_attribute("knowledge.organ", routing.get("organ", ""))
            span.set_attribute("knowledge.label", mo.get("top_label", ""))
            try:
                result = await _post_json_async(f"{knowledge_url}/map", payload)
                km = result.get("knowledge_mapped", {})
                span.set_attribute("knowledge.severity",       km.get("severity", ""))
                span.set_attribute("knowledge.severity_level", km.get("severity_level", 0))
                # spatial was already computed by spatial_node earlier and runs
                # in parallel; still returned here for backward compatibility
                # if spatial_node hasn't run (e.g. old AsyncSequentialFallback
                # or state doesn't have spatial yet).
                return {
                    "knowledge": km,
                    "spatial":   state.get("spatial") or result.get("spatial_derived", {}),
                }
            except Exception as e:
                span.record_exception(e)
                return {"error": f"[knowledge_node] {e}"}
    return knowledge_node


def make_cot_node(llm_client):
    """
    CoT reasoning node - runs in parallel with the knowledge node.
    Does NOT read the mapper's result before reasoning.
    Output must have the same structure as KnowledgeMapped so they're comparable.
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

        prompt = _build_cot_prompt(
            top_label=mo.get("top_label", "benign"),
            confidence=mo.get("confidence", 0.0),
            all_scores=mo.get("all_scores", {}),
            spatial=sd,
            bottleneck=mo.get("bottleneck_features", {}),
            rag_chunks=rag_chunks,
            organ=routing.get("organ", "breast"),
        )

        with get_tracer().start_as_current_span("graph.cot_reasoning") as span:
            span.set_attribute("image_id",    state.get("image_id", ""))
            span.set_attribute("cot.organ",   routing.get("organ", ""))
            span.set_attribute("cot.label",   mo.get("top_label", ""))
            raw = None
            try:
                raw = await asyncio.to_thread(
                    llm_client.generate, prompt, COT_SYSTEM_PROMPT
                )
                clean = raw.strip().replace("```json", "").replace("```", "").strip()
                parsed = json.loads(clean)

                cot = {
                    "severity":       str(parsed.get("severity", "incidental")),
                    "severity_level": int(parsed.get("severity_level", 1)),
                    "icd10_hint":     str(parsed.get("icd10_hint", "R93.8")),
                    "risk_category":  str(parsed.get("risk_category", "undetermined")),
                    "reasoning":      str(parsed.get("reasoning", "")),
                }
                span.set_attribute("cot.severity_level", cot["severity_level"])
            except Exception as e:
                span.record_exception(e)
                raw_preview = raw[:200] if raw else ""
                print(f"[cot_node] Parse error: {e}. Falling back to undetermined.")
                cot = {
                    "severity":       "undetermined",
                    "severity_level": 0,
                    "icd10_hint":     "R93.8",
                    "risk_category":  "undetermined",
                    "reasoning":      f"Parse error: {e}. Raw response: {raw_preview}",
                }

        return {"cot_result": cot}
    return cot_reasoning_node


def make_rag_node(rag_store):
    """
    RAG node runs in parallel with the vision branch.
    Uses the user's question for the query since vision hasn't finished yet.
    """
    async def rag_node(state: OrchestratorState) -> OrchestratorState:
        if state.get("error"):
            return state

        question = state.get("question", "")
        routing = state.get("routing") or {}
        organ = routing.get("organ")

        try:
            if rag_store and rag_store.is_ready():
                meta_list = await asyncio.to_thread(
                    rag_store.retrieve_with_meta, question, 5, organ
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
    """Waits for all three branches (knowledge, cot, rag) before passing to consistency_guard."""
    async def merge_node(state: OrchestratorState) -> OrchestratorState:
        if state.get("error"):
            return state
        if not state.get("knowledge"):
            state["error"] = "[merge_node] knowledge output is missing -- vision/knowledge branch failed."
            return state

        # Once knowledge is available, perform a second retrieve with a richer context.
        # Does not call the LLM - just concatenates strings from existing fields.
        return state
    return merge_node


def make_second_rag_retrieval(rag_store):
    """
    Second retrieve after the knowledge result is available (top_label, icd10_hint, organ).
    Merges with the first retrieve and reranks all candidates.
    """
    async def second_retrieve(state: OrchestratorState, question: str) -> tuple:
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

        # Second query concatenates context: question + label + organ + icd10
        enriched_query = f"{question} {top_label} {organ or ''} {icd10}".strip()

        try:
            meta2 = await asyncio.to_thread(
                rag_store.retrieve_with_meta, enriched_query, 5, organ
            )
        except Exception as e:
            print(f"[second_rag] Second retrieve error: {e}")
            meta2 = []

        # Merge both retrieves, dedup chunks by content
        existing_meta = state.get("rag_meta") or []
        existing_texts = {m["chunk"] for m in existing_meta}
        combined = list(existing_meta)
        for m in meta2:
            if m["chunk"] not in existing_texts:
                combined.append(m)
                existing_texts.add(m["chunk"])

        # Rerank all candidates, keep top 3
        reranked = rag_store.rerank(question, combined, top_n=3)
        chunks = [m["chunk"] for m in reranked]
        return chunks, reranked

    return second_retrieve


def make_consistency_guard_node(rag_store):
    """
    New node placed between merge and qa_agent.
    1. Performs the second RAG retrieve with a richer context.
    2. Compares mapper and CoT severity_level, sets the consensus flag.
    3. Separately compares mapper and CoT icd10_hint, sets the icd10_agreement flag.

    consensus and icd10_agreement are two different clinical questions
    (severity vs disease code) so they are not collapsed into a single flag.
    """
    second_retrieve = make_second_rag_retrieval(rag_store)

    async def consistency_guard_node(state: OrchestratorState) -> OrchestratorState:
        if state.get("error"):
            return state

        question = state.get("question", "")

        with get_tracer().start_as_current_span("graph.consistency_guard") as span:
            span.set_attribute("image_id", state.get("image_id", ""))

            final_chunks, final_meta = await second_retrieve(state, question)
            state["rag_chunks"] = final_chunks
            state["rag_meta"]   = final_meta

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

            span.set_attribute("guard.mapper_level", mapper_level)
            span.set_attribute("guard.cot_level",    cot_level)
            span.set_attribute("guard.mapper_icd10", mapper_icd10 or "")
            span.set_attribute("guard.cot_icd10",    cot_icd10 or "")
            span.set_attribute("guard.consensus",    str(state["consensus"]))
            span.set_attribute("guard.icd10_agreement", str(state["icd10_agreement"]))

        return state
    return consistency_guard_node


def make_qa_agent_node(llm_client, rag_store):
    """Generates the 3-tier report from vision, knowledge, CoT, and RAG results."""
    async def qa_agent_node(state: OrchestratorState) -> OrchestratorState:
        if state.get("error"):
            return state

        routing    = state.get("routing") or {}
        mo         = state.get("model_output") or {}
        km         = state.get("knowledge") or {}
        sd         = state.get("spatial") or {}
        rag_chunks = state.get("rag_chunks") or []
        rag_meta   = state.get("rag_meta") or []
        cot_result = state.get("cot_result")
        consensus  = state.get("consensus")
        icd10_agreement = state.get("icd10_agreement")

        unified = {
            "modality":         routing.get("modality", "ultrasound"),
            "organ":            routing.get("organ", "breast"),
            "image_id":         state["image_id"],
            "model_output":     mo,
            "knowledge_mapped": km,
            "spatial_derived":  sd,
            "coverage_note":    "Model trained on BUSI dataset (benign/malignant/normal only).",
        }

        prompt = _build_report_prompt(
            unified, state["question"], rag_chunks, cot_result, consensus, icd10_agreement
        )

        with get_tracer().start_as_current_span("graph.qa_agent") as span:
            span.set_attribute("image_id",          state.get("image_id", ""))
            span.set_attribute("qa.consensus",       str(consensus))
            span.set_attribute("qa.icd10_agreement", str(icd10_agreement))
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
            "modality":           routing.get("modality", "ultrasound"),
            "organ":              routing.get("organ", "breast"),
            "label":              mo.get("top_label", "unknown"),
            "confidence":         mo.get("confidence", 0.0),
            "risk_category":      km.get("risk_category", "unknown"),
            "severity":           km.get("severity", "unknown"),
            "severity_level":     km.get("severity_level", 1),
            "icd10_hint":         tier1_icd10_hint,
            "location_quadrant":  sd.get("location_quadrant", "unknown"),
            "bbox":               sd.get("bbox", [0, 0, 0, 0]),
            "area_cm2":           sd.get("area_cm2", 0.0),
            "aspect_ratio":       sd.get("aspect_ratio", 1.0),
            "circularity":        sd.get("circularity", 1.0),
            "confidence_calibration_note": km.get("confidence_calibration_note"),
            "hint_conflict":          routing.get("hint_conflict", False),
            "hint_resolution_note":   routing.get("hint_resolution_note"),
            "icd10_agreement":        icd10_agreement,
        }

        rag_sources = [
            {"file": m.get("source_file", "unknown"), "page": m.get("page_number", 0)}
            for m in rag_meta
        ]

        state["report"] = {
            "image_id":                        state["image_id"],
            "tier_1_structured":               tier1,
            "tier_2_radiological_description": tier2,
            "tier_3_diagnostic_suggestion":    tier3,
            "rag_sources":                     rag_sources,
            "rag_disabled_warning": (
                None if (rag_store and rag_store.is_ready()) else
                "RAG context not available -- report generated from classification "
                "label and hardcoded mapping only, without clinical guideline retrieval."
            ),
            "mapper_result":          km,
            "cot_result":             cot_result,
            "consensus":              consensus,
            "icd10_agreement":        icd10_agreement,
            "_rag_chunks_internal":   rag_chunks,
        }
        return state
    return qa_agent_node


def _parse_tiers(llm_text: str) -> tuple:
    """
    Parse the LLM response -> (tier2_text, tier3_text).
    Looks for 'TIER 2' and 'TIER 3' markers, falls back to a midpoint split if not found.
    """
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


# Building and compiling the graph

def build_graph(services_cfg: dict, llm_client, rag_store, registry=None):
    """
    Build and compile the LangGraph pipeline with real fan-out/fan-in.

    Args:
        services_cfg: {
            'router_url':    'http://router:8001',
            'vision_url':    'http://vision:8002',
            'knowledge_url': 'http://knowledge:8003',
        }
        llm_client:  BaseLLMClient instance
        rag_store:   FAISSStore instance
        registry:    ModuleRegistry instance, reads facts from module_registry.yaml.

    Returns a compiled async graph (or AsyncSequentialFallback if langgraph is unavailable).
    """
    router_url    = services_cfg.get("router_url",    "http://router:8001")
    vision_url    = services_cfg.get("vision_url",    "http://vision:8002")
    knowledge_url = services_cfg.get("knowledge_url", "http://knowledge:8003")

    route_node              = make_route_node(router_url)
    vision_node             = make_vision_node(vision_url, registry=registry)
    spatial_node            = make_spatial_node(knowledge_url)
    knowledge_node          = make_knowledge_node(knowledge_url)
    cot_node                = make_cot_node(llm_client)
    rag_node                = make_rag_node(rag_store)
    merge_node              = make_merge_node()
    consistency_guard_node  = make_consistency_guard_node(rag_store)
    qa_agent_node           = make_qa_agent_node(llm_client, rag_store)

    if LANGGRAPH_AVAILABLE:
        g = StateGraph(OrchestratorState)
        g.add_node("route",             route_node)
        g.add_node("vision",            vision_node)
        g.add_node("spatial",           spatial_node)
        g.add_node("knowledge",         knowledge_node)
        g.add_node("cot_reasoning",     cot_node)
        g.add_node("rag_retrieve",      rag_node)
        g.add_node("merge",             merge_node)
        g.add_node("consistency_guard", consistency_guard_node)
        g.add_node("qa_agent",          qa_agent_node)

        g.set_entry_point("route")

        # Fan-out: vision and rag run in parallel after route
        g.add_edge("route",   "vision")
        g.add_edge("route",   "rag_retrieve")
        # spatial runs right after vision, BEFORE fanning out to knowledge/cot_reasoning,
        # so both branches see the real spatial_derived data (avoids the old race
        # condition -- cot_reasoning used to always read spatial={} because it ran
        # in parallel with knowledge_node, the only node computing spatial at the time).
        g.add_edge("vision",  "spatial")
        g.add_edge("spatial", "knowledge")
        # CoT runs after spatial (needs model_output + spatial) but in parallel with knowledge
        g.add_edge("spatial", "cot_reasoning")

        # Fan-in: merge waits for ALL THREE branches
        g.add_edge(["knowledge", "cot_reasoning", "rag_retrieve"], "merge")
        g.add_edge("merge",             "consistency_guard")
        g.add_edge("consistency_guard", "qa_agent")
        g.add_edge("qa_agent",          END)

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
    Mirrors build_graph's real edges exactly: route -> vision -> spatial ->
    {knowledge, cot_reasoning} run in parallel (both depend only on spatial's
    output, not on each other) -> merge waits for knowledge + cot_reasoning +
    rag_retrieve -> consistency_guard -> qa_agent.

    IMPORTANT: route_node, vision_node, spatial_node, knowledge_node, and
    cot_node all return a PARTIAL dict (e.g. {"routing": ...} or
    {"error": ...}), not the full state -- this is the real LangGraph
    contract, since LangGraph merges partial returns into shared state via
    the Annotated[..., reducer] fields on OrchestratorState. This fallback
    must replicate that merging manually (state.update(partial)) instead of
    replacing state outright, or keys like image_bytes/question/image_id
    silently disappear after the first node runs. rag_node, merge_node,
    consistency_guard_node, and qa_agent_node are the exception -- they take
    and return the full state directly (see their signatures in this file).

    cot_node MUST run on a state snapshot taken before knowledge_node executes,
    not after -- otherwise CoT would see the mapper's result (severity,
    icd10_hint) before reasoning, defeating the whole point of comparing two
    independent assessments (see make_cot_node's docstring).
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
        self.image_nodes = image_nodes          # [route, vision, spatial, knowledge]
        self.cot_node = cot_node
        self.rag_node = rag_node
        self.merge_node = merge_node
        self.consistency_guard_node = consistency_guard_node
        self.qa_agent_node = qa_agent_node

    async def ainvoke(self, state: dict) -> dict:
        import copy

        # Step 1: route -- merge its partial dict into state, don't replace state.
        # Indexed (not destructured) so an error-path test can supply a
        # shorter image_nodes list and still short-circuit safely below,
        # without ever needing to unpack the rest of the list.
        route_node = self.image_nodes[0]
        state.update(await route_node(state))
        if state.get("error"):
            return state

        vision_node, spatial_node, knowledge_node = self.image_nodes[1:]

        # Step 2: vision and rag run in parallel after route
        state_for_rag = copy.deepcopy(state)
        vision_partial, rag_result = await asyncio.gather(
            vision_node(state),
            self.rag_node(state_for_rag),
        )
        state.update(vision_partial)
        state["rag_chunks"] = rag_result.get("rag_chunks", [])
        state["rag_meta"]   = rag_result.get("rag_meta", [])

        if state.get("error"):
            return state

        # Step 3: spatial runs right after vision, before fanning out to
        # knowledge/cot_reasoning, so both see the real spatial_derived data
        state.update(await spatial_node(state))
        if state.get("error"):
            return state

        # Step 4: knowledge and cot_reasoning run in parallel, each on its own
        # deepcopy of the SAME pre-knowledge state -- cot_node must never see
        # state["knowledge"] before it finishes reasoning.
        state_for_cot = copy.deepcopy(state)
        knowledge_partial, cot_partial = await asyncio.gather(
            knowledge_node(state),
            self.cot_node(state_for_cot),
        )
        state.update(knowledge_partial)
        if not state.get("error"):
            state["cot_result"] = cot_partial.get("cot_result")

        if state.get("error"):
            return state

        # Step 5: merge (fan-in for knowledge + cot_reasoning + rag_retrieve)
        # merge_node/consistency_guard_node/qa_agent_node take and return the
        # full state directly, unlike the partial-dict nodes above.
        state = await self.merge_node(state)
        if state.get("error"):
            return state

        # Step 6: consistency_guard (second RAG + compare)
        state = await self.consistency_guard_node(state)
        if state.get("error"):
            return state

        # Step 7: qa_agent
        state = await self.qa_agent_node(state)
        return state


# External pipeline entry points

async def run_pipeline_async(
    graph,
    image_bytes: bytes,
    question: str,
    image_id: str = None,
    modality_hint: str = None,
    organ_hint: str = None,
) -> dict:
    """
    Run the entire pipeline from bytes -> ReportOutput dict (async).

    Used by the FastAPI async handler - do not call asyncio.run() nested
    inside an already-async context (will raise a runtime error).
    """
    if image_id is None:
        image_id = uuid.uuid4().hex[:12]

    initial_state: OrchestratorState = {
        "image_bytes":   image_bytes,
        "question":      question,
        "image_id":      image_id,
        "modality_hint": modality_hint,
        "organ_hint":    organ_hint,
        "routing":       None,
        "model_output":  None,
        "knowledge":     None,
        "spatial":       None,
        "cot_result":    None,
        "rag_chunks":    [],
        "rag_meta":      [],
        "consensus":     None,
        "icd10_agreement": None,
        "report":        None,
        "error":         None,
    }

    final_state = await graph.ainvoke(initial_state)

    if final_state.get("error"):
        raise RuntimeError(final_state["error"])

    return final_state["report"]


def run_pipeline(
    graph,
    image_bytes: bytes,
    question: str,
    image_id: str = None,
    modality_hint: str = None,
    organ_hint: str = None,
) -> dict:
    """
    Sync version of run_pipeline_async - only used outside FastAPI
    (e.g. a manual script). Inside FastAPI, use run_pipeline_async.
    """
    return asyncio.run(run_pipeline_async(
        graph, image_bytes, question, image_id, modality_hint, organ_hint
    ))