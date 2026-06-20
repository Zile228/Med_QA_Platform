"""
services/orchestrator/graph.py
================================
LangGraph pipeline - dieu phoi toan bo flow.

Graph (fan-out/fan-in sau khi them CoT):
    route -> vision -> knowledge  --|
          -> cot_reasoning        --|-> merge -> consistency_guard -> qa_agent
          -> rag_retrieve         --|

    Ba nhanh sau route chay song song (knowledge + cot_reasoning + rag).
    merge doi CA BA nhanh xong roi moi chay consistency_guard.
    consistency_guard so sanh severity_level cua mapper vs CoT, gan flag consensus.

Moi node = 1 HTTP call den service tuong ung:
    route_node            -> POST router:8001/route
    vision_node           -> POST vision:8002/analyze/{modality}
    knowledge_node        -> POST knowledge:8003/map
    cot_reasoning_node    -> LLM Chain-of-Thought (chay song song voi knowledge)
    rag_node              -> FAISS retrieve (local, khong HTTP)
    merge_node            -> kiem tra state, short-circuit neu co loi
    consistency_guard_node -> so sanh mapper vs CoT, gan flag consensus
    qa_agent_node         -> LLM generate 3-tier report

Public API:
    build_graph(services_cfg, llm_client, rag_store, registry) -> CompiledGraph
    run_pipeline(graph, image_bytes, question, image_id,
                 modality_hint, organ_hint) -> ReportOutput dict
    run_pipeline_async(...) -> ReportOutput dict  (cho FastAPI async handler)
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
    print("[orchestrator] langgraph chua install, dung sequential fallback")

from shared.schemas import (
    RoutingResult, ModelOutput, KnowledgeMapped,
    SpatialDerived, UnifiedOutput, ReportOutput, Tier1Structured,
    RagSource, CoTResult,
)
from shared.telemetry import get_tracer


# Reducer cho LangGraph state

def _keep_last(a, b):
    """Reducer: giu gia tri moi nhat khac None."""
    return b if b is not None else a


def _merge_list(a, b):
    """Reducer: giu list moi nhat neu khong rong."""
    return b if b else a


class OrchestratorState(TypedDict):
    # Input - khong thay doi sau khi init
    image_bytes:    bytes
    question:       str
    image_id:       str
    modality_hint:  Optional[str]
    organ_hint:     Optional[str]

    # Layer 1 output
    routing:        Annotated[Optional[dict], _keep_last]

    # Layer 2 output
    model_output:   Annotated[Optional[dict], _keep_last]

    # Layer 3 output tu rule-based mapper
    knowledge:      Annotated[Optional[dict], _keep_last]
    spatial:        Annotated[Optional[dict], _keep_last]

    # Layer 3 output tu CoT engine (chay song song voi knowledge)
    cot_result:     Annotated[Optional[dict], _keep_last]

    # RAG context tu nhanh song song voi vision
    rag_chunks:     Annotated[list, _merge_list]
    rag_meta:       Annotated[list, _merge_list]

    # Ket qua so sanh mapper vs CoT
    consensus:      Annotated[Optional[bool], _keep_last]

    # Final output
    report:         Annotated[Optional[dict], _keep_last]

    # Error tracking
    error:          Annotated[Optional[str], _keep_last]


# Helper goi HTTP

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
    """Phien ban async cua _post_multipart - dung cho async node."""
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


# Xay dung prompt cho LLM

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
    """Dinh dang bottleneck_features thanh text mo ta de dua vao prompt."""
    if not bottleneck:
        return "Bottleneck features: khong co du lieu."

    energy = bottleneck.get("activation_energy", "N/A")
    hotspot = bottleneck.get("attention_hotspot_grid", [])
    top_ch = bottleneck.get("top_channel_activations", [])

    # hotspot la 1 cap toa do [row, col] (vi tri model tap trung nhat),
    # KHONG phai ma tran 2D -- xem services/vision/us_breast/model.py: hotspot_pos = [row, col]
    hotspot_str = ""
    if hotspot and len(hotspot) == 2:
        hotspot_str = f"vung [{hotspot[0]},{hotspot[1]}] (grid 7x7)"

    return (
        f"Bottleneck activation energy: {energy}. "
        f"Attention tap trung manh nhat: {hotspot_str or 'khong xac dinh'}. "
        f"Top channels: {top_ch[:3] if top_ch else 'N/A'}."
    )


def _build_report_prompt(
    unified: dict,
    question: str,
    rag_chunks: list,
    cot_result: Optional[dict] = None,
    consensus: Optional[bool] = None,
) -> str:
    """Build prompt cho LLM tu UnifiedOutput + user question + RAG context + CoT."""
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
### Bat dong giua Rule Engine va AI Reasoning (consensus: false)
Rule engine (mapper): severity={mapper_sev}, icd10={mapper_icd}
CoT reasoning:        severity={cot_sev},   icd10={cot_icd}, risk={cot_risk}
CoT audit trail: {cot_reason}

IMPORTANT: Trong Tier 3, trinh bay RO CA HAI goc nhin nay va neu ro:
"Can radiologist xac nhan truc tiep do co bat dong giua rule-based va AI reasoning."
"""
        else:
            cot_reason = cot_result.get("reasoning", "")
            consensus_block = f"""
### AI Reasoning (dong thuan voi rule engine)
CoT audit trail: {cot_reason}
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
Note: neu vung attention lech nhieu so voi bbox segmentation, day la tin hieu model khong chac.

### Clinical Coverage Note
{unified.get('coverage_note', '')}
{consensus_block}
### Retrieved Clinical Guidelines
{rag_context}

---

Please generate:

**TIER 2 -- Radiological Description** (2-3 sentences, professional radiological language):
Describe the findings using the spatial and classification data above.
Reference the ICD-10 code {km.get('icd10_hint', '')} and severity {km.get('severity', '')} explicitly.

**TIER 3 -- Diagnostic Suggestion** (2-3 sentences, include follow-up recommendation):
Based on the findings and clinical guidelines, suggest next steps. Do NOT make a definitive diagnosis.
Start with: "AI-assisted suggestion (must be confirmed by radiologist):"
{"Include both rule-based and AI reasoning perspectives, ending with the disagreement note above." if consensus is False else ""}

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
    Build prompt Chain-of-Thought de suy luan doc lap voi mapper.
    CoT KHONG duoc biet ket qua mapper truoc khi suy luan.
    Output la JSON object khop voi CoTResult schema.
    """
    rag_text = "\n\n".join(rag_chunks) if rag_chunks else "Khong co tai lieu lam sang."
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
    Build prompt multi-turn cho chatbot tu context da co + history.

    unified_context: dict chua unified output + report tu lan analyze goc.
    history: list[dict] voi keys 'role' va 'content'.
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


# Cac node cua LangGraph

ALLOW_DEGRADED_ROUTER = os.getenv("ALLOW_DEGRADED_ROUTER", "false").lower() == "true"


def make_route_node(router_url: str):
    """
    Async node - goi /route voi hint neu co.
    hint_conflict/hint_resolution_note/final_decision_source duoc copy vao state
    de report_node co the dua vao tier_1_structured.
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
                            "Router dang chay voi random weights (chua co checkpoint da train). "
                            "Routing decision khong co y nghia va co the route nham modality "
                            "ma khong co canh bao. Pipeline bi chan de tranh sinh report sai. "
                            "Dat checkpoint vao models/checkpoints/router_effnet_b0.pt, hoac set "
                            "ALLOW_DEGRADED_ROUTER=true trong .env neu day la moi truong dev/demo."
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
    registry: ModuleRegistry instance - doc that tu module_registry.yaml.
    Neu None, fallback ve dict hardcode (chi dung cho test/dev).
    """
    async def vision_node(state: OrchestratorState) -> dict:
        if state.get("error"):
            # Tra ve dict rong de LangGraph giu nguyen state hien tai, khong ghi de gi
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
    Tach rieng buoc derive spatial (tu segmentation mask) ra khoi knowledge_node.

    Ly do: spatial_derived (bbox, area, aspect_ratio, circularity) duoc tinh
    THUAN TUY tu mask segmentation -- khong lien quan gi den severity mapping
    (rule-based). Truoc day no chi co san sau khi knowledge_node (chay song
    song voi cot_reasoning_node) hoan tat, nen cot_reasoning_node luon doc
    duoc spatial=None/rong (race condition). Node nay chay ngay sau vision,
    truoc khi fan-out sang knowledge/cot_reasoning, de ca hai deu thay spatial
    that su thay vi {} rong.
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
            # Pass-through bottleneck_features - knowledge service khong xu ly,
            # chi truyen de orchestrator co the dua vao prompt cua CoT va qa_agent
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
                # spatial da duoc spatial_node tinh truoc do va chay song song;
                # van tra ve o day de tuong thich nguoc neu spatial_node chua chay
                # (vd AsyncSequentialFallback cu hoac state chua co spatial).
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
    CoT reasoning node - chay song song voi knowledge node.
    KHONG doc ket qua mapper truoc khi suy luan.
    Output phai co cung cau truc voi KnowledgeMapped de so sanh duoc.
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
                print(f"[cot_node] Parse loi: {e}. Fallback ve undetermined.")
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
    RAG node chay song song voi nhanh vision.
    Query dung question cua user vi vision chua chay xong luc nay.
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
    """Doi ca ba nhanh (knowledge, cot, rag) xong roi moi cho qua consistency_guard."""
    async def merge_node(state: OrchestratorState) -> OrchestratorState:
        if state.get("error"):
            return state
        if not state.get("knowledge"):
            state["error"] = "[merge_node] knowledge output chua co -- nhanh vision/knowledge bi loi."
            return state

        # Sau khi co knowledge, thuc hien lan retrieve thu hai voi context giau hon.
        # Khong goi LLM - chi ghep string tu cac field co san.
        return state
    return merge_node


def make_second_rag_retrieval(rag_store):
    """
    Lan retrieve thu hai sau khi co ket qua knowledge (top_label, icd10_hint, organ).
    Gop voi lan thu nhat va rerank toan bo candidate.
    """
    async def second_retrieve(state: OrchestratorState, question: str) -> tuple:
        """
        Tra ve (all_chunks: list[str], all_meta: list[dict]) sau khi gop va rerank.
        Duoc goi tu consistency_guard_node, khong phai LangGraph node doc lap.
        """
        if not (rag_store and rag_store.is_ready()):
            return state.get("rag_chunks") or [], state.get("rag_meta") or []

        km = state.get("knowledge") or {}
        routing = state.get("routing") or {}
        organ = routing.get("organ")

        mo = state.get("model_output") or {}
        top_label = mo.get("top_label", "")
        icd10 = km.get("icd10_hint", "")

        # Query lan 2 ghep context: question + label + organ + icd10
        enriched_query = f"{question} {top_label} {organ or ''} {icd10}".strip()

        try:
            meta2 = await asyncio.to_thread(
                rag_store.retrieve_with_meta, enriched_query, 5, organ
            )
        except Exception as e:
            print(f"[second_rag] Lan 2 retrieve error: {e}")
            meta2 = []

        # Gop 2 lan, loai chunk trung lap theo noi dung
        existing_meta = state.get("rag_meta") or []
        existing_texts = {m["chunk"] for m in existing_meta}
        combined = list(existing_meta)
        for m in meta2:
            if m["chunk"] not in existing_texts:
                combined.append(m)
                existing_texts.add(m["chunk"])

        # Rerank toan bo candidate, giu top 3
        reranked = rag_store.rerank(question, combined, top_n=3)
        chunks = [m["chunk"] for m in reranked]
        return chunks, reranked

    return second_retrieve


def make_consistency_guard_node(rag_store):
    """
    Node moi dat giua merge va qa_agent.
    1. Thuc hien lan RAG retrieve thu hai voi context giau.
    2. So sanh severity_level cua mapper va CoT.
    3. Gan flag consensus: True neu chenh lech <= 1, False neu > 1.
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

            mapper_level = (state.get("knowledge") or {}).get("severity_level", 0)
            cot = state.get("cot_result") or {}
            cot_level = cot.get("severity_level", 0)

            if cot_level == 0 or cot.get("severity") == "undetermined":
                state["consensus"] = None
            elif abs(mapper_level - cot_level) <= 1:
                state["consensus"] = True
            else:
                state["consensus"] = False

            span.set_attribute("guard.mapper_level", mapper_level)
            span.set_attribute("guard.cot_level",    cot_level)
            span.set_attribute("guard.consensus",    str(state["consensus"]))

        return state
    return consistency_guard_node


def make_qa_agent_node(llm_client, rag_store):
    """Sinh bao cao 3 tang tu ket qua vision, knowledge, CoT va RAG."""
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
            unified, state["question"], rag_chunks, cot_result, consensus
        )

        with get_tracer().start_as_current_span("graph.qa_agent") as span:
            span.set_attribute("image_id",          state.get("image_id", ""))
            span.set_attribute("qa.consensus",       str(consensus))
            span.set_attribute("qa.rag_chunks_used", len(rag_chunks))
            try:
                llm_response = await asyncio.to_thread(
                    llm_client.generate, prompt, SYSTEM_PROMPT
                )
            except Exception as e:
                span.record_exception(e)
                llm_response = f"[LLM unavailable: {e}]"

        tier2, tier3 = _parse_tiers(llm_response)

        tier1 = {
            "modality":           routing.get("modality", "ultrasound"),
            "organ":              routing.get("organ", "breast"),
            "label":              mo.get("top_label", "unknown"),
            "confidence":         mo.get("confidence", 0.0),
            "risk_category":      km.get("risk_category", "unknown"),
            "severity":           km.get("severity", "unknown"),
            "severity_level":     km.get("severity_level", 1),
            "icd10_hint":         km.get("icd10_hint", "unknown"),
            "location_quadrant":  sd.get("location_quadrant", "unknown"),
            "bbox":               sd.get("bbox", [0, 0, 0, 0]),
            "area_cm2":           sd.get("area_cm2", 0.0),
            "aspect_ratio":       sd.get("aspect_ratio", 1.0),
            "circularity":        sd.get("circularity", 1.0),
            "confidence_calibration_note": km.get("confidence_calibration_note"),
            "hint_conflict":          routing.get("hint_conflict", False),
            "hint_resolution_note":   routing.get("hint_resolution_note"),
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
            "_rag_chunks_internal":   rag_chunks,
        }
        return state
    return qa_agent_node


def _parse_tiers(llm_text: str) -> tuple:
    """
    Parse LLM response -> (tier2_text, tier3_text).
    Tim markers 'TIER 2' va 'TIER 3', fallback split neu khong tim thay.
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


# Xay dung va compile graph

def build_graph(services_cfg: dict, llm_client, rag_store, registry=None):
    """
    Build va compile LangGraph pipeline voi fan-out/fan-in that su.

    Args:
        services_cfg: {
            'router_url':    'http://router:8001',
            'vision_url':    'http://vision:8002',
            'knowledge_url': 'http://knowledge:8003',
        }
        llm_client:  BaseLLMClient instance
        rag_store:   FAISSStore instance
        registry:    ModuleRegistry instance, doc that tu module_registry.yaml.

    Returns compiled async graph (hoac AsyncSequentialFallback neu langgraph khong co).
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

        # Fan-out: vision va rag chay song song sau route
        g.add_edge("route",   "vision")
        g.add_edge("route",   "rag_retrieve")
        # spatial chay ngay sau vision, TRUOC khi fan-out sang knowledge/cot_reasoning,
        # de ca hai nhanh deu thay spatial_derived that su (tranh race condition cu --
        # cot_reasoning truoc day luon doc spatial={} vi no chay song song voi
        # knowledge_node, node duy nhat tinh spatial luc do).
        g.add_edge("vision",  "spatial")
        g.add_edge("spatial", "knowledge")
        # CoT chay sau spatial (can model_output + spatial) nhung song song voi knowledge
        g.add_edge("spatial", "cot_reasoning")

        # Fan-in: merge doi CA BA nhanh xong
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
    Fallback khi langgraph khong install.
    Mo phong fan-out bang asyncio.gather cho cac nhanh song song,
    dam bao test pass ca khi langgraph thieu trong moi truong CI nhe.
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

        # Buoc 1: route
        state = await self.image_nodes[0](state)
        if state.get("error"):
            return state

        # Buoc 2: fan-out vision (+ knowledge) va rag chay song song
        # deepcopy de dam bao 2 nhanh khong chia se reference list mutable
        state_for_rag = copy.deepcopy(state)

        async def run_image_branch(s):
            for node in self.image_nodes[1:]:
                s = await node(s)
            return s

        state, state_for_rag = await asyncio.gather(
            run_image_branch(state),
            self.rag_node(state_for_rag),
        )

        state["rag_chunks"] = state_for_rag.get("rag_chunks", [])
        state["rag_meta"]   = state_for_rag.get("rag_meta", [])

        # Buoc 3: CoT chay sau vision, song song voi merge
        state_for_cot = copy.deepcopy(state)
        state, state_for_cot = await asyncio.gather(
            self.merge_node(state),
            self.cot_node(state_for_cot),
        )
        if not state.get("error"):
            state["cot_result"] = state_for_cot.get("cot_result")

        if state.get("error"):
            return state

        # Buoc 4: consistency_guard (second RAG + compare)
        state = await self.consistency_guard_node(state)
        if state.get("error"):
            return state

        # Buoc 5: qa_agent
        state = await self.qa_agent_node(state)
        return state


# Ham chay pipeline tu ngoai

async def run_pipeline_async(
    graph,
    image_bytes: bytes,
    question: str,
    image_id: str = None,
    modality_hint: str = None,
    organ_hint: str = None,
) -> dict:
    """
    Chay toan bo pipeline tu bytes -> ReportOutput dict (async).

    Dung cho FastAPI async handler - khong goi asyncio.run() long trong
    context da async san (se loi runtime).
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
    Phien ban sync cua run_pipeline_async - chi dung khi chay ngoai
    FastAPI (vd. script thu cong). Trong FastAPI, dung run_pipeline_async.
    """
    return asyncio.run(run_pipeline_async(
        graph, image_bytes, question, image_id, modality_hint, organ_hint
    ))