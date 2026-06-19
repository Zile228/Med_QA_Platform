"""
services/orchestrator/graph.py
================================
LangGraph pipeline -- dieu phoi toan bo flow.

Graph (fan-out/fan-in):
    route -> vision  -> knowledge -> merge -> qa_agent
          -> rag_retrieve         ->

    route xong thi vision va rag_retrieve chay song song.
    merge doi CA HAI nhanh xong roi moi chay qa_agent.

State (OrchestratorState) chay xuyen suot graph.

Moi node = 1 HTTP call den service tuong ung:
    route_node        -> POST router:8001/route
    vision_node       -> POST vision:8002/analyze/{modality}
    knowledge_node    -> POST knowledge:8003/map
    rag_node          -> FAISS retrieve (local, khong HTTP)
    merge_node        -> kiem tra state, short-circuit neu co loi
    qa_agent_node     -> LLM generate 3-tier report

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
    SpatialDerived, UnifiedOutput, ReportOutput, Tier1Structured
)


# Reducer cho LangGraph state

def _keep_last(a, b):
    """Reducer: keep the latest non-None value. Used for all Optional fields."""
    return b if b is not None else a


def _merge_list(a, b):
    """Reducer: keep the latest non-empty list. Used for rag_chunks."""
    return b if b else a



class OrchestratorState(TypedDict):
    # Input - never written by nodes after init, but Annotated is harmless
    image_bytes:    bytes
    question:       str
    image_id:       str
    modality_hint:  Optional[str]
    organ_hint:     Optional[str]

    # Layer 1 output
    routing:        Annotated[Optional[dict], _keep_last]

    # Layer 2 output
    model_output:   Annotated[Optional[dict], _keep_last]

    # Layer 3 output
    knowledge:      Annotated[Optional[dict], _keep_last]
    spatial:        Annotated[Optional[dict], _keep_last]

    # RAG context -- tu nhanh song song voi vision
    rag_chunks:     Annotated[list, _merge_list]

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
    """Phien ban async cua _post_multipart -- dung cho async node."""
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


def _build_report_prompt(unified: dict, question: str, rag_chunks: list) -> str:
    """Build prompt cho LLM tu UnifiedOutput + user question + RAG context."""
    rag_context = (
        "\n\n".join(rag_chunks) if rag_chunks
        else "No additional clinical guidelines retrieved."
    )
    km = unified.get("knowledge_mapped", {})
    sd = unified.get("spatial_derived", {})
    mo = unified.get("model_output", {})

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

### Clinical Coverage Note
{unified.get('coverage_note', '')}

### Retrieved Clinical Guidelines
{rag_context}

---

Please generate:

**TIER 2 -- Radiological Description** (2-3 sentences, professional radiological language):
Describe the findings using the spatial and classification data above.

**TIER 3 -- Diagnostic Suggestion** (2-3 sentences, include follow-up recommendation):
Based on the findings and clinical guidelines, suggest next steps. Do NOT make a definitive diagnosis.
Start with: "AI-assisted suggestion (must be confirmed by radiologist):"

Answer the user's question: "{question}"
"""
    return prompt.strip()


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
    Async node -- goi /route voi hint neu co.
    hint_conflict/hint_resolution_note/final_decision_source duoc copy vao state
    de report_node co the dua vao tier_1_structured.
    """
    async def route_node(state: OrchestratorState) -> dict:
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
                return {
                    "routing": result,
                    "error": (
                        f"Image rejected as out-of-distribution "
                        f"(confidence {result.get('confidence', 0):.0%} < threshold). "
                        "Please upload a breast or thyroid ultrasound image."
                    ),
                }

            return {"routing": result}

        except Exception as e:
            return {"error": f"[route_node] {e}"}
    return route_node


def make_vision_node(vision_url: str, registry=None):
    """
    registry: ModuleRegistry instance -- doc that tu module_registry.yaml.
    Neu None, fallback ve dict hardcode (chi dung cho test/dev).
    """
    async def vision_node(state: OrchestratorState) -> dict:
        if state.get("error"):
            return {}
        routing = state.get("routing", {})
        organ = routing.get("organ", "breast")
        module_key = routing.get("module_key", "us_breast")

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
            return {"model_output": result}
        except Exception as e:
            return {"error": f"[vision_node] {e}"}
    return vision_node


def make_knowledge_node(knowledge_url: str):
    async def knowledge_node(state: OrchestratorState) -> dict:
        if state.get("error"):
            return {}

        routing = state.get("routing", {})
        mo = state.get("model_output", {})

        payload = {
            "modality":         routing.get("modality", "ultrasound"),
            "organ":            routing.get("organ", "breast"),
            "top_label":        mo.get("top_label", "benign"),
            "confidence":       mo.get("confidence", 0.0),
            "all_scores":       mo.get("all_scores", {}),
            "mask_png_base64":  mo.get("mask_png_base64", ""),
            "original_size":    list(mo.get("original_size", [512, 512])),
            "pixel_spacing_mm": 0.1,
        }

        try:
            result = await _post_json_async(f"{knowledge_url}/map", payload)
            return {
                "knowledge": result.get("knowledge_mapped", {}),
                "spatial":   result.get("spatial_derived", {}),
            }
        except Exception as e:
            return {"error": f"[knowledge_node] {e}"}
    return knowledge_node


def make_rag_node(rag_store):
    """
    RAG node chay song song voi nhanh vision.

    Query dung question cua user vi vision chua chay xong luc nay.
    """
    async def rag_node(state: OrchestratorState) -> OrchestratorState:
        if state.get("error"):
            return state
        question = state.get("question", "")
        try:
            # Wrap sync retrieve() vao thread de tranh block event loop
            chunks = await asyncio.to_thread(rag_store.retrieve, question, 3) \
                if rag_store and rag_store.is_ready() else []
            state["rag_chunks"] = chunks
        except Exception as e:
            print(f"[rag_node] Retrieve error: {e}")
            state["rag_chunks"] = []
        return state
    return rag_node


def make_merge_node():
    """
    Doi ca hai nhanh (knowledge va rag) xong roi moi cho qua qa_agent.
    """
    async def merge_node(state: OrchestratorState) -> OrchestratorState:
        # Neu co loi tu bat ky nhanh nao, dung lai, khong goi LLM
        if state.get("error"):
            return state
        if not state.get("knowledge"):
            state["error"] = "[merge_node] knowledge output chua co -- nhanh vision/knowledge bi loi."
            return state
        return state
    return merge_node


def make_qa_agent_node(llm_client, rag_store):
    """Sinh bao cao 3 tang tu ket qua vision, knowledge va RAG."""
    #
    async def qa_agent_node(state: OrchestratorState) -> OrchestratorState:
        if state.get("error"):
            return state

        routing = state.get("routing", {})
        mo      = state.get("model_output", {})
        km      = state.get("knowledge", {})
        sd      = state.get("spatial", {})
        rag_chunks = state.get("rag_chunks", [])

        unified = {
            "modality":         routing.get("modality", "ultrasound"),
            "organ":            routing.get("organ", "breast"),
            "image_id":         state["image_id"],
            "model_output":     mo,
            "knowledge_mapped": km,
            "spatial_derived":  sd,
            "coverage_note":    "Model trained on BUSI dataset (benign/malignant/normal only).",
        }

        prompt = _build_report_prompt(unified, state["question"], rag_chunks)
        try:
            llm_response = await asyncio.to_thread(
                llm_client.generate, prompt, SYSTEM_PROMPT
            )
        except Exception as e:
            llm_response = f"[LLM unavailable: {e}]"

        tier2, tier3 = _parse_tiers(llm_response)

        tier1 = {
            "modality":        routing.get("modality", "ultrasound"),
            "organ":           routing.get("organ", "breast"),
            "label":           mo.get("top_label", "unknown"),
            "confidence":      mo.get("confidence", 0.0),
            "risk_category":   km.get("risk_category", "unknown"),
            "severity":        km.get("severity", "unknown"),
            "severity_level":  km.get("severity_level", 1),
            "icd10_hint":      km.get("icd10_hint", "unknown"),
            "location_quadrant": sd.get("location_quadrant", "unknown"),
            "bbox":            sd.get("bbox", [0, 0, 0, 0]),
            "area_cm2":        sd.get("area_cm2", 0.0),
            "aspect_ratio":    sd.get("aspect_ratio", 1.0),
            "circularity":     sd.get("circularity", 1.0),
            "confidence_calibration_note": km.get("confidence_calibration_note"),
            # Copy hint fields tu routing vao tier1 de UI hien thi banner
            "hint_conflict":          routing.get("hint_conflict", False),
            "hint_resolution_note":   routing.get("hint_resolution_note"),
        }

        state["report"] = {
            "image_id":                        state["image_id"],
            "tier_1_structured":               tier1,
            "tier_2_radiological_description": tier2,
            "tier_3_diagnostic_suggestion":    tier3,
            "rag_sources": [f"chunk_{i}" for i in range(len(rag_chunks))],
            "rag_disabled_warning": (
                None if (rag_store and rag_store.is_ready()) else
                "RAG context not available -- report generated from classification "
                "label and hardcoded mapping only, without clinical guideline retrieval."
            ),
            # rag_chunks duoc embed trong report de main.py doc cho /chat cache
            "_rag_chunks_internal": rag_chunks,
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

    route_node     = make_route_node(router_url)
    vision_node    = make_vision_node(vision_url, registry=registry)
    knowledge_node = make_knowledge_node(knowledge_url)
    rag_node       = make_rag_node(rag_store)
    merge_node     = make_merge_node()
    qa_agent_node  = make_qa_agent_node(llm_client, rag_store)

    if LANGGRAPH_AVAILABLE:
        g = StateGraph(OrchestratorState)
        g.add_node("route",        route_node)
        g.add_node("vision",       vision_node)
        g.add_node("knowledge",    knowledge_node)
        g.add_node("rag_retrieve", rag_node)
        g.add_node("merge",        merge_node)
        g.add_node("qa_agent",     qa_agent_node)

        g.set_entry_point("route")
        # Fan-out: vision va rag chay song song sau route
        g.add_edge("route",        "vision")
        g.add_edge("route",        "rag_retrieve")
        g.add_edge("vision",       "knowledge")
        # Fan-in: merge chi chay sau khi CA HAI nhanh knowledge va rag xong
        g.add_edge(["knowledge", "rag_retrieve"], "merge")
        g.add_edge("merge",        "qa_agent")
        g.add_edge("qa_agent",     END)

        return g.compile()
    else:
        return AsyncSequentialFallback(
            image_nodes=[route_node, vision_node, knowledge_node],
            rag_node=rag_node,
            merge_node=merge_node,
            qa_agent_node=qa_agent_node,
        )


class AsyncSequentialFallback:
    """
    Fallback khi langgraph khong install.
    Mo phong fan-out bang asyncio.gather cho 2 nhanh (vision + rag),
    dam bao test pass ca khi langgraph thieu trong moi truong CI nhe.
    """

    def __init__(self, image_nodes, rag_node, merge_node, qa_agent_node):
        self.image_nodes = image_nodes      # [route, vision, knowledge]
        self.rag_node = rag_node
        self.merge_node = merge_node
        self.qa_agent_node = qa_agent_node

    async def ainvoke(self, state: dict) -> dict:
        # Chay route truoc, roi fan-out vision va rag
        state = await self.image_nodes[0](state)
        if state.get("error"):
            return state

        import copy
        state_for_rag = copy.copy(state)

        async def run_image_branch(s):
            for node in self.image_nodes[1:]:
                s = await node(s)
            return s

        state, state_for_rag = await asyncio.gather(
            run_image_branch(state),
            self.rag_node(state_for_rag),
        )

        state["rag_chunks"] = state_for_rag.get("rag_chunks", [])

        state = await self.merge_node(state)
        if state.get("error"):
            return state
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

    Dung cho FastAPI async handler -- khong goi asyncio.run() long trong
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
        "rag_chunks":    [],
        "report":        None,
        "error":         None,
    }

    if hasattr(graph, "ainvoke"):
        final_state = await graph.ainvoke(initial_state)
    else:
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
    Phien ban sync cua run_pipeline_async -- chi dung khi chay ngoai
    FastAPI (vd. script thu cong). Trong FastAPI, dung run_pipeline_async.
    """
    return asyncio.run(run_pipeline_async(
        graph, image_bytes, question, image_id, modality_hint, organ_hint
    ))