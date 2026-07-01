"""
services/orchestrator/main.py
==============================
FastAPI Orchestrator -- Layer 4 Gateway | port 8000

Endpoints:
    POST /analyze  -- takes an image + question -> ReportOutput
    POST /chat     -- ask follow-up questions about analyzed results (multi-turn)
    GET  /health
    GET  /metrics  -- Prometheus metrics

/chat requires an image_id that was already passed to /analyze. Context is
cached in process memory (in-memory dict with TTL). This limitation must be
clearly noted: it does not work with orchestrator scaled to > 1 replica --
a real Redis/DB is needed for multi-instance deployment.
"""

import os
import sys
import time
import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Optional, List

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel

sys.path.insert(0, "/app")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from shared.schemas import ReportOutput, Tier1Structured, RagSource, CoTResult
from shared.telemetry import setup_tracing, get_tracer
from shared.image_validation import (
    check_upload_size, ImageValidationError
)
from services.orchestrator.llm_client import get_llm_client
from services.orchestrator.rag.faiss_store import FAISSStore
from services.orchestrator.graph import (
    build_graph, run_pipeline_async, _build_chat_prompt,
    CHAT_SYSTEM_PROMPT,
)
from services.orchestrator.module_registry import load_module_registry, ModuleRegistryError

logger = logging.getLogger(__name__)

try:
    from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
    PROM_AVAILABLE = True
    _analyze_latency = Histogram(
        "orchestrator_analyze_duration_seconds",
        "End-to-end latency of /analyze",
        buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
    )
    _analyze_counter = Counter(
        "orchestrator_analyze_requests_total",
        "Total number of /analyze requests",
        ["organ", "label", "status"],
    )
    _ood_counter = Counter(
        "orchestrator_ood_rejections_total",
        "Number of requests rejected due to OOD",
    )
    _consensus_false_counter = Counter(
        "orchestrator_consensus_false_total",
        "Number of requests with mapper vs CoT disagreement (consensus=false)",
    )
    _icd10_disagreement_counter = Counter(
        "orchestrator_icd10_disagreement_total",
        "Number of requests with differing icd10_hint between mapper vs CoT (icd10_agreement=false)",
    )
    _confidence_histogram = Histogram(
        "vision_confidence_score",
        "Distribution of confidence scores from the vision model",
        buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999, 1.0],
    )
    _cot_parse_failure_counter = Counter(
        "orchestrator_cot_parse_failure_total",
        "Number of requests where CoT JSON parsing failed (severity=undetermined). "
        "High values indicate the LLM is not following the required output schema.",
    )
except ImportError:
    PROM_AVAILABLE = False

MODULE_REGISTRY_PATH = os.getenv("MODULE_REGISTRY_PATH", "module_registry.yaml")
_registry = load_module_registry(MODULE_REGISTRY_PATH)

SERVICES_CFG = {
    "router_url":    _registry.router_url,
    "vision_url":    _registry.vision_url,
    "knowledge_url": _registry.knowledge_url,
}

FAISS_INDEX_PATH = os.getenv(
    "FAISS_INDEX_PATH",
    "services/orchestrator/rag/vectordb/index.faiss"
)

_CONTEXT_TTL_SECONDS = int(os.getenv("CHAT_CONTEXT_TTL", "3600"))

_context_cache: dict = {}

_graph       = None
_llm_client  = None
_rag_store   = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _graph, _llm_client, _rag_store
    setup_tracing("orchestrator", app=app)
    print("[orchestrator] Initializing pipeline...")

    _llm_client = get_llm_client()
    _rag_store  = FAISSStore(index_path=FAISS_INDEX_PATH)

    if not _rag_store.is_ready():
        print("[orchestrator] RAG index not ready -- LLM will run without clinical context.")

    _graph = build_graph(SERVICES_CFG, _llm_client, _rag_store, registry=_registry)
    print(f"[orchestrator] Graph ready -- services: {SERVICES_CFG}")
    print(
        f"[orchestrator] Vision modalities: "
        f"{list(_registry.vision_modalities.keys())}"
    )
    yield
    _graph = None


app = FastAPI(
    title="Med-Platform Orchestrator",
    description="Layer 4 -- LangGraph gateway. Orchestrates router -> vision -> knowledge -> LLM.",
    version="1.0.0",
    lifespan=lifespan,
)


# Schema for the /chat endpoint

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    image_id: str
    message: str
    history: List[ChatMessage] = []


class ChatResponse(BaseModel):
    reply: str
    image_id: str


# Manage the context cache by image_id

def _save_context(image_id: str, report_dict: dict, rag_chunks: list):
    """Save context into the cache after /analyze succeeds."""
    organ = report_dict.get("tier_1_structured", {}).get("organ")
    _context_cache[image_id] = {
        "context": {
            "image_id": image_id,
            "report":   report_dict,
        },
        "rag_chunks": rag_chunks,
        "organ": organ,
        "ts": time.time(),
    }


def _get_context(image_id: str) -> dict:
    """
    Get context from the cache by image_id.
    Raises HTTPException 404 if not analyzed yet or the TTL has expired.
    """
    entry = _context_cache.get(image_id)
    if entry is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"image_id '{image_id}' has not been analyzed or does not exist. "
                "Please call /analyze first."
            ),
        )
    age = time.time() - entry["ts"]
    if age > _CONTEXT_TTL_SECONDS:
        del _context_cache[image_id]
        raise HTTPException(
            status_code=404,
            detail=(
                f"Context for image_id '{image_id}' has expired ({_CONTEXT_TTL_SECONDS}s). "
                "Please upload and analyze the image again."
            ),
        )
    return entry


@app.get("/health")
def health():
    return {
        "status": "ok",
        "graph_ready": _graph is not None,
        "services": SERVICES_CFG,
        "llm_backend": os.getenv("LLM_BACKEND", "ollama"),
        "module_registry_path": MODULE_REGISTRY_PATH,
        "vision_modalities": {
            k: {"endpoint": v.endpoint, "enabled": v.enabled}
            for k, v in _registry.vision_modalities.items()
        },
    }


@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    """Prometheus metrics endpoint."""
    if not PROM_AVAILABLE:
        return PlainTextResponse("# prometheus_client not installed\n", status_code=200)
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/analyze", response_model=ReportOutput)
async def analyze(
    image: UploadFile = File(..., description="Ultrasound image PNG/JPG"),
    image_id: str = Form(default=None, description="Optional custom image ID"),
    modality_hint: Optional[str] = Form(default=None, description="'breast' | 'thyroid' | None"),
    organ_hint: Optional[str] = Form(default=None, description="'breast' | 'thyroid' | None"),
    pixel_spacing_mm: Optional[float] = Form(
        default=None,
        description="Real-world mm per pixel, typically from DICOM metadata. None when unknown.",
    ),
    laterality: Optional[str] = Form(
        default=None,
        description="'left' | 'right' | None. Resolves breast outer/inner labeling.",
    ),
):
    """
    Main entry point for the client / Gradio UI.

    modality_hint and organ_hint are forwarded to the router -- the router
    combines them with router_probs by weight to reach the final decision.
    Hint conflicts are recorded in Tier1Structured.hint_conflict and
    hint_resolution_note.

    pixel_spacing_mm and laterality are forwarded to the spatial node.
    Without them, area_cm2 stays None and breast outer/inner stays
    unresolved, exactly as when omitted today.
    """
    if _graph is None:
        raise HTTPException(status_code=503, detail="Orchestrator not ready.")

    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty image file.")
    try:
        check_upload_size(image_bytes)
    except ImageValidationError as e:
        raise HTTPException(status_code=413, detail=str(e))

    t_start = time.perf_counter()
    with get_tracer().start_as_current_span("orchestrator.analyze") as span:
        span.set_attribute("request.image_id", image_id or "")
        span.set_attribute("request.organ_hint", organ_hint or "")
        try:
            report_dict = await run_pipeline_async(
                graph=_graph,
                image_bytes=image_bytes,
                image_id=image_id,
                modality_hint=modality_hint,
                organ_hint=organ_hint,
                pixel_spacing_mm=pixel_spacing_mm,
                laterality=laterality,
            )
            t1_data = report_dict.get("tier_1_structured", {})
            span.set_attribute("result.organ",      t1_data.get("organ", ""))
            span.set_attribute("result.label",      t1_data.get("label", ""))
            span.set_attribute("result.severity",   t1_data.get("severity", ""))
            span.set_attribute("result.consensus",  str(report_dict.get("consensus")))
            span.set_attribute("result.icd10_agreement", str(report_dict.get("icd10_agreement")))
        except RuntimeError as e:
            err_str = str(e)
            span.record_exception(e)
            if PROM_AVAILABLE and "out-of-distribution" in err_str.lower():
                _ood_counter.inc()
            if PROM_AVAILABLE:
                _analyze_counter.labels(organ="unknown", label="unknown", status="error").inc()
            raise HTTPException(status_code=422, detail=err_str)
        except Exception as e:
            span.record_exception(e)
            if PROM_AVAILABLE:
                _analyze_counter.labels(organ="unknown", label="unknown", status="error").inc()
            logger.exception("Pipeline failed")
            raise HTTPException(status_code=500, detail="Internal error during pipeline execution. Check server logs.")

    if PROM_AVAILABLE:
        elapsed = time.perf_counter() - t_start
        _analyze_latency.observe(elapsed)
        t1_data = report_dict.get("tier_1_structured", {})
        organ_lbl = t1_data.get("organ", "unknown")
        label_lbl = t1_data.get("label", "unknown")
        conf = t1_data.get("confidence", 0.0)
        _analyze_counter.labels(organ=organ_lbl, label=label_lbl, status="ok").inc()
        _confidence_histogram.observe(conf)
        if report_dict.get("consensus") is False:
            _consensus_false_counter.inc()
        if report_dict.get("icd10_agreement") is False:
            _icd10_disagreement_counter.inc()

    _save_context(
        report_dict["image_id"],
        report_dict,
        report_dict.get("_rag_chunks_internal", []),
    )

    t1 = report_dict["tier_1_structured"]

    # Convert rag_sources from a list of dicts to a list of RagSource.
    # rag_chunks (text noi dung) va raw_sources (file/page) duoc build tu
    # cung 1 list reranked trong graph.py (xem rag_chunks = [m["chunk"] for m
    # in reranked]), nen cung do dai, cung thu tu -- an toan de zip theo index.
    raw_sources = report_dict.get("rag_sources", [])
    rag_chunks_internal = report_dict.get("_rag_chunks_internal", [])
    rag_sources = []
    for i, src in enumerate(raw_sources):
        if isinstance(src, dict):
            chunk_text = rag_chunks_internal[i] if i < len(rag_chunks_internal) else None
            rag_sources.append(RagSource(
                file=src.get("file", "unknown"),
                page=src.get("page", 0),
                text=chunk_text,
            ))

    # Convert cot_result from dict to CoTResult if present
    cot_raw = report_dict.get("cot_result")
    cot_result = None
    if cot_raw and isinstance(cot_raw, dict) and cot_raw.get("severity") != "undetermined":
        try:
            cot_result = CoTResult(**cot_raw)
        except Exception:
            cot_result = None
    elif cot_raw and isinstance(cot_raw, dict) and cot_raw.get("severity") == "undetermined":
        if PROM_AVAILABLE:
            _cot_parse_failure_counter.inc()

    try:
        tier1_obj = Tier1Structured(**t1)
    except Exception as e:
        logger.exception("Tier1Structured validation failed -- missing field in pipeline output")
        raise HTTPException(
            status_code=500,
            detail="Internal error building report. Check server logs.",
        )

    return ReportOutput(
        image_id=report_dict["image_id"],
        tier_1_structured=tier1_obj,
        tier_2_radiological_description=report_dict["tier_2_radiological_description"],
        tier_3_diagnostic_suggestion=report_dict["tier_3_diagnostic_suggestion"],
        rag_sources=rag_sources,
        rag_disabled_warning=report_dict.get("rag_disabled_warning"),
        mapper_result=report_dict.get("mapper_result"),
        cot_result=cot_result,
        consensus=report_dict.get("consensus"),
        icd10_agreement=report_dict.get("icd10_agreement"),
        hard_conflict=report_dict.get("hard_conflict"),
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    Multi-turn chatbot built on already-analyzed context.

    Does not take the image again -- reuses context from _context_cache by
    image_id. Vision/router/knowledge are NOT called again -- only RAG is
    re-queried with the follow-up question text, then the LLM is called
    with context + history.

    rag_chunks saved at /analyze time stay relevant to the original finding,
    so they are kept as background context. Chunks retrieved for this
    specific follow-up question are placed first, since they are most
    likely to answer what the user is actually asking now.
    """
    if _llm_client is None:
        raise HTTPException(status_code=503, detail="LLM client not ready.")

    entry = _get_context(req.image_id)
    unified_context = entry["context"]
    saved_rag_chunks = entry["rag_chunks"]
    organ = entry.get("organ")

    rag_chunks = saved_rag_chunks
    if _rag_store and _rag_store.is_ready():
        try:
            follow_up_meta = await asyncio.to_thread(
                _rag_store.retrieve_with_meta, req.message, 3, organ
            )
            follow_up_chunks = [m["chunk"] for m in follow_up_meta]
            seen = set(follow_up_chunks)
            rag_chunks = follow_up_chunks + [c for c in saved_rag_chunks if c not in seen]
        except Exception as e:
            logger.exception("Chat follow-up RAG retrieve failed -- falling back to saved context")

    history_dicts = [{"role": m.role, "content": m.content} for m in req.history]

    prompt = _build_chat_prompt(
        unified_context=unified_context,
        rag_chunks=rag_chunks,
        history=history_dicts,
        message=req.message,
    )

    try:
        reply = await asyncio.to_thread(
            _llm_client.generate, prompt, CHAT_SYSTEM_PROMPT
        )
    except Exception as e:
        logger.exception("Chat LLM call failed")
        raise HTTPException(status_code=500, detail="Internal error during chat. Check server logs.")

    return ChatResponse(reply=reply, image_id=req.image_id)