"""
services/orchestrator/main.py
==============================
FastAPI Orchestrator -- Layer 4 Gateway | port 8000

Endpoints:
    POST /analyze  -- nhan anh + question -> ReportOutput
    POST /chat     -- hoi them ve ket qua da phan tich (multi-turn)
    GET  /health

/chat yeu cau image_id da duoc /analyze truoc do. Context duoc cache
trong bo nho process (in-memory dict voi TTL). Gioi han nay phai duoc
ghi nhan ro: khong hoat dong voi orchestrator scale > 1 replica -- can
Redis/DB that cho multi-instance deploy.
"""

import os
import sys
import time
import asyncio
from contextlib import asynccontextmanager
from typing import Optional, List

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

sys.path.insert(0, "/app")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from shared.schemas import ReportOutput, Tier1Structured
from services.orchestrator.llm_client import get_llm_client
from services.orchestrator.rag.faiss_store import FAISSStore
from services.orchestrator.graph import (
    build_graph, run_pipeline_async, _build_chat_prompt,
    CHAT_SYSTEM_PROMPT,
)
from services.orchestrator.module_registry import load_module_registry, ModuleRegistryError

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

# TTL cache context (default 1 gio)
_CONTEXT_TTL_SECONDS = int(os.getenv("CHAT_CONTEXT_TTL", "3600"))

# Cache context theo image_id, chi dung duoc voi single-instance deploy
_context_cache: dict = {}

_graph = None
_llm_client = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _graph, _llm_client
    print("[orchestrator] Initializing pipeline...")

    _llm_client = get_llm_client()
    rag_store   = FAISSStore(index_path=FAISS_INDEX_PATH)

    if not rag_store.is_ready():
        print("[orchestrator] RAG index not ready -- LLM se chay khong co clinical context.")

    _graph = build_graph(SERVICES_CFG, _llm_client, rag_store, registry=_registry)
    print(f"[orchestrator] Graph ready -- services: {SERVICES_CFG}")
    print(
        f"[orchestrator] Vision modalities: "
        f"{list(_registry.vision_modalities.keys())}"
    )
    yield
    _graph = None


app = FastAPI(
    title="Med-Platform Orchestrator",
    description="Layer 4 -- LangGraph gateway. Dieu phoi router -> vision -> knowledge -> LLM.",
    version="1.0.0",
    lifespan=lifespan,
)


# Schema cho /chat endpoint

class ChatMessage(BaseModel):
    role: str   # 'user' | 'assistant'
    content: str


class ChatRequest(BaseModel):
    image_id: str
    message: str
    history: List[ChatMessage] = []


class ChatResponse(BaseModel):
    reply: str
    image_id: str


# Quan ly context cache theo image_id

def _save_context(image_id: str, report_dict: dict, rag_chunks: list):
    """Luu context vao cache sau khi /analyze thanh cong."""
    _context_cache[image_id] = {
        "context": {
            "image_id": image_id,
            "report":   report_dict,
        },
        "rag_chunks": rag_chunks,
        "ts": time.time(),
    }


def _get_context(image_id: str) -> dict:
    """
    Lay context tu cache theo image_id.
    Raise HTTPException 404 neu chua analyze hoac da het TTL.
    """
    entry = _context_cache.get(image_id)
    if entry is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"image_id '{image_id}' chua duoc phan tich hoac khong ton tai. "
                "Vui long goi /analyze truoc."
            ),
        )
    age = time.time() - entry["ts"]
    if age > _CONTEXT_TTL_SECONDS:
        del _context_cache[image_id]
        raise HTTPException(
            status_code=404,
            detail=(
                f"Context cho image_id '{image_id}' da het han ({_CONTEXT_TTL_SECONDS}s). "
                "Vui long upload va phan tich lai anh."
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


@app.post("/analyze", response_model=ReportOutput)
async def analyze(
    image: UploadFile = File(..., description="Anh ultrasound PNG/JPG"),
    question: str = Form(
        default="What are the findings in this ultrasound image?",
        description="Cau hoi lam sang",
    ),
    image_id: str = Form(default=None, description="Optional custom image ID"),
    modality_hint: Optional[str] = Form(default=None, description="'breast' | 'thyroid' | None"),
    organ_hint: Optional[str] = Form(default=None, description="'breast' | 'thyroid' | None"),
):
    """
    Entry point chinh cho client / Gradio UI.

    modality_hint va organ_hint duoc forward xuong router -- router ket hop
    voi router_probs theo trong so de ra quyet dinh cuoi. Hint conflict
    duoc ghi ro trong Tier1Structured.hint_conflict va hint_resolution_note.
    """
    if _graph is None:
        raise HTTPException(status_code=503, detail="Orchestrator chua san sang.")

    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="File anh rong.")

    try:
        report_dict = await run_pipeline_async(
            graph=_graph,
            image_bytes=image_bytes,
            question=question,
            image_id=image_id,
            modality_hint=modality_hint,
            organ_hint=organ_hint,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")

    # Luu context de /chat su dung, khong chay lai pipeline
    _save_context(
        report_dict["image_id"],
        report_dict,
        report_dict.get("_rag_chunks_internal", []),
    )

    t1 = report_dict["tier_1_structured"]
    return ReportOutput(
        image_id=report_dict["image_id"],
        tier_1_structured=Tier1Structured(**t1),
        tier_2_radiological_description=report_dict["tier_2_radiological_description"],
        tier_3_diagnostic_suggestion=report_dict["tier_3_diagnostic_suggestion"],
        rag_sources=report_dict.get("rag_sources", []),
        rag_disabled_warning=report_dict.get("rag_disabled_warning"),
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    Multi-turn chatbot dua tren context da phan tich.

    Khong nhan lai anh -- tai su dung context tu _context_cache theo image_id.
    Vision/router/knowledge KHONG duoc goi lai -- chi goi LLM voi context + history.

    """
    if _llm_client is None:
        raise HTTPException(status_code=503, detail="LLM client chua san sang.")

    entry = _get_context(req.image_id)
    unified_context = entry["context"]
    rag_chunks = entry["rag_chunks"]

    history_dicts = [{"role": m.role, "content": m.content} for m in req.history]

    prompt = _build_chat_prompt(
        unified_context=unified_context,
        rag_chunks=rag_chunks,
        history=history_dicts,
        message=req.message,
    )

    try:
        if hasattr(_llm_client, "chat"):
            messages = history_dicts + [{"role": "user", "content": req.message}]
            reply = await asyncio.to_thread(
                _llm_client.chat, messages, CHAT_SYSTEM_PROMPT
            )
        else:
            # Fallback khi khong co phuong thuc chat()
            reply = await asyncio.to_thread(
                _llm_client.generate, prompt, CHAT_SYSTEM_PROMPT
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {str(e)}")

    return ChatResponse(reply=reply, image_id=req.image_id)