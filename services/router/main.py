"""
services/router/main.py

FastAPI Router Service -- Layer 1 | port 8001

Endpoints:
    POST /route    -- classify image -> RoutingResult
    GET  /health
    GET  /metrics  -- Prometheus metrics

Request: multipart/form-data
    image:          UploadFile
    modality_hint:  str (optional) -- 'breast' | 'thyroid'
    organ_hint:     str (optional) -- 'breast' | 'thyroid'

Response: RoutingResult schema (JSON)
"""

import os
import sys
import time
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import PlainTextResponse

sys.path.insert(0, "/app")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from shared.schemas import RoutingResult
from shared.telemetry import setup_tracing, get_tracer
from shared.image_validation import (
    check_upload_size, check_image_dimensions, ImageValidationError
)
from services.router.model import (
    load_router, run_routing, HINT_ORGAN_TO_MODULE_KEY
)

logger = logging.getLogger(__name__)

try:
    from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
    PROM_AVAILABLE = True
    _route_latency = Histogram(
        "router_route_duration_seconds",
        "Latency of the /route endpoint",
        buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.0],
    )
    _route_counter = Counter(
        "router_route_requests_total",
        "Total number of /route requests",
        ["module_key", "is_ood", "status"],
    )
except ImportError:
    PROM_AVAILABLE = False

CHECKPOINT_PATH = os.getenv(
    "ROUTER_CHECKPOINT",
    "models/checkpoints/router_effnet_b0.pth"
)
OOD_THRESHOLD = float(os.getenv("OOD_THRESHOLD", "0.6"))
DEVICE = os.getenv("DEVICE", None)

_VALID_ORGAN_HINTS = set(HINT_ORGAN_TO_MODULE_KEY.keys())

_model     = None
_transform = None
_device    = None
_degraded  = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _transform, _device, _degraded
    setup_tracing("router", app=app)
    _model, _transform, _device, _degraded = load_router(CHECKPOINT_PATH, DEVICE)
    print(f"[router] Ready on {_device} -- degraded={_degraded}")
    yield
    _model = None


app = FastAPI(
    title="Router Service",
    description="Layer 1 -- Modality classifier (EfficientNet-B0)",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "checkpoint": CHECKPOINT_PATH,
        "ood_threshold": OOD_THRESHOLD,
        "degraded": _degraded,
    }


@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    if not PROM_AVAILABLE:
        return PlainTextResponse("# prometheus_client not installed\n", status_code=200)
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/route", response_model=RoutingResult)
async def route_image(
    image: UploadFile = File(..., description="Image to classify the modality of"),
    modality_hint: Optional[str] = Form(default=None),
    organ_hint: Optional[str] = Form(default=None),
):
    """
    Takes an image + optional hint -> returns a RoutingResult.

    Valid organ_hint: 'breast' | 'thyroid'. Returns 400 for any other value.
    The hint is completely ignored when is_ood=True.
    """
    if _model is None:
        raise HTTPException(status_code=503, detail="Router model not loaded.")

    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty image file.")
    try:
        check_upload_size(image_bytes)
    except ImageValidationError as e:
        raise HTTPException(status_code=413, detail=str(e))

    hint_key = None
    raw_hint = organ_hint or modality_hint
    if raw_hint is not None:
        raw_hint = raw_hint.strip().lower()
        if raw_hint not in _VALID_ORGAN_HINTS:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"organ_hint '{raw_hint}' is not valid. "
                    f"Accepted values: {sorted(_VALID_ORGAN_HINTS)}"
                ),
            )
        hint_key = HINT_ORGAN_TO_MODULE_KEY[raw_hint]

    t_start = time.perf_counter()
    with get_tracer().start_as_current_span("router.classify") as span:
        try:
            result = run_routing(
                model=_model,
                transform=_transform,
                device=_device,
                image_bytes=image_bytes,
                ood_threshold=OOD_THRESHOLD,
                degraded=_degraded,
                hint_module_key=hint_key,
            )
            span.set_attribute("router.module_key",  result.get("module_key", ""))
            span.set_attribute("router.is_ood",      result.get("is_ood", False))
            span.set_attribute("router.confidence",  result.get("confidence", 0.0))
            if PROM_AVAILABLE:
                _route_counter.labels(
                    module_key=result.get("module_key", "unknown"),
                    is_ood=str(result.get("is_ood", False)),
                    status="ok",
                ).inc()
                _route_latency.observe(time.perf_counter() - t_start)
        except ValueError as e:
            if PROM_AVAILABLE:
                _route_counter.labels(module_key="unknown", is_ood="false", status="error").inc()
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            span.record_exception(e)
            if PROM_AVAILABLE:
                _route_counter.labels(module_key="unknown", is_ood="false", status="error").inc()
            logger.exception("Routing failed")
            raise HTTPException(status_code=500, detail="Internal error during routing. Check server logs.")

    return RoutingResult(**result)
