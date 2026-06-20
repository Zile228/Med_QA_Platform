"""
services/vision/main.py
=========================
FastAPI Vision Service - Layer 2 | port 8002

Endpoints:
    POST /analyze/us_breast   - breast ultrasound inference
    POST /analyze/us_thyroid  - thyroid ultrasound inference (TN3K)
    GET  /health
    GET  /metrics             - Prometheus metrics

Request: multipart/form-data
    image: UploadFile  - anh PNG/JPG
    organ: str         - 'breast' | 'thyroid'

Response: ModelOutput schema (JSON)
"""

import os
import sys
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import PlainTextResponse

sys.path.insert(0, "/app")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from shared.schemas import ModelOutput
from shared.telemetry import setup_tracing, get_tracer
from services.vision.us_breast.model  import load_model as load_breast_model,  run_inference as run_breast_inference
from services.vision.us_thyroid.model import load_model as load_thyroid_model, run_inference as run_thyroid_inference

try:
    from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
    PROM_AVAILABLE = True
    _infer_latency = Histogram(
        "vision_inference_duration_seconds",
        "Latency cua inference theo organ",
        ["organ"],
        buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
    )
    _infer_counter = Counter(
        "vision_inference_requests_total",
        "Tong so request inference",
        ["organ", "label", "status"],
    )
except ImportError:
    PROM_AVAILABLE = False

BUSI_CHECKPOINT    = os.getenv("BUSI_CHECKPOINT",    "models/checkpoints/mtl_effnet_fc_conv.pt")
THYROID_CHECKPOINT = os.getenv("THYROID_CHECKPOINT", "models/checkpoints/mtl_effnet_fc_conv_thyroid.pt")
DEVICE             = os.getenv("DEVICE", None)

_breast_model  = None
_breast_cfg    = None
_thyroid_model = None
_thyroid_cfg   = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _breast_model, _breast_cfg, _thyroid_model, _thyroid_cfg

    setup_tracing("vision", app=app)

    try:
        _breast_model, _breast_cfg = load_breast_model(BUSI_CHECKPOINT, DEVICE)
        print(f"[vision] Breast model loaded OK - device: {_breast_cfg.DEVICE}")
    except FileNotFoundError as e:
        print(f"[vision] WARNING (breast): {e}")
        print("[vision] /analyze/us_breast se return 503 cho den khi co checkpoint.")

    try:
        _thyroid_model, _thyroid_cfg = load_thyroid_model(THYROID_CHECKPOINT, DEVICE)
        print(f"[vision] Thyroid model loaded OK - device: {_thyroid_cfg.DEVICE}")
    except FileNotFoundError as e:
        print(f"[vision] WARNING (thyroid): {e}")
        print("[vision] /analyze/us_thyroid se return 503 cho den khi co checkpoint.")

    yield

    _breast_model  = None
    _thyroid_model = None


app = FastAPI(
    title="Vision Service",
    description="Layer 2 - Medical image segmentation + classification (Breast & Thyroid)",
    version="2.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "breast_model_loaded":  _breast_model  is not None,
        "thyroid_model_loaded": _thyroid_model is not None,
        "breast_checkpoint":    BUSI_CHECKPOINT,
        "thyroid_checkpoint":   THYROID_CHECKPOINT,
    }


@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    if not PROM_AVAILABLE:
        return PlainTextResponse("# prometheus_client chua install\n", status_code=200)
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)


async def _read_image(image: UploadFile) -> bytes:
    """Doc va validate anh upload."""
    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="File anh rong.")
    content_type = image.content_type or ""
    if content_type and not content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"File khong phai anh: {content_type}",
        )
    return image_bytes


@app.post("/analyze/us_breast", response_model=ModelOutput)
async def analyze_us_breast(
    image: UploadFile = File(..., description="Anh ultrasound breast (PNG/JPG)"),
    organ: str = Form(default="breast"),
):
    """
    Inference pipeline cho Breast Ultrasound (BUSI dataset).
    Tra ve ModelOutput co bottleneck_features de orchestrator dua vao LLM prompt.
    """
    if _breast_model is None:
        raise HTTPException(
            status_code=503,
            detail="Breast model chua load. Dat checkpoint vao models/checkpoints/ va restart.",
        )
    image_bytes = await _read_image(image)

    t_start = time.perf_counter()
    with get_tracer().start_as_current_span("vision.us_breast") as span:
        try:
            result = run_breast_inference(
                model=_breast_model, cfg=_breast_cfg, image_bytes=image_bytes
            )
            span.set_attribute("vision.organ",      "breast")
            span.set_attribute("vision.top_label",  result.get("top_label", ""))
            span.set_attribute("vision.confidence", result.get("confidence", 0.0))
            if PROM_AVAILABLE:
                _infer_counter.labels(
                    organ="breast", label=result.get("top_label", "unknown"), status="ok"
                ).inc()
                _infer_latency.labels(organ="breast").observe(time.perf_counter() - t_start)
        except Exception as e:
            span.record_exception(e)
            if PROM_AVAILABLE:
                _infer_counter.labels(organ="breast", label="unknown", status="error").inc()
            raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

    return ModelOutput(
        top_label=result["top_label"],
        confidence=result["confidence"],
        all_scores=result["all_scores"],
        mask_png_base64=result["mask_png_base64"],
        bottleneck_features=result["bottleneck_features"],
        original_size=list(result["original_size"]),
    )


@app.post("/analyze/us_thyroid", response_model=ModelOutput)
async def analyze_us_thyroid(
    image: UploadFile = File(..., description="Anh ultrasound thyroid (PNG/JPG)"),
    organ: str = Form(default="thyroid"),
):
    """
    Inference pipeline cho Thyroid Ultrasound (TN3K dataset).
    Cung schema voi us_breast -- orchestrator dung chung 1 code path.
    """
    if _thyroid_model is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Thyroid model chua load. "
                "Chay notebook tn3k_thyroid_train.ipynb de tao checkpoint, "
                "sau do dat vao models/checkpoints/mtl_effnet_fc_conv_thyroid.pt va restart."
            ),
        )
    image_bytes = await _read_image(image)

    t_start = time.perf_counter()
    with get_tracer().start_as_current_span("vision.us_thyroid") as span:
        try:
            result = run_thyroid_inference(
                model=_thyroid_model, cfg=_thyroid_cfg, image_bytes=image_bytes
            )
            span.set_attribute("vision.organ",      "thyroid")
            span.set_attribute("vision.top_label",  result.get("top_label", ""))
            span.set_attribute("vision.confidence", result.get("confidence", 0.0))
            if PROM_AVAILABLE:
                _infer_counter.labels(
                    organ="thyroid", label=result.get("top_label", "unknown"), status="ok"
                ).inc()
                _infer_latency.labels(organ="thyroid").observe(time.perf_counter() - t_start)
        except Exception as e:
            span.record_exception(e)
            if PROM_AVAILABLE:
                _infer_counter.labels(organ="thyroid", label="unknown", status="error").inc()
            raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

    return ModelOutput(
        top_label=result["top_label"],
        confidence=result["confidence"],
        all_scores=result["all_scores"],
        mask_png_base64=result["mask_png_base64"],
        bottleneck_features=result["bottleneck_features"],
        original_size=list(result["original_size"]),
    )


@app.post("/analyze/xray")
async def analyze_xray(image: UploadFile = File(...)):
    """Placeholder - Phase 2 (NIH ChestX-ray14)."""
    raise HTTPException(
        status_code=501,
        detail="X-Ray module chua implement. Roadmap: Phase 2.",
    )
