"""
services/knowledge/main.py
===========================
FastAPI Knowledge Service - Layer 3 | port 8003

Endpoints:
    POST /map     - ModelOutput + routing info -> KnowledgeMapped + SpatialDerived
    GET  /health
    GET  /metrics - Prometheus metrics

Request body (JSON):
    {
        "modality":            "ultrasound",
        "organ":               "breast",
        "top_label":           "malignant",
        "confidence":          0.87,
        "all_scores":          {"benign": 0.10, "malignant": 0.87, "normal": 0.03},
        "mask_png_base64":     "iVBORw0KGgoAAAANS...",
        "original_size":       [512, 512],
        "pixel_spacing_mm":    0.1,
        "bottleneck_features": {...}
    }

Response:
    {
        "knowledge_mapped":    KnowledgeMapped,
        "spatial_derived":     SpatialDerived,
        "bottleneck_features": {...}
    }

bottleneck_features is received from the orchestrator and forwarded straight
through the response (pass-through). The knowledge service does NOT process
this field -- it only preserves it so the orchestrator can use it in the
LLM prompt without calling vision again.
"""

import os
import sys
import time
import logging

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field
from typing import Optional

sys.path.insert(0, "/app")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from shared.schemas import KnowledgeMapped, SpatialDerived
from services.knowledge.mapper import map_knowledge, derive_spatial

logger = logging.getLogger(__name__)

try:
    from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
    PROM_AVAILABLE = True
    _map_latency = Histogram(
        "knowledge_map_duration_seconds",
        "Latency of the /map endpoint",
        buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0],
    )
    _map_counter = Counter(
        "knowledge_map_requests_total",
        "Total number of /map requests",
        ["organ", "label", "status"],
    )
except ImportError:
    PROM_AVAILABLE = False


# Request and response models

class MapRequest(BaseModel):
    modality:            str   = Field(..., example="ultrasound")
    organ:               str   = Field(..., example="breast")
    top_label:           str   = Field(..., example="malignant")
    confidence:          float = Field(..., example=0.87)
    all_scores:          dict  = Field(..., example={"benign": 0.10, "malignant": 0.87, "normal": 0.03})
    mask_png_base64:     str   = Field(..., example="iVBORw0KGgoAAAANSUhEUgAA...")
    original_size:       list  = Field(..., example=[512, 512])
    pixel_spacing_mm:    float = Field(default=0.1)
    bottleneck_features: Optional[dict] = Field(
        default=None,
        description="Pass-through from vision -- not processed, just forwarded in the response",
    )


class MapResponse(BaseModel):
    knowledge_mapped:    KnowledgeMapped
    spatial_derived:     SpatialDerived
    bottleneck_features: dict = Field(
        default_factory=dict,
        description="Pass-through from vision -- the orchestrator reads it to use in the LLM prompt",
    )


app = FastAPI(
    title="Knowledge Service",
    description="Layer 3 - Clinical severity mapping + spatial feature derivation",
    version="1.0.0",
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    """Prometheus metrics endpoint."""
    if not PROM_AVAILABLE:
        return PlainTextResponse("# prometheus_client not installed\n", status_code=200)
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/map", response_model=MapResponse)
def map_endpoint(req: MapRequest):
    """
    Takes ModelOutput fields -> returns KnowledgeMapped + SpatialDerived.

    Not stateful - each request is independent, no model loading needed.
    Rule-based: auditable by a clinician without needing to read complex code.
    bottleneck_features is received but not processed -- just passed through.
    """
    valid_labels = {"benign", "malignant", "normal"}
    if req.top_label.lower() not in valid_labels:
        raise HTTPException(
            status_code=400,
            detail=f"top_label must be one of: {valid_labels}. Received: '{req.top_label}'",
        )

    valid_organs = {"breast", "thyroid"}
    if req.organ.lower() not in valid_organs:
        raise HTTPException(
            status_code=400,
            detail=f"organ must be one of: {valid_organs}. Received: '{req.organ}'",
        )

    if len(req.original_size) != 2:
        raise HTTPException(status_code=400, detail="original_size must be [H, W].")

    t_start = time.perf_counter()
    try:
        km_dict = map_knowledge(
            modality=req.modality,
            organ=req.organ,
            top_label=req.top_label,
            confidence=req.confidence,
            all_scores=req.all_scores,
        )
        sd_dict = derive_spatial(
            mask_png_base64=req.mask_png_base64,
            original_size=req.original_size,
            organ=req.organ,
            pixel_spacing_mm=req.pixel_spacing_mm,
        )
        if PROM_AVAILABLE:
            _map_counter.labels(
                organ=req.organ, label=req.top_label, status="ok"
            ).inc()
    except ValueError as e:
        if PROM_AVAILABLE:
            _map_counter.labels(
                organ=req.organ, label=req.top_label, status="error"
            ).inc()
        raise HTTPException(status_code=400, detail=f"Mask decode error: {str(e)}")
    except Exception as e:
        if PROM_AVAILABLE:
            _map_counter.labels(
                organ=req.organ, label=req.top_label, status="error"
            ).inc()
        logger.exception("Mapping failed")
        raise HTTPException(status_code=500, detail="Internal error during mapping. Check server logs.")
    finally:
        if PROM_AVAILABLE:
            _map_latency.observe(time.perf_counter() - t_start)

    return MapResponse(
        knowledge_mapped=KnowledgeMapped(**km_dict),
        spatial_derived=SpatialDerived(**sd_dict),
        bottleneck_features=req.bottleneck_features or {},
    )
