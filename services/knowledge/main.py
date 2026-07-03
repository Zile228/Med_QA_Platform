"""
services/knowledge/main.py
FastAPI Knowledge Service - Layer 3 | port 8003

Endpoints:
    POST /map/spatial   - mask -> SpatialDerived only
    POST /map/knowledge - label/scores -> KnowledgeMapped only
    POST /map           - legacy alias calling both (backward compat)
    GET  /health
    GET  /metrics
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
        "Latency of /map* endpoints",
        buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0],
    )
    _map_counter = Counter(
        "knowledge_map_requests_total",
        "Total number of /map* requests",
        ["organ", "label", "status", "endpoint"],
    )
except ImportError:
    PROM_AVAILABLE = False


class SpatialMapRequest(BaseModel):
    organ: str
    mask_png_base64: str
    original_size: list
    pixel_spacing_mm: Optional[float] = None
    laterality: Optional[str] = None


class SpatialMapResponse(BaseModel):
    spatial_derived: SpatialDerived


class KnowledgeMapRequest(BaseModel):
    modality: str
    organ: str
    top_label: str
    confidence: float
    all_scores: dict


class KnowledgeMapResponse(BaseModel):
    knowledge_mapped: KnowledgeMapped


class MapRequest(BaseModel):
    modality: str = Field(..., example="ultrasound")
    organ: str = Field(..., example="breast")
    top_label: str = Field(..., example="malignant")
    confidence: float = Field(..., example=0.87)
    all_scores: dict = Field(...)
    mask_png_base64: str = Field(...)
    original_size: list = Field(...)
    pixel_spacing_mm: Optional[float] = Field(default=None)
    laterality: Optional[str] = Field(default=None)


class MapResponse(BaseModel):
    knowledge_mapped: KnowledgeMapped
    spatial_derived: SpatialDerived


app = FastAPI(
    title="Knowledge Service",
    description="Layer 3 - Clinical severity mapping + spatial feature derivation",
    version="2.0.0",
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    if not PROM_AVAILABLE:
        return PlainTextResponse("# prometheus_client not installed\n", status_code=200)
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/map/spatial", response_model=SpatialMapResponse)
def map_spatial_endpoint(req: SpatialMapRequest):
    """Decode mask and compute spatial features only."""
    valid_organs = {"breast", "thyroid"}
    if req.organ.lower() not in valid_organs:
        raise HTTPException(
            status_code=400,
            detail=f"organ must be one of: {valid_organs}",
        )
    if len(req.original_size) != 2:
        raise HTTPException(status_code=400, detail="original_size must be [H, W]")

    t_start = time.perf_counter()
    try:
        sd_dict = derive_spatial(
            mask_png_base64=req.mask_png_base64,
            original_size=req.original_size,
            organ=req.organ,
            pixel_spacing_mm=req.pixel_spacing_mm,
            laterality=req.laterality,
        )
        if PROM_AVAILABLE:
            _map_counter.labels(
                organ=req.organ, label="n/a", status="ok", endpoint="/map/spatial"
            ).inc()
    except ValueError as e:
        if PROM_AVAILABLE:
            _map_counter.labels(
                organ=req.organ, label="n/a", status="error", endpoint="/map/spatial"
            ).inc()
        raise HTTPException(status_code=400, detail=f"Mask decode error: {e}")
    except Exception as e:
        if PROM_AVAILABLE:
            _map_counter.labels(
                organ=req.organ, label="n/a", status="error", endpoint="/map/spatial"
            ).inc()
        logger.exception("Spatial derivation failed")
        raise HTTPException(status_code=500, detail="Internal error during spatial derivation.")
    finally:
        if PROM_AVAILABLE:
            _map_latency.observe(time.perf_counter() - t_start)

    return SpatialMapResponse(spatial_derived=SpatialDerived(**sd_dict))


@app.post("/map/knowledge", response_model=KnowledgeMapResponse)
def map_knowledge_endpoint(req: KnowledgeMapRequest):
    """Rule-based severity + ICD-10 + risk mapping only. No mask, no image data."""
    valid_labels = {"benign", "malignant", "normal"}
    if req.top_label.lower() not in valid_labels:
        raise HTTPException(
            status_code=400,
            detail=f"top_label must be one of: {valid_labels}",
        )

    valid_organs = {"breast", "thyroid"}
    if req.organ.lower() not in valid_organs:
        raise HTTPException(
            status_code=400,
            detail=f"organ must be one of: {valid_organs}",
        )

    t_start = time.perf_counter()
    try:
        km_dict = map_knowledge(
            modality=req.modality,
            organ=req.organ,
            top_label=req.top_label,
            confidence=req.confidence,
            all_scores=req.all_scores,
        )
        if PROM_AVAILABLE:
            _map_counter.labels(
                organ=req.organ, label=req.top_label, status="ok", endpoint="/map/knowledge"
            ).inc()
    except Exception as e:
        if PROM_AVAILABLE:
            _map_counter.labels(
                organ=req.organ, label=req.top_label, status="error", endpoint="/map/knowledge"
            ).inc()
        logger.exception("Knowledge mapping failed")
        raise HTTPException(status_code=500, detail="Internal error during knowledge mapping.")
    finally:
        if PROM_AVAILABLE:
            _map_latency.observe(time.perf_counter() - t_start)

    return KnowledgeMapResponse(knowledge_mapped=KnowledgeMapped(**km_dict))


@app.post("/map", response_model=MapResponse)
def map_endpoint(req: MapRequest):
    """
    Legacy endpoint -- calls /map/spatial and /map/knowledge internally.
    Prefer the focused endpoints for new code.
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
            laterality=req.laterality,
        )
        if PROM_AVAILABLE:
            _map_counter.labels(
                organ=req.organ, label=req.top_label, status="ok", endpoint="/map"
            ).inc()
    except ValueError as e:
        if PROM_AVAILABLE:
            _map_counter.labels(
                organ=req.organ, label=req.top_label, status="error", endpoint="/map"
            ).inc()
        raise HTTPException(status_code=400, detail=f"Mask decode error: {str(e)}")
    except Exception as e:
        if PROM_AVAILABLE:
            _map_counter.labels(
                organ=req.organ, label=req.top_label, status="error", endpoint="/map"
            ).inc()
        logger.exception("Mapping failed")
        raise HTTPException(status_code=500, detail="Internal error during mapping. Check server logs.")
    finally:
        if PROM_AVAILABLE:
            _map_latency.observe(time.perf_counter() - t_start)

    return MapResponse(
        knowledge_mapped=KnowledgeMapped(**km_dict),
        spatial_derived=SpatialDerived(**sd_dict),
    )
