"""
services/knowledge/main.py
===========================
FastAPI Knowledge Service - Layer 3 | port 8003

Endpoints:
    POST /map     - ModelOutput + routing info -> KnowledgeMapped + SpatialDerived
    GET  /health

Request body (JSON):
    {
        "modality":         "ultrasound",
        "organ":             "breast",
        "top_label":         "malignant",
        "confidence":        0.87,
        "all_scores":        {"benign": 0.10, "malignant": 0.87, "normal": 0.03},
        "mask_png_base64":   "iVBORw0KGgoAAAANS...",
        "original_size":     [512, 512],
        "pixel_spacing_mm":  0.1
    }

Response:
    {
        "knowledge_mapped": KnowledgeMapped,
        "spatial_derived":  SpatialDerived
    }
"""

import os
import sys

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

sys.path.insert(0, "/app")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from shared.schemas import KnowledgeMapped, SpatialDerived
from services.knowledge.mapper import map_knowledge, derive_spatial


# Model cho request va response

class MapRequest(BaseModel):
    modality:         str   = Field(..., example="ultrasound")
    organ:            str   = Field(..., example="breast")
    top_label:        str   = Field(..., example="malignant")
    confidence:       float = Field(..., example=0.87)
    all_scores:       dict  = Field(..., example={"benign": 0.10, "malignant": 0.87, "normal": 0.03})
    mask_png_base64:  str   = Field(..., example="iVBORw0KGgoAAAANSUhEUgAA...")
    original_size:    list  = Field(..., example=[512, 512])
    pixel_spacing_mm: float = Field(default=0.1)


class MapResponse(BaseModel):
    knowledge_mapped: KnowledgeMapped
    spatial_derived:  SpatialDerived



app = FastAPI(
    title="Knowledge Service",
    description="Layer 3 - Clinical severity mapping + spatial feature derivation",
    version="1.0.0",
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/map", response_model=MapResponse)
def map_endpoint(req: MapRequest):
    """
    Nhận ModelOutput fields -> trả về KnowledgeMapped + SpatialDerived.

    Không stateful - mỗi request độc lập, không cần model load.
    Rule-based: audit được bởi clinician mà không cần đọc code phức tạp.
    """
    # Validate label
    valid_labels = {"benign", "malignant", "normal"}
    if req.top_label.lower() not in valid_labels:
        raise HTTPException(
            status_code=400,
            detail=f"top_label phải là một trong: {valid_labels}. Nhận: '{req.top_label}'"
        )

    # Validate organ
    valid_organs = {"breast", "thyroid"}
    if req.organ.lower() not in valid_organs:
        raise HTTPException(
            status_code=400,
            detail=f"organ phải là một trong: {valid_organs}. Nhận: '{req.organ}'"
        )

    # Validate original_size
    if len(req.original_size) != 2:
        raise HTTPException(
            status_code=400,
            detail="original_size phải là [H, W]."
        )

    try:
        # Knowledge mapping - rule-based
        km_dict = map_knowledge(
            modality=req.modality,
            organ=req.organ,
            top_label=req.top_label,
            confidence=req.confidence,
            all_scores=req.all_scores,
        )

        # Spatial derivation - từ mask base64 (KHÔNG đọc path trên disk)
        sd_dict = derive_spatial(
            mask_png_base64=req.mask_png_base64,
            original_size=req.original_size,
            organ=req.organ,
            pixel_spacing_mm=req.pixel_spacing_mm,
        )

    except ValueError as e:
        # Loi decode mask: fail loud, khong fallback im lang
        raise HTTPException(status_code=400, detail=f"Mask decode error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Mapping error: {str(e)}")

    return MapResponse(
        knowledge_mapped=KnowledgeMapped(**km_dict),
        spatial_derived=SpatialDerived(**sd_dict),
    )
