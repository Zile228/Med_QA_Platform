"""
services/router/main.py
========================
FastAPI Router Service -- Layer 1 | port 8001

Endpoints:
    POST /route    -- classify image -> RoutingResult
    GET  /health

Request: multipart/form-data
    image:          UploadFile
    modality_hint:  str (optional) -- 'breast' | 'thyroid'
    organ_hint:     str (optional) -- 'breast' | 'thyroid'

Response: RoutingResult schema (JSON)
"""

import os
import sys
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException

sys.path.insert(0, "/app")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from shared.schemas import RoutingResult
from services.router.model import (
    load_router, run_routing, HINT_ORGAN_TO_MODULE_KEY
)

CHECKPOINT_PATH = os.getenv(
    "ROUTER_CHECKPOINT",
    "models/checkpoints/router_effnet_b0.pth"
)
OOD_THRESHOLD = float(os.getenv("OOD_THRESHOLD", "0.6"))
DEVICE = os.getenv("DEVICE", None)

# Tap gia tri hop le cua organ_hint
_VALID_ORGAN_HINTS = set(HINT_ORGAN_TO_MODULE_KEY.keys())

_model = None
_transform = None
_device = None
_degraded = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _transform, _device, _degraded
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


@app.post("/route", response_model=RoutingResult)
async def route_image(
    image: UploadFile = File(..., description="Anh can classify modality"),
    modality_hint: Optional[str] = Form(default=None),
    organ_hint: Optional[str] = Form(default=None),
):
    """
    Nhan anh + optional hint -> tra ve RoutingResult.

    organ_hint hop le: 'breast' | 'thyroid'. Tra 400 neu gia tri khac.
    Hint bi bo qua hoan toan khi is_ood=True.
    """
    if _model is None:
        raise HTTPException(status_code=503, detail="Router model chua load.")

    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="File anh rong.")

    # Validate hint, tra 400 ngay neu khong hop le
    hint_key = None
    raw_hint = organ_hint or modality_hint
    if raw_hint is not None:
        raw_hint = raw_hint.strip().lower()
        if raw_hint not in _VALID_ORGAN_HINTS:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"organ_hint '{raw_hint}' khong hop le. "
                    f"Gia tri chap nhan: {sorted(_VALID_ORGAN_HINTS)}"
                ),
            )
        hint_key = HINT_ORGAN_TO_MODULE_KEY[raw_hint]

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
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Routing error: {str(e)}")

    return RoutingResult(**result)
