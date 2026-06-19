"""
services/vision/main.py
=========================
FastAPI Vision Service - Layer 2 | port 8002

Endpoints:
    POST /analyze/us_breast   - breast ultrasound inference
    POST /analyze/us_thyroid  - thyroid ultrasound inference (TN3K)
    GET  /health

Request: multipart/form-data
    image: UploadFile  - ảnh PNG/JPG
    organ: str         - 'breast' | 'thyroid'

Response: ModelOutput schema (JSON)
"""

import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse

sys.path.insert(0, "/app")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from shared.schemas import ModelOutput
from services.vision.us_breast.model  import load_model as load_breast_model,  run_inference as run_breast_inference
from services.vision.us_thyroid.model import load_model as load_thyroid_model, run_inference as run_thyroid_inference

# Cau hinh tu env

BUSI_CHECKPOINT    = os.getenv("BUSI_CHECKPOINT",    "models/checkpoints/mtl_effnet_fc_conv.pt")
THYROID_CHECKPOINT = os.getenv("THYROID_CHECKPOINT", "models/checkpoints/mtl_effnet_fc_conv_thyroid.pt")
DEVICE             = os.getenv("DEVICE", None)

# Load ca hai model khi khoi dong

_breast_model  = None
_breast_cfg    = None
_thyroid_model = None
_thyroid_cfg   = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _breast_model, _breast_cfg, _thyroid_model, _thyroid_cfg

    # Breast model
    try:
        _breast_model, _breast_cfg = load_breast_model(BUSI_CHECKPOINT, DEVICE)
        print(f"[vision] Breast model loaded OK - device: {_breast_cfg.DEVICE}")
    except FileNotFoundError as e:
        print(f"[vision] WARNING (breast): {e}")
        print("[vision] /analyze/us_breast sẽ return 503 cho đến khi có checkpoint.")

    # Thyroid model
    try:
        _thyroid_model, _thyroid_cfg = load_thyroid_model(THYROID_CHECKPOINT, DEVICE)
        print(f"[vision] Thyroid model loaded OK - device: {_thyroid_cfg.DEVICE}")
    except FileNotFoundError as e:
        print(f"[vision] WARNING (thyroid): {e}")
        print("[vision] /analyze/us_thyroid sẽ return 503 cho đến khi có checkpoint.")

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



async def _read_image(image: UploadFile) -> bytes:
    """Đọc và validate ảnh upload."""
    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="File ảnh rỗng.")
    content_type = image.content_type or ""
    if content_type and not content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"File không phải ảnh: {content_type}"
        )
    return image_bytes


# Endpoint cho Breast Ultrasound

@app.post("/analyze/us_breast", response_model=ModelOutput)
async def analyze_us_breast(
    image: UploadFile = File(..., description="Ảnh ultrasound breast (PNG/JPG)"),
    organ: str = Form(default="breast"),
):
    """
    Inference pipeline cho Breast Ultrasound (BUSI dataset):
    1. Nhận ảnh bytes
    2. run_inference -> mask (base64) + classification + bottleneck
    3. Trả về ModelOutput
    """
    if _breast_model is None:
        raise HTTPException(
            status_code=503,
            detail="Breast model chưa load. Đặt checkpoint vào models/checkpoints/ và restart.",
        )
    image_bytes = await _read_image(image)
    try:
        result = run_breast_inference(model=_breast_model, cfg=_breast_cfg, image_bytes=image_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

    return ModelOutput(
        top_label=result["top_label"],
        confidence=result["confidence"],
        all_scores=result["all_scores"],
        mask_png_base64=result["mask_png_base64"],
        bottleneck_features=result["bottleneck_features"],
        original_size=list(result["original_size"]),
    )


# Endpoint cho Thyroid Ultrasound

@app.post("/analyze/us_thyroid", response_model=ModelOutput)
async def analyze_us_thyroid(
    image: UploadFile = File(..., description="Ảnh ultrasound thyroid (PNG/JPG)"),
    organ: str = Form(default="thyroid"),
):
    """
    Inference pipeline cho Thyroid Ultrasound (TN3K dataset):
    1. Nhận ảnh bytes
    2. run_inference -> mask (base64) + classification (benign/malignant) + bottleneck
    3. Trả về ModelOutput - cùng schema với us_breast

    Model: UNet_MTL với EfficientNet-B4, FC head, 2 classes.
    Checkpoint: models/checkpoints/mtl_effnet_fc_conv_thyroid.pt
    """
    if _thyroid_model is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Thyroid model chưa load. "
                "Chạy notebook tn3k_thyroid_train.ipynb để tạo checkpoint, "
                "sau đó đặt vào models/checkpoints/mtl_effnet_fc_conv_thyroid.pt và restart."
            ),
        )
    image_bytes = await _read_image(image)
    try:
        result = run_thyroid_inference(model=_thyroid_model, cfg=_thyroid_cfg, image_bytes=image_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

    return ModelOutput(
        top_label=result["top_label"],
        confidence=result["confidence"],
        all_scores=result["all_scores"],
        mask_png_base64=result["mask_png_base64"],
        bottleneck_features=result["bottleneck_features"],
        original_size=list(result["original_size"]),
    )


# Placeholder cho X-ray (chua implement)

@app.post("/analyze/xray")
async def analyze_xray(image: UploadFile = File(...)):
    """Placeholder - Phase 2 (NIH ChestX-ray14)."""
    raise HTTPException(
        status_code=501,
        detail="X-Ray module chưa implement. Roadmap: Phase 2.",
    )
