"""
services/orchestrator/birads_describer.py

Calls Gemini Vision to describe a lesion/nodule using ACR BI-RADS/TI-RADS
lexicon. Output is plain observation, not a final diagnosis -- CoT reasoning
stays independent from this description, it only adds visual context.

Must NOT import from services.knowledge, services.vision, or any other
service -- only from shared, stdlib, and services.orchestrator.llm_client.
"""

import io
import json
from typing import Optional


BIRADS_VISION_SYSTEM_PROMPT = (
    "You are a radiologist assistant analyzing medical images. "
    "Output ONLY valid JSON matching the schema exactly. "
    "No preamble, no explanation, no markdown fences."
)


def _build_birads_prompt(modality: str, organ: str) -> str:
    """
    Returns the vision prompt for a given (modality, organ) pair.

    Keyed on (modality, organ) instead of organ alone, so adding chest X-ray,
    CT, or MRI later only needs a new branch here, not a new call signature.
    """
    if modality == "ultrasound" and organ == "breast":
        return """Analyze this breast ultrasound image using ACR BI-RADS lexicon.
Describe ONLY what you directly observe. Do NOT infer a final BI-RADS category.

Output ONLY this JSON (no other text):
{
  "shape": "oval|round|irregular",
  "orientation": "parallel|not-parallel",
  "margin": "circumscribed|indistinct|angular|microlobulated|spiculated",
  "echo_pattern": "anechoic|hyperechoic|complex|hypoechoic|isoechoic|heterogeneous",
  "posterior_features": "none|enhancement|shadowing|combined",
  "calcifications": "none|macrocalcification|microcalcification",
  "observation_confidence": "high|medium|low",
  "notes": "string -- additional observations or image quality caveats, empty string if none"
}"""
    elif modality == "ultrasound" and organ == "thyroid":
        return """Analyze this thyroid ultrasound image using ACR TI-RADS lexicon.
Describe ONLY what you directly observe. Do NOT infer a final TI-RADS score.

Output ONLY this JSON (no other text):
{
  "composition": "cystic|mostly-cystic|mixed|mostly-solid|solid",
  "echogenicity": "anechoic|hyperechoic|isoechoic|hypoechoic|very-hypoechoic",
  "shape": "wider-than-tall|taller-than-wide",
  "margin": "smooth|ill-defined|lobulated|irregular|extrathyroidal-extension",
  "echogenic_foci": "none|macrocalcification|peripheral-calcification|punctate-echogenic-foci|comet-tail-artifact",
  "observation_confidence": "high|medium|low",
  "notes": "string -- additional observations or image quality caveats, empty string if none"
}"""
    else:
        raise ValueError(
            f"No visual description prompt defined for modality='{modality}', organ='{organ}'"
        )


def _make_mask_overlay(
    image_bytes: bytes,
    mask_png_base64: str,
    alpha: float = 0.35,
    color: tuple = (220, 60, 60),
) -> bytes:
    """
    Overlays the lesion mask on the original image so Gemini Vision looks
    at the right region. mask_png_base64 comes from ModelOutput.mask_png_base64.

    Returns PNG bytes of the overlaid image, or the original image_bytes if
    the mask is empty or decoding fails (graceful fallback).
    """
    import base64

    import cv2
    import numpy as np
    from PIL import Image

    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_np = np.array(img, dtype=np.float32)

        mask = None
        if mask_png_base64:
            mask_bytes = base64.b64decode(mask_png_base64)
            mask_arr = np.frombuffer(mask_bytes, dtype=np.uint8)
            mask = cv2.imdecode(mask_arr, cv2.IMREAD_GRAYSCALE)
            if mask is not None and mask.shape[:2] != img_np.shape[:2]:
                mask = cv2.resize(
                    mask,
                    (img_np.shape[1], img_np.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )

        if mask is not None and mask.max() > 0:
            overlay = img_np.copy()
            lesion = mask > 127
            for c, val in enumerate(color):
                overlay[:, :, c] = np.where(
                    lesion,
                    img_np[:, :, c] * (1 - alpha) + val * alpha,
                    img_np[:, :, c],
                )
            result_img = Image.fromarray(overlay.astype(np.uint8))
        else:
            result_img = img

        buf = io.BytesIO()
        result_img.save(buf, format="PNG")
        return buf.getvalue()

    except Exception as e:
        print(f"[birads_describer] _make_mask_overlay failed: {e} -- using raw image")
        return image_bytes


def describe_image(
    image_bytes: bytes,
    llm_client,
    modality: str,
    organ: str,
) -> Optional[dict]:
    """
    Returns a BI-RADS/TI-RADS observation dict from Gemini Vision, or None
    if the client has no multimodal support or the call/parse fails.

    Never raises -- caller (graph.py pipeline) must keep running even when
    visual description is unavailable.
    """
    if not getattr(llm_client, "_supports_multimodal", False):
        return None

    try:
        prompt = _build_birads_prompt(modality, organ)
    except ValueError as e:
        print(f"[birads_describer] {e}")
        return None

    try:
        raw = llm_client.generate_with_image(
            image_bytes=image_bytes,
            prompt=prompt,
            system=BIRADS_VISION_SYSTEM_PROMPT,
        )
        clean = raw.strip().replace("```json", "").replace("```", "").strip()
        parsed = json.loads(clean)
    except json.JSONDecodeError as e:
        print(f"[birads_describer] JSON parse failed: {e} -- continuing without visual description")
        return None
    except Exception as e:
        print(f"[birads_describer] describe_image failed: {e} -- continuing without visual description")
        return None

    if not isinstance(parsed, dict):
        print(f"[birads_describer] Unexpected response type ({type(parsed).__name__}) -- discarding")
        return None

    return parsed
