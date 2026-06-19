"""
Test cho services/knowledge/mapper.py - dispatch postprocess theo organ
và rà soát hardcode 1 organ ẩn ở nơi khác trong services/.
"""

import base64
import os
import subprocess
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from services.knowledge import mapper


def _make_mask_base64(blob_xy, size=(400, 400)):
    """Tạo 1 mask PNG base64 với 1 blob tròn tại vị trí blob_xy (cx, cy)."""
    h, w = size
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, blob_xy, 20, 255, -1)
    ok, buf = cv2.imencode(".png", mask)
    assert ok
    return base64.b64encode(buf.tobytes()).decode("ascii")


def test_derive_spatial_uses_breast_postprocess_for_breast_organ(monkeypatch):
    called = {"breast": False, "thyroid": False}

    def fake_breast(**kwargs):
        called["breast"] = True
        return {"bbox": [0, 0, 1, 1], "area_cm2": 0.0, "centroid": [0, 0],
                "location_quadrant": "central", "aspect_ratio": 1.0,
                "circularity": 1.0, "width_px": 1, "height_px": 1,
                "location_confidence": "high"}

    def fake_thyroid(**kwargs):
        called["thyroid"] = True
        return fake_breast(**kwargs)

    monkeypatch.setitem(mapper._POSTPROCESS_BY_ORGAN, "breast", fake_breast)
    monkeypatch.setitem(mapper._POSTPROCESS_BY_ORGAN, "thyroid", fake_thyroid)

    mapper.derive_spatial(
        mask_png_base64=_make_mask_base64((100, 100)),
        original_size=(400, 400),
        organ="breast",
    )

    assert called["breast"] is True
    assert called["thyroid"] is False


def test_derive_spatial_uses_thyroid_postprocess_for_thyroid_organ(monkeypatch):
    called = {"breast": False, "thyroid": False}

    def fake_breast(**kwargs):
        called["breast"] = True
        return {}

    def fake_thyroid(**kwargs):
        called["thyroid"] = True
        return {"bbox": [0, 0, 1, 1], "area_cm2": 0.0, "centroid": [0, 0],
                "location_quadrant": "left-lobe", "aspect_ratio": 1.0,
                "circularity": 1.0, "width_px": 1, "height_px": 1,
                "location_confidence": "high"}

    monkeypatch.setitem(mapper._POSTPROCESS_BY_ORGAN, "breast", fake_breast)
    monkeypatch.setitem(mapper._POSTPROCESS_BY_ORGAN, "thyroid", fake_thyroid)

    mapper.derive_spatial(
        mask_png_base64=_make_mask_base64((100, 100)),
        original_size=(400, 400),
        organ="thyroid",
    )

    assert called["thyroid"] is True
    assert called["breast"] is False


def test_derive_spatial_thyroid_location_quadrant_values():
    # Blob ở x=50 trong ảnh rộng 400 -> x < W*0.35 (140) -> left-lobe
    result = mapper.derive_spatial(
        mask_png_base64=_make_mask_base64((50, 200)),
        original_size=(400, 400),
        organ="thyroid",
    )
    assert result["location_quadrant"] == "left-lobe"
    assert result["location_quadrant"] not in {
        "upper-inner", "upper-outer", "lower-inner", "lower-outer", "central",
    }


def test_derive_spatial_unsupported_organ_raises():
    with pytest.raises(ValueError):
        mapper.derive_spatial(
            mask_png_base64=_make_mask_base64((100, 100)),
            original_size=(400, 400),
            organ="unknown_organ",
        )


def test_knowledge_dockerfile_includes_thyroid_postprocess():
    dockerfile_path = os.path.join(
        os.path.dirname(__file__), "..", "services", "knowledge", "Dockerfile"
    )
    with open(dockerfile_path, encoding="utf-8") as f:
        content = f.read()
    assert "services/vision/us_thyroid/postprocess.py" in content
    assert "services/vision/us_breast/postprocess.py" in content


def test_no_hardcoded_us_breast_import_outside_own_module():
    """
    Rà soát mục 0.2/2.3 - mọi import trực tiếp symbol từ us_breast bên ngoài
    chính module us_breast phải có nhánh us_thyroid tương ứng song song.
    """
    repo_root = os.path.join(os.path.dirname(__file__), "..")
    out = subprocess.run(
        ["grep", "-rn", "us_breast\\.", os.path.join(repo_root, "services"),
         "--include=*.py"],
        capture_output=True, text=True,
    ).stdout

    offending = []
    for line in out.splitlines():
        path = line.split(":", 1)[0]
        rel = os.path.relpath(path, repo_root)
        if rel.startswith(os.path.join("services", "vision", "us_breast")):
            continue
        offending.append(rel)

    # Mỗi file có hardcode us_breast phải đồng thời có dòng us_thyroid tương ứng
    for rel in set(offending):
        with open(os.path.join(repo_root, rel), encoding="utf-8") as f:
            file_content = f.read()
        assert "us_thyroid" in file_content, (
            f"{rel} import us_breast nhưng không có nhánh us_thyroid song song"
        )
