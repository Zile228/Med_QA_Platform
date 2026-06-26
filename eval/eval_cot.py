"""
eval/eval_cot.py
=================
Giai doan 2 - Danh gia CoT (Chain-of-Thought reasoning).
Chay offline, khong can Docker. Goi truc tiep cac ham noi bo
(knowledge mapper, vision model, visual_interpreter) va LLM client.
Ket qua:
  1. CoT vs Ground Truth (nhan goc dataset):
       - Precision / Recall / F1 macro per-class va tong the.
       - Confusion matrix.
  2. CoT vs CNN (label_agreement):
       - Cohen's Kappa (dung cho agreement, khong phai F1).
       - Ty le label_agreement / consensus / hard_conflict (theo dung dinh nghia code thuc te).
       - Bang cross-tab 2x2 (label_agreement x consensus) de show overlap.
  3. CoT parse failure rate:
       - Dem so case cot_label == "unknown" hoac severity == "undetermined".
       - Cac metric chinh LOai cac case nay khoi mau so va bao cao rieng.
  4. Kiem tra bất nhất cot_label="normal" tren thyroid:
       - Bao cao so luong va ti le case thyroid ma CoT chon "normal" (loi thiet ke prompt).
       - Sau khi fix _build_cot_prompt, van can audit de xac nhan so nay = 0.
  5. Self-consistency (--consistency_runs > 1):
       - Chon ngau nhien --consistency_n mau, chay CoT --consistency_runs lan moi mau.
       - Bao cao % lan ra cung cot_label / severity_level.

THROTTLING / RELIABILITY (moi):
  6. Rate limiting cho LLM API (--rate_limit, mac dinh 10 req/phut):
       - Tat ca request goi qua llm_client.generate() deu di qua RateLimitedLLMClient,
         dam bao khong vuot qua N request / 60s (sliding window).
       - Dat --rate_limit 0 de tat throttling (vd khi dung backend local "ollama").
  7. Retry-with-backoff (--max_retries, --retry_base_delay):
       - Neu llm_client.generate() loi (rate-limit 429, timeout, network...),
         tu dong retry voi exponential backoff truoc khi cho la parse-fail.
  8. Resume / checkpoint (--resume):
       - Moi record duoc ghi ngay ra file JSONL trong out_dir khi vua chay xong
         (khong doi den cuoi dataset).
       - Neu chay lai voi --resume, cac sample co image_path da co trong file
         checkpoint se duoc bo qua (doc lai ket qua cu), tranh goi lai LLM
         va ton quota khi script bi crash/mat mang giua duong.

Cau truc thu muc can thiet (giong eval_vision.py):
  BUSI:
    data/busi/test_busi/
      benign/
        benign (1).png
        benign (1)_mask.png
        ...
      malignant/
        malignant (1).png
        malignant (1)_mask.png
        ...
      normal/
        normal (1).png
        normal (1)_mask.png   <- co the khong ton tai hoac rong
        ...
  TN3K:
    data/tn3k/test_tn3k/
      test-image/
        001.jpg
        002.jpg
        ...
      test-mask/
        001.png
        ...
      label4test.csv   <- cot "image_name", cot "label" (0=benign, 1=malignant)
LLM backend duoc chon qua env LLM_BACKEND (ollama | google | mock).
Neu khong set, mac dinh la ollama. Xem services/orchestrator/llm_client.py.
Chay:
  python eval/eval_cot.py \\
    --busi_dir           data/busi/test_busi \\
    --tn3k_dir           data/tn3k/test_tn3k \\
    --busi_ckpt          models/checkpoints/mtl_effnet_fc_conv_breast.pt \\
    --thyroid_ckpt       models/checkpoints/mtl_effnet_fc_conv_thyroid.pt \\
    --out_dir            eval/results/cot \\
    [--device            cpu|cuda] \\
    [--max_busi          N]         # gioi han so mau BUSI (bo trong = chay het) \\
    [--max_tn3k          N]         # gioi han so mau TN3K \\
    [--consistency_runs  N]         # so lan chay CoT tren moi mau (mac dinh: 1) \\
    [--consistency_n     N]         # so mau dung cho consistency test (mac dinh: 20) \\
    [--skip_busi]                   # bo qua BUSI \\
    [--skip_tn3k]                   # bo qua TN3K \\
    [--seed              42]        # seed cho shuffle \\
    [--rate_limit        10]        # so request LLM toi da / phut (0 = tat) \\
    [--max_retries       3]         # so lan retry khi llm_client.generate() loi \\
    [--retry_base_delay  2.0]       # delay co so (giay) cho exponential backoff \\
    [--resume]                      # bo qua sample da co trong checkpoint cu
"""
import argparse
import json
import os
import random
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from sklearn.metrics import (
        classification_report,
        confusion_matrix,
        cohen_kappa_score,
    )
    _SKLEARN_OK = True
except ImportError:
    print("[warn] scikit-learn not installed -- some metrics will be skipped")
    _SKLEARN_OK = False


# ---------------------------------------------------------------------------
# Rate limiting + retry cho LLM client
# ---------------------------------------------------------------------------

class RateLimitedLLMClient:
    """
    Wrap mot llm_client bat ky, dam bao khong goi .generate() qua N lan/phut.

    Dung sliding-window don gian: ghi lai timestamp cua cac lan goi gan nhat
    (trong 60s gan nhat). Neu da dat N lan trong window, sleep cho den khi
    timestamp cu nhat het han roi moi cho goi tiep.

    Cong them retry-with-exponential-backoff: neu inner_client.generate()
    raise exception (vd HTTP 429 / timeout / loi mang tam thoi), tu dong
    cho roi thu lai toi da max_retries lan truoc khi raise len tren cho
    _run_cot_once xu ly nhu parse-fail binh thuong.

    Thread-safe (dung threading.Lock) de phong truong hop sau nay code
    duoc chay da luong; hien tai script chay tuan tu nen khong bat buoc
    nhung khong gay hai gi.
    """

    def __init__(
        self,
        inner_client,
        max_calls_per_minute: int = 10,
        max_retries: int = 3,
        retry_base_delay: float = 2.0,
    ):
        self._inner = inner_client
        self._max_calls = max_calls_per_minute
        self._max_retries = max_retries
        self._retry_base_delay = retry_base_delay
        self._timestamps = []
        self._lock = threading.Lock()

    def _wait_for_slot(self):
        """Block (sleep) cho den khi co cho trong sliding window 60s."""
        if not self._max_calls or self._max_calls <= 0:
            return  # throttling bi tat
        while True:
            with self._lock:
                now = time.monotonic()
                self._timestamps = [t for t in self._timestamps if now - t < 60]
                if len(self._timestamps) < self._max_calls:
                    self._timestamps.append(now)
                    return
                sleep_time = 60 - (now - self._timestamps[0]) + 0.05
            if sleep_time > 0:
                print(
                    f"      [rate_limit] da dat {self._max_calls} request/phut, "
                    f"cho {sleep_time:.1f}s truoc khi goi tiep...",
                    flush=True,
                )
                time.sleep(sleep_time)
            # loop lai de kiem tra + dang ky slot mot cach an toan

    def generate(self, *args, **kwargs):
        """
        Goi inner_client.generate(...) voi rate-limit + retry-backoff.
        Moi lan thu (bao gom ca retry) deu phai xin slot rieng trong
        sliding window, tranh viec retry lam vuot rate limit.
        """
        last_exc = None
        for attempt in range(self._max_retries + 1):
            self._wait_for_slot()
            try:
                return self._inner.generate(*args, **kwargs)
            except Exception as e:  # noqa: BLE001 - can bat moi loai loi tu API/network
                last_exc = e
                if attempt >= self._max_retries:
                    break
                delay = self._retry_base_delay * (2 ** attempt)
                print(
                    f"      [retry] llm_client.generate() loi (lan {attempt + 1}"
                    f"/{self._max_retries + 1}): {type(e).__name__}: {e} "
                    f"-> cho {delay:.1f}s roi thu lai...",
                    flush=True,
                )
                time.sleep(delay)
        # het so lan retry, raise loi cuoi cung de _run_cot_once xu ly nhu binh thuong
        raise last_exc

    def __getattr__(self, name):
        # forward moi attribute/method khac sang client goc (vd close(), config...)
        return getattr(self._inner, name)


# ---------------------------------------------------------------------------
# Checkpoint / resume
# ---------------------------------------------------------------------------

def _checkpoint_path(out_dir: Path, dataset_name: str) -> Path:
    return out_dir / f"checkpoint_{dataset_name.lower()}.jsonl"


def _load_checkpoint(out_dir: Path, dataset_name: str) -> dict:
    """
    Doc checkpoint JSONL cu (neu co). Tra ve dict {image_path: record}.
    Moi dong file la 1 record JSON da duoc tra ve boi _run_cot_once truoc do.
    Dong loi/khong parse duoc se bi bo qua (in warning) de khong lam crash resume.
    """
    path = _checkpoint_path(out_dir, dataset_name)
    records_by_path = {}
    if not path.exists():
        return records_by_path
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  [warn] checkpoint {path} dong {line_no} loi JSON: {e} -> bo qua dong nay")
                continue
            img_path = rec.get("image_path")
            if img_path:
                records_by_path[img_path] = rec
    return records_by_path


def _append_checkpoint(out_dir: Path, dataset_name: str, record: dict):
    """Ghi ngay 1 record ra file checkpoint JSONL (append, flush ngay)."""
    path = _checkpoint_path(out_dir, dataset_name)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


def _import_modules():
    """
    Import cac module noi bo (vision, knowledge, orchestrator) theo lazy pattern.
    Tra ve dict chua tat ca cac ham va lop can thiet.
    """
    from services.vision.us_breast.model import (
        load_model as load_breast,
        run_inference as run_breast,
    )
    from services.vision.us_thyroid.model import (
        load_model as load_thyroid,
        run_inference as run_thyroid,
    )
    from services.knowledge.mapper import map_knowledge, derive_spatial
    from services.orchestrator.visual_interpreter import interpret_visual_features
    from services.orchestrator.llm_client import get_llm_client
    from services.orchestrator.graph import _build_cot_prompt, COT_SYSTEM_PROMPT
    return {
        "load_breast": load_breast,
        "run_breast": run_breast,
        "load_thyroid": load_thyroid,
        "run_thyroid": run_thyroid,
        "map_knowledge": map_knowledge,
        "derive_spatial": derive_spatial,
        "interpret_visual_features": interpret_visual_features,
        "get_llm_client": get_llm_client,
        "_build_cot_prompt": _build_cot_prompt,
        "COT_SYSTEM_PROMPT": COT_SYSTEM_PROMPT,
    }


def _load_image_bytes(path: Path) -> bytes:
    return path.read_bytes()


def collect_busi_dataset(busi_dir: Path) -> list:
    """
    Thu thap anh BUSI tu thu muc test_busi/benign, malignant, normal.
    Tra ve list dict: {image_bytes, gt_label, mask_bytes (co the None), image_path}.
    """
    samples = []
    for gt_label in ("benign", "malignant", "normal"):
        cls_dir = busi_dir / gt_label
        if not cls_dir.exists():
            print(f"  [warn] Khong tim thay {cls_dir}, bo qua.")
            continue
        for p in sorted(cls_dir.iterdir()):
            if p.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp"}:
                continue
            name = p.stem
            if name.endswith("_mask"):
                continue
            mask_path = cls_dir / f"{name}_mask{p.suffix}"
            mask_bytes = mask_path.read_bytes() if mask_path.exists() else None
            samples.append({
                "image_bytes": p.read_bytes(),
                "gt_label": gt_label,
                "mask_bytes": mask_bytes,
                "image_path": str(p),
                "organ": "breast",
                "dataset": "busi",
            })
    return samples


def collect_tn3k_dataset(tn3k_dir: Path) -> list:
    """
    Thu thap anh TN3K tu thu muc test_tn3k/test-image va label4test.csv.
    label4test.csv KHONG co header: cot 0 = ten file (co duoi, vd "0000.jpg"),
    cot 1 = nhan so (0=benign, 1=malignant).
    Tra ve list dict: {image_bytes, gt_label, mask_bytes (co the None), image_path}.
    """
    import csv
    csv_path = tn3k_dir / "label4test.csv"
    image_dir = tn3k_dir / "test-image"
    mask_dir = tn3k_dir / "test-mask"
    if not csv_path.exists():
        print(f"  [warn] Khong tim thay {csv_path}")
        return []
    if not image_dir.exists():
        print(f"  [warn] Khong tim thay {image_dir}")
        return []
    label_map = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            name = row[0].strip()
            lbl_raw = row[1].strip()
            if not name or lbl_raw not in ("0", "1"):
                continue
            gt = "benign" if lbl_raw == "0" else "malignant"
            label_map[name] = gt
            label_map[Path(name).stem] = gt
    if not label_map:
        print(f"  [warn] label4test.csv doc duoc 0 dong hop le, kiem tra lai dinh dang file.")
        return []
    samples = []
    n_missing = 0
    for p in sorted(image_dir.iterdir()):
        if p.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp"}:
            continue
        gt_label = label_map.get(p.name) or label_map.get(p.stem)
        if gt_label is None:
            n_missing += 1
            continue
        mask_candidates = [
            mask_dir / f"{p.stem}.png",
            mask_dir / p.name,
        ]
        mask_bytes = None
        for mc in mask_candidates:
            if mc.exists():
                mask_bytes = mc.read_bytes()
                break
        samples.append({
            "image_bytes": p.read_bytes(),
            "gt_label": gt_label,
            "mask_bytes": mask_bytes,
            "image_path": str(p),
            "organ": "thyroid",
            "dataset": "tn3k",
        })
    if n_missing > 0:
        print(f"  [warn] {n_missing} anh khong tim duoc nhan trong CSV, bo qua.")
    return samples


def _decode_mask_for_derive(mask_bytes: Optional[bytes], original_size: list) -> Optional[str]:
    """
    Neu co mask_bytes, encode lai sang base64 PNG de truyen vao derive_spatial.
    Neu khong co, tra ve None (se dung empty_spatial fallback trong mapper).
    """
    if mask_bytes is None:
        return None
    import base64
    arr = np.frombuffer(mask_bytes, dtype=np.uint8)
    mask = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    ok, enc = cv2.imencode(".png", mask)
    if not ok:
        return None
    return base64.b64encode(enc.tobytes()).decode("ascii")


def _run_cot_once(
    sample: dict,
    run_model_fn,
    model,
    cfg,
    fns: dict,
    llm_client,
) -> dict:
    """
    Chay toan bo pipeline CoT mot lan cho 1 sample.
    Pipeline:
      1. run_inference (vision model) -> model_output
      2. derive_spatial (tu mask GT neu co, neu khong tu mask du doan) -> spatial
      3. interpret_visual_features -> visual_features (chi dung spatial flags)
      4. _build_cot_prompt -> prompt
      5. llm_client.generate (CoT) -> raw_text -> parse JSON
         (llm_client co the la RateLimitedLLMClient, da tu xu ly throttle + retry)
      6. Tinh label_agreement, consensus, hard_conflict tu code thuc te cua graph.py
    Tra ve dict chua:
      cnn_label, cnn_confidence, gt_label, organ,
      cot_label, cot_severity, cot_severity_level, cot_icd10, cot_reasoning,
      mapper_severity_level, mapper_icd10,
      label_agreement, consensus, hard_conflict,
      cot_parse_failed, image_path, latency_cot_ms
    """
    organ = sample["organ"]
    gt_label = sample["gt_label"]
    image_path = sample["image_path"]
    t_infer_start = time.perf_counter()
    try:
        mo = run_model_fn(model, cfg, sample["image_bytes"])
    except Exception as e:
        return {
            "error": f"vision_inference: {e}",
            "image_path": image_path,
            "gt_label": gt_label,
            "organ": organ,
        }
    t_infer = (time.perf_counter() - t_infer_start) * 1000
    cnn_label = mo.get("top_label", "unknown")
    cnn_confidence = mo.get("confidence", 0.0)
    original_size = mo.get("original_size", [512, 512])
    if sample.get("mask_bytes") is not None:
        mask_b64 = _decode_mask_for_derive(sample["mask_bytes"], original_size)
    else:
        mask_b64 = mo.get("mask_png_base64", "")
    try:
        if mask_b64:
            spatial = fns["derive_spatial"](
                mask_png_base64=mask_b64,
                original_size=tuple(original_size),
                organ=organ,
                pixel_spacing_mm=None,
                laterality=None,
            )
        else:
            spatial = {
                "bbox": [0, 0, 0, 0],
                "area_cm2": None,
                "pixel_spacing_reliable": False,
                "centroid": [original_size[1] // 2, original_size[0] // 2],
                "location_quadrant": "none",
                "aspect_ratio": 1.0,
                "aspect_ratio_interpretation": "",
                "circularity": 1.0,
                "width_px": 0,
                "height_px": 0,
                "location_confidence": "low",
            }
    except Exception as e:
        print(f"  [warn] derive_spatial failed ({image_path}): {e}")
        spatial = {
            "bbox": [0, 0, 0, 0],
            "area_cm2": None,
            "pixel_spacing_reliable": False,
            "centroid": [256, 256],
            "location_quadrant": "unknown",
            "aspect_ratio": 1.0,
            "aspect_ratio_interpretation": "",
            "circularity": 1.0,
            "width_px": 0,
            "height_px": 0,
            "location_confidence": "low",
        }
    km = fns["map_knowledge"](
        modality="ultrasound",
        organ=organ,
        top_label=cnn_label,
        confidence=cnn_confidence,
        all_scores=mo.get("all_scores", {}),
    )
    mapper_severity_level = km.get("severity_level", 0)
    mapper_icd10 = km.get("icd10_hint", "")
    visual_features = fns["interpret_visual_features"](
        bottleneck=mo.get("bottleneck_enriched", {}),
        texture=mo.get("texture_features", {}),
        uncertainty=mo.get("uncertainty", {}),
        gradcam_overlap=mo.get("gradcam_mask_overlap", {}),
        spatial=spatial,
        organ=organ,
    )
    prompt = fns["_build_cot_prompt"](
        spatial=spatial,
        visual_features=visual_features,
        rag_chunks=[],
        organ=organ,
    )
    t_cot_start = time.perf_counter()
    cot_parse_failed = False
    raw = None
    try:
        raw = llm_client.generate(prompt, fns["COT_SYSTEM_PROMPT"])
        clean = raw.strip().replace("```json", "").replace("```", "").strip()
        parsed = json.loads(clean)
        cot_label = str(parsed.get("cot_label", "unknown"))
        cot_severity = str(parsed.get("severity", "incidental"))
        cot_severity_level = int(parsed.get("severity_level", 1))
        cot_icd10 = str(parsed.get("icd10_hint", "R93.8"))
        cot_risk = str(parsed.get("risk_category", ""))
        cot_reasoning = str(parsed.get("reasoning", ""))
    except Exception as e:
        cot_parse_failed = True
        cot_label = "unknown"
        cot_severity = "undetermined"
        cot_severity_level = 0
        cot_icd10 = "R93.8"
        cot_risk = "undetermined"
        raw_preview = raw[:300] if raw else "(empty)"
        cot_reasoning = f"parse_error: {e} | raw_preview: {raw_preview}"
        print(f"      [parse_fail] {type(e).__name__}: {e}", flush=True)
        print(f"      [raw_preview] {raw_preview!r}", flush=True)
    t_cot = (time.perf_counter() - t_cot_start) * 1000
    cot_undetermined = cot_severity_level == 0 or cot_severity == "undetermined"
    if cot_undetermined:
        consensus = None
        label_agreement = None
        hard_conflict = None
    else:
        consensus = abs(mapper_severity_level - cot_severity_level) <= 1
        label_agreement = cnn_label.lower() == cot_label.lower()
        hard_conflict = (
            not label_agreement
            and (
                abs(mapper_severity_level - cot_severity_level) > 1
                or (cot_label in ("malignant",) and cnn_label in ("benign", "normal"))
            )
        )
    return {
        "image_path": image_path,
        "organ": organ,
        "gt_label": gt_label,
        "cnn_label": cnn_label,
        "cnn_confidence": cnn_confidence,
        "cot_label": cot_label,
        "cot_severity": cot_severity,
        "cot_severity_level": cot_severity_level,
        "cot_icd10": cot_icd10,
        "cot_risk": cot_risk,
        "cot_reasoning": cot_reasoning,
        "mapper_severity_level": mapper_severity_level,
        "mapper_icd10": mapper_icd10,
        "label_agreement": label_agreement,
        "consensus": consensus,
        "hard_conflict": hard_conflict,
        "cot_parse_failed": cot_parse_failed,
        "latency_vision_ms": round(t_infer, 2),
        "latency_cot_ms": round(t_cot, 2),
    }


def _compute_cot_vs_gt_metrics(records: list, valid_labels: list) -> dict:
    """
    Tinh F1 / Precision / Recall cua cot_label so voi gt_label (Ground Truth dataset).
    Chi tinh tren cac record khong bi parse-failed.
    """
    valid = [r for r in records if not r.get("cot_parse_failed") and r.get("cot_label") != "unknown"]
    if not valid:
        return {"error": "Khong co record hop le de tinh CoT vs GT"}
    y_true = [r["gt_label"] for r in valid]
    y_pred = [r["cot_label"] for r in valid]
    if not _SKLEARN_OK:
        return {"error": "scikit-learn khong co san"}
    label_set = sorted(set(y_true) | set(y_pred))
    label_set = [l for l in valid_labels if l in label_set] + [l for l in label_set if l not in valid_labels]
    report = classification_report(
        y_true, y_pred,
        labels=label_set,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=label_set).tolist()
    return {
        "n_valid": len(valid),
        "labels_used": label_set,
        "macro_f1":        round(report.get("macro avg", {}).get("f1-score", 0), 4),
        "macro_precision": round(report.get("macro avg", {}).get("precision", 0), 4),
        "macro_recall":    round(report.get("macro avg", {}).get("recall", 0), 4),
        "weighted_f1":     round(report.get("weighted avg", {}).get("f1-score", 0), 4),
        "per_class": {
            cls: {
                "precision": round(report.get(cls, {}).get("precision", 0), 4),
                "recall":    round(report.get(cls, {}).get("recall", 0), 4),
                "f1":        round(report.get(cls, {}).get("f1-score", 0), 4),
                "support":   report.get(cls, {}).get("support", 0),
            }
            for cls in label_set if cls in report
        },
        "confusion_matrix": {
            "labels": label_set,
            "matrix": cm,
        },
    }


def _compute_cot_vs_cnn_metrics(records: list, valid_labels: list) -> dict:
    """
    Tinh Cohen's Kappa giua cot_label va cnn_label.
    Them ty le label_agreement, consensus, hard_conflict.
    Them bang cross-tab 2x2 (label_agreement x consensus).
    """
    valid = [r for r in records if not r.get("cot_parse_failed") and r.get("cot_label") != "unknown"]
    total = len(records)
    n_valid = len(valid)
    n_parse_failed = total - n_valid
    if not valid:
        return {
            "n_total": total,
            "n_parse_failed": n_parse_failed,
            "error": "Khong co record hop le",
        }
    y_cnn = [r["cnn_label"] for r in valid]
    y_cot = [r["cot_label"] for r in valid]
    kappa = None
    if _SKLEARN_OK:
        try:
            kappa = round(float(cohen_kappa_score(y_cnn, y_cot)), 4)
        except Exception as e:
            kappa = f"error: {e}"
    n_la_true  = sum(1 for r in valid if r["label_agreement"] is True)
    n_la_false = sum(1 for r in valid if r["label_agreement"] is False)
    n_la_none  = sum(1 for r in valid if r["label_agreement"] is None)
    n_con_true  = sum(1 for r in valid if r["consensus"] is True)
    n_con_false = sum(1 for r in valid if r["consensus"] is False)
    n_con_none  = sum(1 for r in valid if r["consensus"] is None)
    n_hc_true  = sum(1 for r in valid if r["hard_conflict"] is True)
    n_hc_false = sum(1 for r in valid if r["hard_conflict"] is False)
    n_hc_none  = sum(1 for r in valid if r["hard_conflict"] is None)

    def pct(n):
        return round(n / n_valid * 100, 2) if n_valid > 0 else 0.0

    la_and_con_tt = sum(1 for r in valid if r["label_agreement"] is True and r["consensus"] is True)
    la_and_con_tf = sum(1 for r in valid if r["label_agreement"] is True and r["consensus"] is False)
    la_and_con_ft = sum(1 for r in valid if r["label_agreement"] is False and r["consensus"] is True)
    la_and_con_ff = sum(1 for r in valid if r["label_agreement"] is False and r["consensus"] is False)
    thyroid_normal_cot = sum(
        1 for r in valid if r["organ"] == "thyroid" and r["cot_label"] == "normal"
    )
    n_thyroid = sum(1 for r in valid if r["organ"] == "thyroid")
    return {
        "n_total": total,
        "n_valid": n_valid,
        "n_parse_failed": n_parse_failed,
        "parse_failure_rate_pct": round(n_parse_failed / total * 100, 2) if total > 0 else 0.0,
        "cohen_kappa_cnn_vs_cot": kappa,
        "label_agreement": {
            "true":  n_la_true,  "true_pct":  pct(n_la_true),
            "false": n_la_false, "false_pct": pct(n_la_false),
            "none":  n_la_none,  "none_pct":  pct(n_la_none),
        },
        "consensus": {
            "true":  n_con_true,  "true_pct":  pct(n_con_true),
            "false": n_con_false, "false_pct": pct(n_con_false),
            "none":  n_con_none,  "none_pct":  pct(n_con_none),
        },
        "hard_conflict": {
            "true":  n_hc_true,  "true_pct":  pct(n_hc_true),
            "false": n_hc_false, "false_pct": pct(n_hc_false),
            "none":  n_hc_none,  "none_pct":  pct(n_hc_none),
        },
        "cross_tab_label_agreement_x_consensus": {
            "description": (
                "Hang = label_agreement (True/False), Cot = consensus (True/False). "
                "Loai None. So nay cho thay muc do phu thuoc giua 2 metric."
            ),
            "la_true_con_true":   la_and_con_tt,
            "la_true_con_false":  la_and_con_tf,
            "la_false_con_true":  la_and_con_ft,
            "la_false_con_false": la_and_con_ff,
        },
        "thyroid_cot_normal_audit": {
            "n_thyroid_valid": n_thyroid,
            "n_thyroid_cot_normal": thyroid_normal_cot,
            "rate_pct": round(thyroid_normal_cot / n_thyroid * 100, 2) if n_thyroid > 0 else 0.0,
            "note": (
                "CoT chon 'normal' cho anh thyroid la loi thiet ke prompt, "
                "vi TN3K chi co 2 lop (benign/malignant). "
                "Gia tri nay nen = 0 sau khi fix _build_cot_prompt."
            ),
        },
    }


def _compute_self_consistency(
    samples: list,
    run_model_fn,
    model,
    cfg,
    fns: dict,
    llm_client,
    n_samples: int = 20,
    n_runs: int = 5,
    seed: int = 42,
) -> dict:
    """
    Chon ngau nhien n_samples mau, chay CoT n_runs lan moi mau.
    Do self-consistency = % lan ra cung cot_label va severity_level.
    """
    rng = random.Random(seed)
    chosen = rng.sample(samples, min(n_samples, len(samples)))
    results_by_sample = []
    for s in chosen:
        runs = []
        for _ in range(n_runs):
            r = _run_cot_once(s, run_model_fn, model, cfg, fns, llm_client)
            runs.append({
                "cot_label": r.get("cot_label", "unknown"),
                "cot_severity_level": r.get("cot_severity_level", 0),
                "cot_parse_failed": r.get("cot_parse_failed", False),
            })
        valid_runs = [run for run in runs if not run["cot_parse_failed"]]
        if not valid_runs:
            results_by_sample.append({
                "image_path": s["image_path"],
                "gt_label": s["gt_label"],
                "n_valid_runs": 0,
                "label_consistency_pct": None,
                "severity_consistency_pct": None,
            })
            continue
        labels = [run["cot_label"] for run in valid_runs]
        sevs = [run["cot_severity_level"] for run in valid_runs]
        most_common_label = max(set(labels), key=labels.count)
        most_common_sev = max(set(sevs), key=sevs.count)
        label_agree_pct = round(labels.count(most_common_label) / len(valid_runs) * 100, 1)
        sev_agree_pct = round(sevs.count(most_common_sev) / len(valid_runs) * 100, 1)
        results_by_sample.append({
            "image_path": s["image_path"],
            "gt_label": s["gt_label"],
            "n_valid_runs": len(valid_runs),
            "label_mode": most_common_label,
            "severity_mode": most_common_sev,
            "label_consistency_pct": label_agree_pct,
            "severity_consistency_pct": sev_agree_pct,
        })
    valid_entries = [e for e in results_by_sample if e["label_consistency_pct"] is not None]
    mean_label_cons = (
        round(np.mean([e["label_consistency_pct"] for e in valid_entries]), 2)
        if valid_entries else None
    )
    mean_sev_cons = (
        round(np.mean([e["severity_consistency_pct"] for e in valid_entries]), 2)
        if valid_entries else None
    )
    return {
        "n_runs_per_sample": n_runs,
        "n_samples_tested": len(chosen),
        "mean_label_consistency_pct": mean_label_cons,
        "mean_severity_consistency_pct": mean_sev_cons,
        "note": (
            "mean_label_consistency_pct < 80 cho thay CoT khong on dinh "
            "va tat ca so lieu agreement/consensus trong phan chinh co the bi sai lech cao."
        ),
        "per_sample": results_by_sample,
    }


def eval_dataset(
    samples: list,
    run_model_fn,
    model,
    cfg,
    fns: dict,
    llm_client,
    dataset_name: str,
    valid_labels: list,
    max_samples: Optional[int],
    seed: int,
    consistency_runs: int,
    consistency_n: int,
    out_dir: Optional[Path] = None,
    resume: bool = False,
) -> dict:
    """
    Chay toan bo danh gia Phase 2 cho 1 dataset (BUSI hoac TN3K).

    Neu out_dir duoc truyen vao:
      - Moi record (thanh cong hoac loi) duoc append ngay vao
        out_dir/checkpoint_<dataset_name>.jsonl khi vua tinh xong, khong doi
        den cuoi loop. Giup khong mat ket qua neu script bi crash/mat mang
        giua duong (vi du het quota API giua chung, mat ket noi, etc).
      - Neu resume=True: cac sample co image_path da ton tai trong checkpoint
        cu se duoc doc lai ket qua cu, KHONG goi lai vision model / LLM,
        giup tiet kiem quota khi chay lai sau khi bi loi giua duong.
    """
    rng = random.Random(seed)
    rng.shuffle(samples)
    if max_samples is not None:
        samples = samples[:max_samples]

    cached_records = {}
    if out_dir is not None and resume:
        cached_records = _load_checkpoint(out_dir, dataset_name)
        if cached_records:
            print(
                f"  [resume] Tim thay {len(cached_records)} record cu trong checkpoint, "
                f"se bo qua cac sample da co ket qua.",
                flush=True,
            )

    print(f"  Chay CoT tren {len(samples)} mau ({dataset_name})...", flush=True)
    records = []
    n_skipped_resume = 0
    for i, s in enumerate(samples):
        img_name = Path(s["image_path"]).name
        cached = cached_records.get(s["image_path"]) if resume else None
        if cached is not None:
            n_skipped_resume += 1
            print(f"    [{i + 1}/{len(samples)}] {img_name} ... [resume] dung ket qua cu", flush=True)
            records.append(cached)
            continue

        print(f"    [{i + 1}/{len(samples)}] {img_name} ... ", end="", flush=True)
        r = _run_cot_once(s, run_model_fn, model, cfg, fns, llm_client)
        if "error" in r:
            print(f"ERROR: {r['error']}", flush=True)
        elif r.get("cot_parse_failed"):
            print(f"parse_failed | cnn={r.get('cnn_label')} gt={r.get('gt_label')}", flush=True)
        else:
            print(
                f"cot={r.get('cot_label')} cnn={r.get('cnn_label')} "
                f"gt={r.get('gt_label')} agree={r.get('label_agreement')} "
                f"t={r.get('latency_cot_ms')}ms",
                flush=True,
            )
        records.append(r)
        if out_dir is not None:
            _append_checkpoint(out_dir, dataset_name, r)

    if n_skipped_resume:
        print(f"  [resume] Da bo qua {n_skipped_resume}/{len(samples)} mau nho checkpoint cu.")

    error_count = sum(1 for r in records if "error" in r)
    if error_count:
        print(f"  [warn] {error_count}/{len(records)} records bi loi inference (se bo qua trong tinh metric).")
    cot_gt = _compute_cot_vs_gt_metrics(records, valid_labels)
    cot_cnn = _compute_cot_vs_cnn_metrics(records, valid_labels)
    consistency_result = None
    if consistency_runs > 1:
        print(f"  Chay self-consistency ({consistency_runs} lan / {consistency_n} mau)...")
        consistency_result = _compute_self_consistency(
            samples=samples,
            run_model_fn=run_model_fn,
            model=model,
            cfg=cfg,
            fns=fns,
            llm_client=llm_client,
            n_samples=consistency_n,
            n_runs=consistency_runs,
            seed=seed,
        )
    valid_records = [r for r in records if "error" not in r]
    cot_latencies = [r["latency_cot_ms"] for r in valid_records]
    return {
        "dataset": dataset_name,
        "organ": samples[0]["organ"] if samples else "unknown",
        "n_total": len(records),
        "n_inference_errors": error_count,
        "valid_labels": valid_labels,
        "cot_vs_ground_truth": cot_gt,
        "cot_vs_cnn": cot_cnn,
        "latency_cot_ms": {
            "mean":   round(float(np.mean(cot_latencies)), 2) if cot_latencies else None,
            "median": round(float(np.median(cot_latencies)), 2) if cot_latencies else None,
            "p95":    round(float(np.percentile(cot_latencies, 95)), 2) if cot_latencies else None,
        },
        "self_consistency": consistency_result,
        "per_sample_records": records,
    }


def print_cot_gt_summary(res: dict, title: str):
    print(f"\n{'=' * 60}")
    print(f"COT vs GROUND TRUTH - {title}")
    print(f"{'=' * 60}")
    m = res.get("cot_vs_ground_truth", {})
    if "error" in m:
        print(f"  Loi: {m['error']}")
        return
    print(f"  Mau hop le (khong parse-failed): {m.get('n_valid')}")
    print(f"  Macro F1        : {m.get('macro_f1')}")
    print(f"  Macro Precision : {m.get('macro_precision')}")
    print(f"  Macro Recall    : {m.get('macro_recall')}")
    print(f"  Weighted F1     : {m.get('weighted_f1')}")
    print(f"\n  Per-class:")
    for cls, vals in (m.get("per_class") or {}).items():
        print(f"    {cls:10s}: P={vals['precision']:.4f}  R={vals['recall']:.4f}  "
              f"F1={vals['f1']:.4f}  n={vals['support']}")
    cm_data = m.get("confusion_matrix", {})
    labels_cm = cm_data.get("labels", [])
    mat = cm_data.get("matrix", [])
    if labels_cm and mat:
        print(f"\n  Confusion matrix (rows=GT, cols=CoT prediction):")
        print(f"    {'':12s} " + "  ".join(f"{l:10s}" for l in labels_cm))
        for i, row in enumerate(mat):
            print(f"    {labels_cm[i]:12s} " + "  ".join(f"{v:10d}" for v in row))


def print_cot_cnn_summary(res: dict, title: str):
    print(f"\n{'=' * 60}")
    print(f"COT vs CNN / AGREEMENT METRICS - {title}")
    print(f"{'=' * 60}")
    m = res.get("cot_vs_cnn", {})
    if "error" in m:
        print(f"  Loi: {m['error']}")
        return
    print(f"  Tong mau: {m.get('n_total')}  (valid: {m.get('n_valid')}  "
          f"parse-failed: {m.get('n_parse_failed')})")
    print(f"  Parse failure rate: {m.get('parse_failure_rate_pct')}%")
    print(f"  Cohen's Kappa (CNN vs CoT): {m.get('cohen_kappa_cnn_vs_cot')}")
    la = m.get("label_agreement", {})
    print(f"\n  Label Agreement (CNN label == CoT label):")
    print(f"    True : {la.get('true')} ({la.get('true_pct')}%)")
    print(f"    False: {la.get('false')} ({la.get('false_pct')}%)")
    print(f"    None (CoT parse failed): {la.get('none')} ({la.get('none_pct')}%)")
    con = m.get("consensus", {})
    print(f"\n  Consensus (|mapper_level - cot_level| <= 1):")
    print(f"    True : {con.get('true')} ({con.get('true_pct')}%)")
    print(f"    False: {con.get('false')} ({con.get('false_pct')}%)")
    print(f"    None : {con.get('none')} ({con.get('none_pct')}%)")
    hc = m.get("hard_conflict", {})
    print(f"\n  Hard Conflict (label khac VA (severity_diff > 1 OR malignant-flip)):")
    print(f"    True : {hc.get('true')} ({hc.get('true_pct')}%)")
    print(f"    False: {hc.get('false')} ({hc.get('false_pct')}%)")
    print(f"    None : {hc.get('none')} ({hc.get('none_pct')}%)")
    ct = m.get("cross_tab_label_agreement_x_consensus", {})
    print(f"\n  Cross-tab label_agreement x consensus (loai None):")
    print(f"    la=True  & con=True : {ct.get('la_true_con_true')}")
    print(f"    la=True  & con=False: {ct.get('la_true_con_false')}")
    print(f"    la=False & con=True : {ct.get('la_false_con_true')}")
    print(f"    la=False & con=False: {ct.get('la_false_con_false')}")
    ta = m.get("thyroid_cot_normal_audit", {})
    if ta.get("n_thyroid_valid", 0) > 0:
        print(f"\n  [Thyroid audit] CoT chon 'normal' cho thyroid: "
              f"{ta.get('n_thyroid_cot_normal')}/{ta.get('n_thyroid_valid')} "
              f"({ta.get('rate_pct')}%)")


def print_consistency_summary(res: dict, title: str):
    if res is None:
        return
    print(f"\n{'=' * 60}")
    print(f"SELF-CONSISTENCY - {title}")
    print(f"{'=' * 60}")
    print(f"  So lan chay / mau: {res.get('n_runs_per_sample')}")
    print(f"  So mau test: {res.get('n_samples_tested')}")
    print(f"  Mean label consistency  : {res.get('mean_label_consistency_pct')}%")
    print(f"  Mean severity consistency: {res.get('mean_severity_consistency_pct')}%")
    if res.get("mean_label_consistency_pct") is not None:
        if res["mean_label_consistency_pct"] < 80:
            print("  [canh bao] Consistency < 80%: tat ca so lieu agreement/consensus co the khong tin cay.")


def main():
    parser = argparse.ArgumentParser(description="Giai doan 2 - Danh gia CoT reasoning")
    parser.add_argument("--busi_dir",      default="data/busi/test_busi")
    parser.add_argument("--tn3k_dir",      default="data/tn3k/test_tn3k")
    parser.add_argument("--busi_ckpt",     default="models/checkpoints/mtl_effnet_fc_conv_breast.pt")
    parser.add_argument("--thyroid_ckpt",  default="models/checkpoints/mtl_effnet_fc_conv_thyroid.pt")
    parser.add_argument("--out_dir",       default="eval/results/cot")
    parser.add_argument("--device",        default=None)
    parser.add_argument("--max_busi",      type=int, default=None)
    parser.add_argument("--max_tn3k",      type=int, default=None)
    parser.add_argument("--consistency_runs", type=int, default=1,
                        help="So lan chay CoT cho self-consistency test (mac dinh: 1 = bo qua)")
    parser.add_argument("--consistency_n",   type=int, default=20,
                        help="So mau dung cho self-consistency test")
    parser.add_argument("--seed",          type=int, default=42)
    parser.add_argument("--skip_busi",     action="store_true")
    parser.add_argument("--skip_tn3k",     action="store_true")
    parser.add_argument("--rate_limit",    type=int, default=10,
                        help="So request LLM toi da / phut (sliding window 60s). "
                             "Dat 0 de tat throttling (vd backend local nhu ollama). Mac dinh: 10.")
    parser.add_argument("--max_retries",   type=int, default=3,
                        help="So lan retry khi llm_client.generate() loi (vd 429/timeout) "
                             "truoc khi cho la parse-fail. Mac dinh: 3.")
    parser.add_argument("--retry_base_delay", type=float, default=2.0,
                        help="Delay co so (giay) cho exponential backoff khi retry "
                             "(lan 1: base, lan 2: base*2, lan 3: base*4, ...). Mac dinh: 2.0.")
    parser.add_argument("--resume",        action="store_true",
                        help="Bo qua cac sample da co ket qua trong checkpoint_*.jsonl cu trong out_dir "
                             "(khong goi lai vision model / LLM cho cac sample do).")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[eval_cot] Import cac module...")
    fns = _import_modules()

    print("[eval_cot] Khoi tao LLM client (LLM_BACKEND env var)...")
    llm_client = fns["get_llm_client"]()
    if args.rate_limit and args.rate_limit > 0:
        print(
            f"[eval_cot] Bat rate-limit cho LLM client: toi da {args.rate_limit} request/phut, "
            f"max_retries={args.max_retries}, retry_base_delay={args.retry_base_delay}s."
        )
    else:
        print(
            f"[eval_cot] Rate-limit bi tat (--rate_limit 0), nhung van giu retry-backoff "
            f"(max_retries={args.max_retries})."
        )
    llm_client = RateLimitedLLMClient(
        llm_client,
        max_calls_per_minute=args.rate_limit,
        max_retries=args.max_retries,
        retry_base_delay=args.retry_base_delay,
    )

    output = {}

    if not args.skip_busi:
        busi_dir = Path(args.busi_dir)
        print(f"\n[eval_cot] BUSI - Thu muc: {busi_dir}")
        if not busi_dir.exists():
            print(f"  [warn] Khong tim thay {busi_dir}")
        else:
            print(f"  Loading checkpoint breast: {args.busi_ckpt}")
            try:
                breast_model, breast_cfg = fns["load_breast"](args.busi_ckpt, args.device)
            except FileNotFoundError as e:
                print(f"  [error] {e}")
                breast_model = None
            if breast_model is not None:
                busi_samples = collect_busi_dataset(busi_dir)
                print(f"  Tong so mau BUSI: {len(busi_samples)}")
                busi_result = eval_dataset(
                    samples=busi_samples,
                    run_model_fn=fns["run_breast"],
                    model=breast_model,
                    cfg=breast_cfg,
                    fns=fns,
                    llm_client=llm_client,
                    dataset_name="BUSI",
                    valid_labels=["benign", "malignant", "normal"],
                    max_samples=args.max_busi,
                    seed=args.seed,
                    consistency_runs=args.consistency_runs,
                    consistency_n=args.consistency_n,
                    out_dir=out_dir,
                    resume=args.resume,
                )
                output["busi"] = busi_result
                print_cot_gt_summary(busi_result, "BUSI (breast, 3-lop)")
                print_cot_cnn_summary(busi_result, "BUSI (breast, 3-lop)")
                print_consistency_summary(busi_result.get("self_consistency"), "BUSI")

    if not args.skip_tn3k:
        tn3k_dir = Path(args.tn3k_dir)
        print(f"\n[eval_cot] TN3K - Thu muc: {tn3k_dir}")
        if not tn3k_dir.exists():
            print(f"  [warn] Khong tim thay {tn3k_dir}")
        else:
            print(f"  Loading checkpoint thyroid: {args.thyroid_ckpt}")
            try:
                thyroid_model, thyroid_cfg = fns["load_thyroid"](args.thyroid_ckpt, args.device)
            except FileNotFoundError as e:
                print(f"  [error] {e}")
                thyroid_model = None
            if thyroid_model is not None:
                tn3k_samples = collect_tn3k_dataset(tn3k_dir)
                print(f"  Tong so mau TN3K: {len(tn3k_samples)}")
                tn3k_result = eval_dataset(
                    samples=tn3k_samples,
                    run_model_fn=fns["run_thyroid"],
                    model=thyroid_model,
                    cfg=thyroid_cfg,
                    fns=fns,
                    llm_client=llm_client,
                    dataset_name="TN3K",
                    valid_labels=["benign", "malignant"],
                    max_samples=args.max_tn3k,
                    seed=args.seed,
                    consistency_runs=args.consistency_runs,
                    consistency_n=args.consistency_n,
                    out_dir=out_dir,
                    resume=args.resume,
                )
                output["tn3k"] = tn3k_result
                print_cot_gt_summary(tn3k_result, "TN3K (thyroid, 2-lop)")
                print_cot_cnn_summary(tn3k_result, "TN3K (thyroid, 2-lop)")
                print_consistency_summary(tn3k_result.get("self_consistency"), "TN3K")

    if output:
        records_path = out_dir / "cot_eval_records.json"
        output_clean = {}
        for ds, res in output.items():
            res_no_records = {k: v for k, v in res.items() if k != "per_sample_records"}
            output_clean[ds] = res_no_records
        summary_path = out_dir / "cot_eval_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(output_clean, f, indent=2, ensure_ascii=False)
        print(f"\n[eval_cot] Summary luu tai: {summary_path}")
        all_records = {}
        for ds, res in output.items():
            all_records[ds] = res.get("per_sample_records", [])
        with open(records_path, "w", encoding="utf-8") as f:
            json.dump(all_records, f, indent=2, ensure_ascii=False)
        print(f"[eval_cot] Per-sample records luu tai: {records_path}")
    else:
        print("\n[eval_cot] Khong co ket qua. Kiem tra lai --busi_dir va --tn3k_dir.")

    print("\n[eval_cot] Hoan thanh.\n")


if __name__ == "__main__":
    main()