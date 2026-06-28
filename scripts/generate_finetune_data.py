"""
scripts/generate_finetune_data.py
====================================
Giai doan 2.5b - Sinh synthetic training data cho fine-tune Qwen tu Gemini teacher.

Dung train split (KHONG dung test split, test split danh rieng cho eval_cot.py):
  data/busi/train_busi/
    benign/        benign (X).png, benign (X)_mask.png
    malignant/     malignant (X).png, malignant (X)_mask.png
    normal/        normal (X).png, normal (X)_mask.png (co the khong co mask)
  data/tn3k/train_tn3k/
    train-image/   001.jpg ...
    train-mask/    001.png ...
    label4train.csv  <- KHONG co header: cot 0 = ten file, cot 1 = nhan so (0=benign, 1=malignant)

Pipeline cho moi sample (dung lai dung cac ham noi bo, giong eval_cot.py):
  1. run_breast / run_thyroid (vision model)          -> model_output
  2. derive_spatial (tu mask GT neu co)                -> spatial
  3. interpret_visual_features                          -> visual_features
  4. _make_mask_overlay + describe_image (Gemini Vision) -> birads_description
  5. Goi Gemini text, ep tra ve cot_label = gt_label    -> teacher reasoning JSON
  6. Validate JSON parse duoc va cot_label khop gt_label, skip neu khong khop
  7. Ghi ngay 1 dong JSONL (ChatML) ra file, ho tro --resume

Chay:
  python scripts/generate_finetune_data.py \\
    --busi_train_dir   data/busi/train_busi \\
    --tn3k_train_dir   data/tn3k/train_tn3k \\
    --busi_ckpt        models/checkpoints/mtl_effnet_fc_conv_breast.pt \\
    --thyroid_ckpt     models/checkpoints/mtl_effnet_fc_conv_thyroid.pt \\
    --out_file         scripts/finetune_data/cot_training.jsonl \\
    [--max_busi        N] \\
    [--max_tn3k        N] \\
    [--device          cpu|cuda] \\
    [--rate_limit      10] \\
    [--max_retries     3] \\
    [--retry_base_delay 2.0] \\
    [--resume]

LLM_BACKEND phai la "google" -- script can Gemini Vision (BI-RADS/TI-RADS) va
Gemini text de sinh reasoning chain, khong dung duoc voi backend khac.
"""
import argparse
import csv
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Script nay chay standalone (khong qua docker-compose), nen .env KHONG duoc
# tu doc nhu khi chay trong container. Phai tu load o day, neu khong
# os.getenv("LLM_BACKEND")/GOOGLE_API_KEY luon la None/rong du .env co ghi gi.
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[1] / ".env")


# ---------------------------------------------------------------------------
# Rate limiting + retry cho LLM client (giong eval_cot.py)
# ---------------------------------------------------------------------------

class RateLimitedLLMClient:
    """
    Wrap llm_client, dam bao khong goi .generate() qua N lan/phut.
    Dung sliding-window 60s, cong them retry-with-exponential-backoff.
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
            return
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

    def generate(self, *args, **kwargs):
        last_exc = None
        for attempt in range(self._max_retries + 1):
            self._wait_for_slot()
            try:
                return self._inner.generate(*args, **kwargs)
            except Exception as e:  # noqa: BLE001 - bat moi loai loi tu API/network
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
        raise last_exc

    def generate_with_image(self, *args, **kwargs):
        """Goi truc tiep, khong rate-limit -- Gemini Vision dung quota rieng."""
        return self._inner.generate_with_image(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._inner, name)


# ---------------------------------------------------------------------------
# Data loaders cho train split (cau truc giong test split trong eval_cot.py)
# ---------------------------------------------------------------------------

def collect_busi_train(busi_train_dir: Path, max_n: Optional[int] = None) -> list:
    """
    Thu thap anh BUSI tu train_busi/benign, malignant, normal.
    Tra ve list dict: {image_bytes, gt_label, mask_bytes (co the None), image_path, organ, modality}.
    """
    samples = []
    for gt_label in ("benign", "malignant", "normal"):
        cls_dir = busi_train_dir / gt_label
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
                "modality": "ultrasound",
                "dataset": "busi_train",
            })
            if max_n and len(samples) >= max_n:
                return samples
    return samples


def collect_tn3k_train(tn3k_train_dir: Path, max_n: Optional[int] = None) -> list:
    """
    Thu thap anh TN3K tu train_tn3k/train-image va label4train.csv.
    label4train.csv KHONG co header: cot 0 = ten file, cot 1 = nhan so (0=benign, 1=malignant).
    Tra ve list dict cung cau truc voi collect_busi_train.
    """
    csv_path = tn3k_train_dir / "label4train.csv"
    image_dir = tn3k_train_dir / "train-image"
    mask_dir = tn3k_train_dir / "train-mask"
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
        print("  [warn] label4train.csv doc duoc 0 dong hop le, kiem tra lai dinh dang file.")
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
        mask_candidates = [mask_dir / f"{p.stem}.png", mask_dir / p.name]
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
            "modality": "ultrasound",
            "dataset": "tn3k_train",
        })
        if max_n and len(samples) >= max_n:
            break
    if n_missing > 0:
        print(f"  [warn] {n_missing} anh khong tim duoc nhan trong CSV, bo qua.")
    return samples


def _decode_mask_for_derive(mask_bytes: Optional[bytes], original_size: list) -> Optional[str]:
    """Encode mask_bytes (PNG) sang base64 de truyen vao derive_spatial."""
    if mask_bytes is None:
        return None
    import base64
    import cv2
    import numpy as np

    arr = np.frombuffer(mask_bytes, dtype=np.uint8)
    mask = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    ok, enc = cv2.imencode(".png", mask)
    if not ok:
        return None
    return base64.b64encode(enc.tobytes()).decode("ascii")


def _empty_spatial(original_size: list) -> dict:
    """Fallback khi khong co mask -- giu nguyen shape voi derive_spatial thuc te."""
    return {
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


# ---------------------------------------------------------------------------
# Checkpoint / resume
# ---------------------------------------------------------------------------

def _load_done_paths(out_file: Path) -> set:
    """Doc lai out_file JSONL cu (neu co) -> set cac image_path da xu ly xong."""
    done = set()
    if not out_file.exists():
        return done
    with open(out_file, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  [warn] {out_file} dong {line_no} loi JSON: {e} -> bo qua dong nay")
                continue
            img_path = (rec.get("metadata") or {}).get("image_path")
            if img_path:
                done.add(img_path)
    return done


def _append_jsonl(out_file: Path, record: dict):
    """Ghi ngay 1 record ra file JSONL (append, flush ngay) de ho tro --resume."""
    with open(out_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


# ---------------------------------------------------------------------------
# Sinh 1 training record cho 1 sample
# ---------------------------------------------------------------------------

def _generate_one_record(
    sample: dict,
    run_model_fn,
    model,
    cfg,
    fns: dict,
    llm_client,
) -> Optional[dict]:
    """
    Chay vision + spatial + visual_features + BI-RADS vision + Gemini teacher
    cho 1 sample, tra ve 1 ChatML record hoac None neu validate khong qua.
    """
    organ = sample["organ"]
    modality = sample["modality"]
    gt_label = sample["gt_label"]
    image_path = sample["image_path"]

    try:
        mo = run_model_fn(model, cfg, sample["image_bytes"])
    except Exception as e:
        print(f"  [skip] {image_path}: vision_inference loi: {e}")
        return None

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
            spatial = _empty_spatial(original_size)
    except Exception as e:
        print(f"  [warn] derive_spatial loi ({image_path}): {e} -- dung empty_spatial")
        spatial = _empty_spatial(original_size)

    visual_features = fns["interpret_visual_features"](
        bottleneck=mo.get("bottleneck_enriched", {}),
        texture=mo.get("texture_features", {}),
        uncertainty=mo.get("uncertainty", {}),
        gradcam_overlap=mo.get("gradcam_mask_overlap", {}),
        spatial=spatial,
        organ=organ,
    )

    try:
        overlay_bytes = fns["make_mask_overlay"](sample["image_bytes"], mask_b64 or "")
        birads_description = fns["describe_image"](
            image_bytes=overlay_bytes,
            llm_client=llm_client,
            modality=modality,
            organ=organ,
        )
    except Exception as e:
        print(f"  [warn] BI-RADS vision loi ({image_path}): {e} -- tiep tuc khong co visual description")
        birads_description = None

    user_prompt = fns["_build_cot_prompt"](
        spatial=spatial,
        visual_features=visual_features,
        rag_chunks=[],
        organ=organ,
        modality=modality,
        birads_description=birads_description,
    )

    teacher_prompt = f"""{user_prompt}

The ground truth diagnosis for this case is: {gt_label}
Write the step-by-step clinical reasoning so that cot_label in your JSON output
equals exactly "{gt_label}". Follow the same JSON schema described above."""

    try:
        raw = llm_client.generate(teacher_prompt, fns["COT_SYSTEM_PROMPT"])
        clean = raw.strip().replace("```json", "").replace("```", "").strip()
        parsed = json.loads(clean)
        if str(parsed.get("cot_label", "")).lower() != gt_label.lower():
            print(
                f"  [skip] {image_path}: cot_label='{parsed.get('cot_label')}' "
                f"!= gt_label='{gt_label}'"
            )
            return None
    except Exception as e:
        print(f"  [skip] {image_path}: {type(e).__name__}: {e}")
        return None

    return {
        "messages": [
            {"role": "system", "content": fns["COT_SYSTEM_PROMPT"]},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": json.dumps(parsed, ensure_ascii=False)},
        ],
        "metadata": {
            "organ": organ,
            "gt_label": gt_label,
            "dataset": sample["dataset"],
            "image_path": image_path,
        },
    }


def _import_modules():
    """Lazy import cac module noi bo can thiet, giong eval_cot.py."""
    from services.vision.us_breast.model import (
        load_model as load_breast,
        run_inference as run_breast,
    )
    from services.vision.us_thyroid.model import (
        load_model as load_thyroid,
        run_inference as run_thyroid,
    )
    from services.knowledge.mapper import derive_spatial
    from services.orchestrator.visual_interpreter import interpret_visual_features
    from services.orchestrator.llm_client import get_llm_client
    from services.orchestrator.graph import _build_cot_prompt, COT_SYSTEM_PROMPT
    from services.orchestrator.birads_describer import (
        describe_image,
        _make_mask_overlay,
    )
    return {
        "load_breast": load_breast,
        "run_breast": run_breast,
        "load_thyroid": load_thyroid,
        "run_thyroid": run_thyroid,
        "derive_spatial": derive_spatial,
        "interpret_visual_features": interpret_visual_features,
        "get_llm_client": get_llm_client,
        "_build_cot_prompt": _build_cot_prompt,
        "COT_SYSTEM_PROMPT": COT_SYSTEM_PROMPT,
        "describe_image": describe_image,
        "make_mask_overlay": _make_mask_overlay,
    }


def _run_split(
    samples: list,
    run_model_fn,
    model,
    cfg,
    fns: dict,
    llm_client,
    out_file: Path,
    done_paths: set,
    split_name: str,
):
    n_total = len(samples)
    n_skipped_resume = 0
    n_written = 0
    n_dropped = 0
    for i, sample in enumerate(samples, start=1):
        if sample["image_path"] in done_paths:
            n_skipped_resume += 1
            continue
        print(f"  [{split_name} {i}/{n_total}] {sample['image_path']}")
        record = _generate_one_record(sample, run_model_fn, model, cfg, fns, llm_client)
        if record is None:
            n_dropped += 1
            continue
        _append_jsonl(out_file, record)
        n_written += 1
    print(
        f"  [{split_name}] xong: {n_written} ghi moi, {n_dropped} bi loai (parse/label mismatch), "
        f"{n_skipped_resume} bo qua (--resume)."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Giai doan 2.5b - Sinh synthetic training data tu Gemini teacher"
    )
    parser.add_argument("--busi_train_dir", default="data/busi/train_busi")
    parser.add_argument("--tn3k_train_dir", default="data/tn3k/train_tn3k")
    parser.add_argument("--busi_ckpt", default="models/checkpoints/mtl_effnet_fc_conv_breast.pt")
    parser.add_argument("--thyroid_ckpt", default="models/checkpoints/mtl_effnet_fc_conv_thyroid.pt")
    parser.add_argument("--out_file", default="scripts/finetune_data/cot_training.jsonl")
    parser.add_argument("--device", default=None)
    parser.add_argument("--max_busi", type=int, default=None)
    parser.add_argument("--max_tn3k", type=int, default=None)
    parser.add_argument("--skip_busi", action="store_true")
    parser.add_argument("--skip_tn3k", action="store_true")
    parser.add_argument(
        "--rate_limit", type=int, default=10,
        help="So request LLM text toi da / phut. Dat 0 de tat throttling. Mac dinh: 10.",
    )
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--retry_base_delay", type=float, default=2.0)
    parser.add_argument(
        "--resume", action="store_true",
        help="Bo qua sample da co trong --out_file cu (theo image_path), khong goi lai LLM cho sample do.",
    )
    args = parser.parse_args()

    out_file = Path(args.out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    backend = os.getenv("LLM_BACKEND", "ollama").lower()
    if backend != "google":
        print(
            f"[generate_finetune_data] WARNING: LLM_BACKEND='{backend}' nhung script can "
            "Gemini Vision + Gemini text de sinh teacher data. Dat LLM_BACKEND=google."
        )

    print("[generate_finetune_data] Import cac module...")
    fns = _import_modules()

    print("[generate_finetune_data] Khoi tao LLM client...")
    llm_client = fns["get_llm_client"]()
    llm_client = RateLimitedLLMClient(
        llm_client,
        max_calls_per_minute=args.rate_limit,
        max_retries=args.max_retries,
        retry_base_delay=args.retry_base_delay,
    )

    done_paths = _load_done_paths(out_file) if args.resume else set()
    if args.resume and done_paths:
        print(f"[generate_finetune_data] --resume: {len(done_paths)} sample da co trong {out_file}, se bo qua.")

    if not args.skip_busi:
        busi_dir = Path(args.busi_train_dir)
        print(f"\n[generate_finetune_data] BUSI train - Thu muc: {busi_dir}")
        if not busi_dir.exists():
            print(f"  [warn] Khong tim thay {busi_dir}")
        else:
            breast_model, breast_cfg = fns["load_breast"](args.busi_ckpt, args.device)
            busi_samples = collect_busi_train(busi_dir, max_n=args.max_busi)
            print(f"  Tong so mau BUSI train: {len(busi_samples)}")
            _run_split(
                busi_samples, fns["run_breast"], breast_model, breast_cfg,
                fns, llm_client, out_file, done_paths, "busi_train",
            )

    if not args.skip_tn3k:
        tn3k_dir = Path(args.tn3k_train_dir)
        print(f"\n[generate_finetune_data] TN3K train - Thu muc: {tn3k_dir}")
        if not tn3k_dir.exists():
            print(f"  [warn] Khong tim thay {tn3k_dir}")
        else:
            thyroid_model, thyroid_cfg = fns["load_thyroid"](args.thyroid_ckpt, args.device)
            tn3k_samples = collect_tn3k_train(tn3k_dir, max_n=args.max_tn3k)
            print(f"  Tong so mau TN3K train: {len(tn3k_samples)}")
            _run_split(
                tn3k_samples, fns["run_thyroid"], thyroid_model, thyroid_cfg,
                fns, llm_client, out_file, done_paths, "tn3k_train",
            )

    print(f"\n[generate_finetune_data] Hoan tat. Output: {out_file}")
    print("[generate_finetune_data] Upload file JSONL nay len Google Drive / Kaggle Dataset truoc khi fine-tune.")


if __name__ == "__main__":
    main()