"""
eval/eval_vision.py
====================
Giai doan 1 buoc 2 - Danh gia Vision CNN (UNet_MTL - EfficientNet-B4).

Chay offline, khong can Docker. Load model truc tiep tu checkpoint.

Ket qua cho BUSI (breast, 3-lop):
  - Segmentation: Dice, IoU tren tap test co mask GT.
    Anh "normal" (khong co ton thuong) duoc bao cao rieng - khong gop vao
    trung binh Dice/IoU chinh (vi mask GT rong va mask du doan cung rong
    la "dung" theo nghia khac, lam so tang gia).
  - Classification: confusion matrix 3x3, macro-F1, Precision/Recall per-class.
    Dung macro-F1 vi BUSI mat can bang (malignant it mau hon benign nhieu).
  - Inference time: chi do forward-only (XAI timing bi bo qua theo mac dinh).

Ket qua cho TN3K (thyroid, 2-lop: benign/malignant, KHONG co "normal"):
  - Segmentation: Dice, IoU tuong tu BUSI (khong co anh normal -> moi anh
    deu co lesion, nen khong can loai mau).
  - Classification: confusion matrix 2x2, binary F1/Precision/Recall.
  - Inference time: tuong tu BUSI.

Parameters/FLOPs:
  - Do 1 lan cho UNet_MTL breast (EfficientNet-B4 backbone + decoder).
  - Thyroid co cung kien truc, chi khac NUM_CLASSES o cls_head cuoi
    (3 vs 2) -> parameters gan nhu giong nhau, chi can do 1 lan va ghi chu.

Cau truc thu muc can thiet:

  BUSI:
    data/busi/test_busi/
      benign/
        benign (1).png
        benign (1)_mask.png
        benign (2).png
        benign (2)_mask.png
        ...
      malignant/
        malignant (1).png
        malignant (1)_mask.png
        ...
      normal/
        normal (1).png
        normal (1)_mask.png   <- co the rong (all-zero), hoac khong ton tai
        ...

  TN3K:
    data/tn3k/test_tn3k/
      test-image/
        001.jpg
        002.jpg
        ...
      test-mask/
        001.png
        002.png
        ...
      label4test.csv        <- cot "image_name", cot "label" (0=benign, 1=malignant)

Chay:
  python eval/eval_vision.py \\
    --busi_dir     data/busi/test_busi \\
    --tn3k_dir     data/tn3k/test_tn3k \\
    --busi_ckpt    models/checkpoints/mtl_effnet_fc_conv_breast.pt \\
    --thyroid_ckpt models/checkpoints/mtl_effnet_fc_conv_thyroid.pt \\
    --out_dir      eval/results/vision \\
    [--device      cpu|cuda] \\
    [--run_xai_timing]    # them flag nay moi do XAI timing (mac dinh: bo qua)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


def _import_vision_modules():
    """
    Import lazily de tranh loi khi timm/torch chua cai.
    Tra ve dict chua cac ham load/run cho breast va thyroid.
    """
    from services.vision.us_breast.model import (
        load_model as load_breast,
        run_inference as run_breast,
    )
    from services.vision.us_thyroid.model import (
        load_model as load_thyroid,
        run_inference as run_thyroid,
    )
    from services.vision._vision_helpers import (
        _compute_gradcam,
        _compute_gradcam_mask_overlap,
        _extract_texture_features,
        _predict_with_uncertainty,
    )
    return {
        "load_breast": load_breast,
        "run_breast": run_breast,
        "load_thyroid": load_thyroid,
        "run_thyroid": run_thyroid,
        "_compute_gradcam": _compute_gradcam,
        "_compute_gradcam_mask_overlap": _compute_gradcam_mask_overlap,
        "_extract_texture_features": _extract_texture_features,
        "_predict_with_uncertainty": _predict_with_uncertainty,
    }


def _dice_iou(pred_bin: np.ndarray, gt_bin: np.ndarray) -> tuple:
    """
    Tinh Dice va IoU giua 2 mask nhi phan (gia tri 0 hoac 1).
    Tra ve (dice, iou).
    """
    inter = float((pred_bin & gt_bin).sum())
    union = float((pred_bin | gt_bin).sum())
    pred_sum = float(pred_bin.sum())
    gt_sum = float(gt_bin.sum())

    dice = (2.0 * inter) / (pred_sum + gt_sum + 1e-8)
    iou = inter / (union + 1e-8)
    return round(dice, 4), round(iou, 4)


def _load_image_bytes(path: Path) -> bytes:
    return path.read_bytes()


def _load_gt_mask_binary(path: Path) -> Optional[np.ndarray]:
    """
    Doc mask GT tu file PNG/JPG, tra ve array nhi phan (0/1).
    Tra ve None neu file khong ton tai.
    """
    if not path.exists():
        return None
    arr = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    mask = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    return (mask > 127).astype(np.uint8)


def _decode_pred_mask_binary(mask_png_base64: str, original_size: tuple) -> np.ndarray:
    """
    Giai ma mask du doan (base64 PNG) -> array nhi phan kich thuoc original_size.
    Threshold cung voi gia tri 0.5 nhu trong model.py (mask_np = ... > 0.5).
    Mask da duoc threshold khi encode (gia tri 0 hoac 255), nen > 127 la du.
    """
    import base64
    mask_bytes = base64.b64decode(mask_png_base64)
    arr = np.frombuffer(mask_bytes, dtype=np.uint8)
    mask = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        H, W = original_size
        return np.zeros((H, W), dtype=np.uint8)
    if (mask.shape[0], mask.shape[1]) != (original_size[0], original_size[1]):
        mask = cv2.resize(mask, (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST)
    return (mask > 127).astype(np.uint8)


def collect_busi_dataset(busi_dir: Path) -> list:
    """
    Thu thap tap test BUSI tu cau truc data/busi/test_busi/.

    Qui uoc ten file goc: "benign (1).png" -> mask: "benign (1)_mask.png".
    Chi thu thap anh goc (khong co "_mask" trong ten).

    Tra ve list of dict:
        image_path, mask_path (co the None), gt_label (str), is_normal (bool)
    """
    samples = []
    img_exts = {".png", ".jpg", ".jpeg", ".bmp"}
    label_dirs = {"benign": "benign", "malignant": "malignant", "normal": "normal"}

    for label_str, subdir in label_dirs.items():
        cls_dir = busi_dir / subdir
        if not cls_dir.exists():
            print(f"  [warn] BUSI: khong tim thay thu muc {cls_dir}, bo qua")
            continue
        for p in sorted(cls_dir.iterdir()):
            if p.suffix.lower() not in img_exts:
                continue
            if "_mask" in p.stem:
                continue
            mask_path = p.parent / f"{p.stem}_mask{p.suffix}"
            samples.append({
                "image_path": p,
                "mask_path": mask_path if mask_path.exists() else None,
                "gt_label": label_str,
                "is_normal": label_str == "normal",
            })

    return samples


def collect_tn3k_dataset(tn3k_dir: Path) -> list:
    """
    Thu thap tap test TN3K tu cau truc data/tn3k/test_tn3k/.

    tn3k_dir truyen vao phai la thu muc test_tn3k (da chua test-image/, test-mask/, label4test.csv).
    label4test.csv chua it nhat 2 cot: ten file anh va nhan (0=benign, 1=malignant).

    Tra ve list of dict:
        image_path, mask_path (co the None), gt_label (str: "benign"|"malignant")
    """
    import csv

    img_dir = tn3k_dir / "test-image"
    mask_dir = tn3k_dir / "test-mask"
    label_csv = tn3k_dir / "label4test.csv"

    if not img_dir.exists():
        print(f"  [warn] TN3K: khong tim thay {img_dir}")
        return []
    if not label_csv.exists():
        print(f"  [warn] TN3K: khong tim thay {label_csv}, se su dung ten thu muc thay the")
        return _collect_tn3k_no_csv(img_dir, mask_dir)

    idx_to_label = {0: "benign", 1: "malignant"}

    samples = []
    with open(label_csv, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)

        name_col, label_col = 0, 1
        if header:
            h_lower = [h.strip().lower() for h in header]
            for possible in ["image_name", "filename", "name", "file"]:
                if possible in h_lower:
                    name_col = h_lower.index(possible)
                    break
            for possible in ["label", "class", "category"]:
                if possible in h_lower:
                    label_col = h_lower.index(possible)
                    break

        for row in reader:
            if len(row) <= max(name_col, label_col):
                continue
            fname = row[name_col].strip()
            try:
                lbl_int = int(row[label_col].strip())
            except ValueError:
                lbl_str = row[label_col].strip().lower()
                lbl_int = 0 if lbl_str == "benign" else 1

            img_path = img_dir / fname
            if not img_path.exists():
                for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
                    candidate = img_dir / (Path(fname).stem + ext)
                    if candidate.exists():
                        img_path = candidate
                        break

            if not img_path.exists():
                print(f"  [warn] TN3K: khong tim thay anh {fname}, bo qua")
                continue

            stem = Path(fname).stem
            mask_path = None
            for ext in [".png", ".jpg", ".jpeg"]:
                candidate = mask_dir / (stem + ext)
                if candidate.exists():
                    mask_path = candidate
                    break

            samples.append({
                "image_path": img_path,
                "mask_path": mask_path,
                "gt_label": idx_to_label.get(lbl_int, "benign"),
                "is_normal": False,
            })

    return samples


def _collect_tn3k_no_csv(img_dir: Path, mask_dir: Path) -> list:
    """
    Fallback khi khong co label4test.csv: thu thap anh va mask theo ten file.
    Khong the gan nhan -> tra ve list rong, phan classification se bi bo qua.
    """
    print("  [warn] TN3K: khong co label4test.csv, chi co the do segmentation")
    img_exts = {".png", ".jpg", ".jpeg", ".bmp"}
    samples = []
    for p in sorted(img_dir.iterdir()):
        if p.suffix.lower() not in img_exts:
            continue
        mask_path = None
        if mask_dir.exists():
            for ext in [".png", ".jpg", ".jpeg"]:
                candidate = mask_dir / (p.stem + ext)
                if candidate.exists():
                    mask_path = candidate
                    break
        samples.append({
            "image_path": p,
            "mask_path": mask_path,
            "gt_label": None,
            "is_normal": False,
        })
    return samples


def eval_segmentation(samples: list, run_inference_fn, model, cfg, dataset_name: str) -> dict:
    """
    Danh gia dau ra segmentation tren toan bo tap test.

    BUSI co anh "normal" (is_normal=True) -> tinh Dice/IoU rieng va LOAI khoi
    trung binh chinh. TN3K khong co normal -> moi mau deu duoc tinh.

    Tra ve dict chua:
        dice_mean, iou_mean (chi tinh tren anh co lesion)
        per_class_seg (BUSI), normal_seg_summary (BUSI)
    """
    dice_lesion, iou_lesion = [], []
    dice_normal, iou_normal = [], []
    normal_empty_gt_empty_pred = 0
    normal_total = 0
    per_class_dice = {}
    per_class_iou = {}
    seg_errors = 0

    for s in samples:
        img_bytes = _load_image_bytes(s["image_path"])
        gt_mask = _load_gt_mask_binary(s["mask_path"]) if s["mask_path"] else None

        try:
            result = run_inference_fn(model=model, cfg=cfg, image_bytes=img_bytes)
        except Exception as e:
            print(f"    [seg error] {s['image_path'].name}: {e}")
            seg_errors += 1
            continue

        pred_bin = _decode_pred_mask_binary(
            result["mask_png_base64"],
            tuple(result["original_size"]),
        )

        if gt_mask is None:
            continue

        if gt_mask.shape != pred_bin.shape:
            gt_mask = cv2.resize(
                gt_mask.astype(np.uint8),
                (pred_bin.shape[1], pred_bin.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        dice, iou = _dice_iou(pred_bin, gt_mask.astype(np.uint8))
        lbl = s.get("gt_label")

        if s.get("is_normal", False):
            normal_total += 1
            gt_empty = gt_mask.sum() == 0
            pred_empty = pred_bin.sum() == 0
            if gt_empty and pred_empty:
                normal_empty_gt_empty_pred += 1
            dice_normal.append(dice)
            iou_normal.append(iou)
        else:
            dice_lesion.append(dice)
            iou_lesion.append(iou)
            if lbl:
                per_class_dice.setdefault(lbl, []).append(dice)
                per_class_iou.setdefault(lbl, []).append(iou)

    def _agg(vals: list) -> dict:
        if not vals:
            return {"mean": None, "median": None, "std": None, "n": 0}
        a = np.array(vals)
        return {
            "mean":   round(float(a.mean()), 4),
            "median": round(float(np.median(a)), 4),
            "std":    round(float(a.std()), 4),
            "n":      len(vals),
        }

    result_dict = {
        "dataset": dataset_name,
        "seg_errors": seg_errors,
        "lesion_samples": {
            "dice": _agg(dice_lesion),
            "iou":  _agg(iou_lesion),
        },
    }

    if per_class_dice:
        result_dict["per_class_seg"] = {
            cls: {"dice": _agg(per_class_dice[cls]), "iou": _agg(per_class_iou.get(cls, []))}
            for cls in per_class_dice
        }

    if normal_total > 0:
        result_dict["normal_samples"] = {
            "total": normal_total,
            "both_empty_rate": round(normal_empty_gt_empty_pred / normal_total, 4),
            "dice": _agg(dice_normal),
            "iou":  _agg(iou_normal),
            "note": (
                "Dice/IoU cho anh normal khong co y nghia diagnostic ro rang. "
                "Su dung 'both_empty_rate' de danh gia kha nang nhan biet anh khong co ton thuong."
            ),
        }

    return result_dict


def eval_classification(samples: list, run_inference_fn, model, cfg, dataset_name: str) -> dict:
    """
    Danh gia dau ra classification tren tap test.

    BUSI: 3-lop (benign/malignant/normal), dung macro-F1.
    TN3K: 2-lop (benign/malignant), dung binary F1.

    Loai cac mau co gt_label=None (khong co nhan).
    """
    labeled = [s for s in samples if s.get("gt_label") is not None]
    if not labeled:
        return {"error": "Khong co mau nao co nhan (gt_label=None)", "dataset": dataset_name}

    all_labels = sorted({s["gt_label"] for s in labeled})
    label_to_idx = {l: i for i, l in enumerate(all_labels)}

    y_true, y_pred = [], []
    cls_errors = 0

    for s in labeled:
        img_bytes = _load_image_bytes(s["image_path"])
        try:
            result = run_inference_fn(model=model, cfg=cfg, image_bytes=img_bytes)
        except Exception as e:
            print(f"    [cls error] {s['image_path'].name}: {e}")
            cls_errors += 1
            continue

        pred_label = result["top_label"]
        if pred_label not in label_to_idx:
            # Model tra ve nhan khong co trong tap nhan GT -> gan vao lop dau tien
            pred_label = all_labels[0]

        y_true.append(label_to_idx[s["gt_label"]])
        y_pred.append(label_to_idx[pred_label])

    if not y_true:
        return {"error": "Tat ca mau bi loi khi inference", "dataset": dataset_name}

    average_mode = "binary" if len(all_labels) == 2 else "macro"
    report = classification_report(
        y_true, y_pred,
        labels=list(range(len(all_labels))),
        target_names=all_labels,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(all_labels)))).tolist()
    acc = accuracy_score(y_true, y_pred)

    avg_key = "macro avg" if average_mode == "macro" else "binary"
    if avg_key not in report:
        avg_key = "macro avg"

    per_class_out = {}
    for cls in all_labels:
        if cls in report:
            per_class_out[cls] = {
                "precision": round(report[cls]["precision"], 4),
                "recall":    round(report[cls]["recall"], 4),
                "f1":        round(report[cls]["f1-score"], 4),
                "support":   int(report[cls]["support"]),
            }

    return {
        "dataset": dataset_name,
        "n_samples": len(y_true),
        "cls_errors": cls_errors,
        "labels": all_labels,
        "accuracy": round(acc, 4),
        "averaging": average_mode,
        f"{average_mode}_f1":        round(report[avg_key]["f1-score"], 4),
        f"{average_mode}_precision": round(report[avg_key]["precision"], 4),
        f"{average_mode}_recall":    round(report[avg_key]["recall"], 4),
        "per_class": per_class_out,
        "confusion_matrix": {
            "labels": all_labels,
            "matrix": cm,
        },
    }


def measure_inference_time(
    samples: list,
    run_inference_fn,
    model,
    cfg,
    fns: dict,
    n_samples: int = 20,
    skip_xai: bool = True,
) -> dict:
    """
    Do thoi gian inference theo 2 che do:
      - forward_only: chi 1 forward pass (seg + cls), DISABLE_XAI=true tuong duong.
      - with_xai:     forward + Grad-CAM (1 backward) + 10x MC-Dropout forward.

    Lay ngau nhien toi da n_samples mau de do, uu tien mau co lesion.
    Tra ve dict chua latency stats (ms) cho ca 2 che do.
    """
    import random
    rng = random.Random(42)
    pool = [s for s in samples if not s.get("is_normal", False)] or samples
    chosen = rng.sample(pool, min(n_samples, len(pool)))

    latency_forward = []
    latency_xai = []

    for s in chosen:
        img_bytes = _load_image_bytes(s["image_path"])
        img_arr = np.frombuffer(img_bytes, dtype=np.uint8)
        img_bgr = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            continue
        original_size = (img_bgr.shape[0], img_bgr.shape[1])

        from torchvision import transforms
        from PIL import Image
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        transform = transforms.Compose([
            transforms.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg.MEAN, std=cfg.STD),
        ])
        tensor = transform(pil_img).unsqueeze(0).to(cfg.DEVICE)

        # Forward-only (no grad, no XAI)
        t0 = time.perf_counter()
        with torch.no_grad():
            seg_out, cls_out, bot_out = model(tensor)
        latency_forward.append((time.perf_counter() - t0) * 1000)

        if skip_xai:
            continue

        # Forward + XAI (Grad-CAM + MC-Dropout)
        top_idx = int(F.softmax(cls_out, dim=1).squeeze(0).argmax())
        t1 = time.perf_counter()
        try:
            fns["_compute_gradcam"](model, tensor, top_idx, original_size)
            fns["_predict_with_uncertainty"](model, tensor, cfg, n_passes=10)
        except Exception:
            pass
        latency_xai.append((time.perf_counter() - t1) * 1000 + (t1 - t0) * 1000)

    def _stats(vals: list) -> dict:
        if not vals:
            return None
        a = np.array(vals)
        return {
            "mean_ms":   round(float(a.mean()), 2),
            "median_ms": round(float(np.median(a)), 2),
            "p95_ms":    round(float(np.percentile(a, 95)), 2),
            "min_ms":    round(float(a.min()), 2),
            "max_ms":    round(float(a.max()), 2),
            "n":         len(vals),
        }

    return {
        "forward_only":         _stats(latency_forward),
        "forward_with_xai":     _stats(latency_xai) if not skip_xai else None,
        "note": (
            "forward_only tuong duong DISABLE_XAI=true (1 forward pass). "
            "forward_with_xai = 1 forward + 1 backward Grad-CAM + 10 MC-Dropout forward "
            "(hien dang chay mac dinh khi DISABLE_XAI=false)."
        ),
    }


def count_parameters(model) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total":       total,
        "trainable":   trainable,
        "total_M":     round(total / 1e6, 2),
        "trainable_M": round(trainable / 1e6, 2),
    }


def try_count_flops(model, cfg) -> Optional[dict]:
    """
    Thu tinh FLOPs bang thop (neu da cai). Tra ve None neu khong co.
    Dung input gia (1, 3, IMG_SIZE, IMG_SIZE) de tinh.
    """
    try:
        from thop import profile
        dummy = torch.zeros(1, 3, cfg.IMG_SIZE, cfg.IMG_SIZE).to(cfg.DEVICE)
        macs, params = profile(model, inputs=(dummy,), verbose=False)
        return {
            "gmacs": round(macs / 1e9, 3),
            "input_size": f"1x3x{cfg.IMG_SIZE}x{cfg.IMG_SIZE}",
            "note": "MACs tinh bang thop, 1 FLOP ~ 2 MACs theo quy uoc pho bien.",
        }
    except ImportError:
        return {"error": "thop chua duoc cai dat. Chay: pip install thop --break-system-packages"}
    except Exception as e:
        return {"error": str(e)}


def print_segmentation_summary(res: dict, dataset_name: str):
    print(f"\n{'=' * 60}")
    print(f"SEGMENTATION - {dataset_name}")
    print(f"{'=' * 60}")
    ls = res.get("lesion_samples", {})
    dice = ls.get("dice", {})
    iou = ls.get("iou", {})
    print(f"  Mau co lesion: {dice.get('n', 0)}")
    print(f"  Dice (trung binh): {dice.get('mean')}  (std={dice.get('std')})")
    print(f"  IoU  (trung binh): {iou.get('mean')}  (std={iou.get('std')})")

    if "per_class_seg" in res:
        print("\n  Per-class segmentation:")
        for cls, m in res["per_class_seg"].items():
            d = m["dice"]
            i = m["iou"]
            print(f"    {cls}: Dice={d.get('mean')} (n={d.get('n')})  IoU={i.get('mean')}")

    if "normal_samples" in res:
        n = res["normal_samples"]
        print(f"\n  Anh normal: {n['total']} mau")
        print(f"    Ca GT va Pred deu rong: {n['both_empty_rate']:.4f} ({int(n['both_empty_rate']*n['total'])}/{n['total']})")
        print(f"    Dice trung binh (tham khao): {n['dice'].get('mean')}")
        print(f"    Luu y: {n['note']}")

    if res.get("seg_errors", 0):
        print(f"\n  [warn] Loi inference (bo qua): {res['seg_errors']}")


def print_classification_summary(res: dict, dataset_name: str):
    print(f"\n{'=' * 60}")
    print(f"CLASSIFICATION - {dataset_name}")
    print(f"{'=' * 60}")
    if "error" in res:
        print(f"  [error] {res['error']}")
        return
    avg = res["averaging"]
    print(f"  Mau: {res['n_samples']}  Accuracy: {res['accuracy']:.4f}")
    print(f"  {avg}-F1:        {res.get(avg + '_f1'):.4f}")
    print(f"  {avg}-Precision: {res.get(avg + '_precision'):.4f}")
    print(f"  {avg}-Recall:    {res.get(avg + '_recall'):.4f}")
    print(f"\n  Per-class:")
    for cls, m in res.get("per_class", {}).items():
        print(f"    {cls}: P={m['precision']:.4f}  R={m['recall']:.4f}  F1={m['f1']:.4f}  n={m['support']}")
    print(f"\n  Confusion matrix (rows=true, cols=pred):")
    labels_cm = res["confusion_matrix"]["labels"]
    print(f"    {'':15s} " + "  ".join(f"{l:12s}" for l in labels_cm))
    for i, row in enumerate(res["confusion_matrix"]["matrix"]):
        print(f"    {labels_cm[i]:15s} " + "  ".join(f"{v:12d}" for v in row))
    if res.get("cls_errors", 0):
        print(f"\n  [warn] Loi inference (bo qua): {res['cls_errors']}")


def print_timing_summary(res: dict, dataset_name: str):
    print(f"\n{'=' * 60}")
    print(f"INFERENCE TIME - {dataset_name}")
    print(f"{'=' * 60}")
    fo = res.get("forward_only")
    xai = res.get("forward_with_xai")
    if fo:
        print(f"  Forward-only (DISABLE_XAI=true tuong duong):")
        print(f"    mean={fo['mean_ms']}ms  median={fo['median_ms']}ms  p95={fo['p95_ms']}ms  (n={fo['n']})")
    if xai:
        print(f"  Forward + Grad-CAM + 10x MC-Dropout (hien dang chay mac dinh):")
        print(f"    mean={xai['mean_ms']}ms  median={xai['median_ms']}ms  p95={xai['p95_ms']}ms  (n={xai['n']})")
    if fo and xai and fo["mean_ms"] and xai["mean_ms"]:
        ratio = xai["mean_ms"] / fo["mean_ms"]
        print(f"\n  Overhead XAI: ~{ratio:.1f}x so voi forward-only")


def main():
    parser = argparse.ArgumentParser(description="Danh gia Vision CNN (UNet_MTL EfficientNet-B4)")
    parser.add_argument(
        "--busi_dir", default="data/busi/test_busi",
        help="Thu muc test BUSI (chua benign/, malignant/, normal/)"
    )
    parser.add_argument(
        "--tn3k_dir", default="data/tn3k/test_tn3k",
        help="Thu muc test TN3K (chua test-image/, test-mask/, label4test.csv)"
    )
    parser.add_argument(
        "--busi_ckpt", default="models/checkpoints/mtl_effnet_fc_conv_breast.pt",
        help="Checkpoint UNet_MTL breast"
    )
    parser.add_argument(
        "--thyroid_ckpt", default="models/checkpoints/mtl_effnet_fc_conv_thyroid.pt",
        help="Checkpoint UNet_MTL thyroid"
    )
    parser.add_argument(
        "--out_dir", default="eval/results/vision",
        help="Thu muc luu ket qua JSON va bao cao text"
    )
    parser.add_argument("--device", default=None, help="'cuda' hoac 'cpu' (tu dong neu de trong)")
    parser.add_argument(
        "--run_xai_timing", action="store_true",
        help="Do them thoi gian XAI (Grad-CAM + MC-Dropout). Mac dinh: bo qua, them flag nay de bat"
    )
    parser.add_argument(
        "--timing_samples", type=int, default=20,
        help="So mau dung de do inference time (mac dinh: 20)"
    )
    parser.add_argument(
        "--skip_busi", action="store_true", help="Bo qua danh gia BUSI"
    )
    parser.add_argument(
        "--skip_tn3k", action="store_true", help="Bo qua danh gia TN3K"
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[eval_vision] Import cac module vision...")
    fns = _import_vision_modules()

    output = {}

    if not args.skip_busi:
        busi_dir = Path(args.busi_dir)
        print(f"\n[eval_vision] BUSI - Thu muc: {busi_dir}")
        if not busi_dir.exists():
            print(f"  [warn] Khong tim thay {busi_dir}. Dung --busi_dir de chi dinh dung duong dan.")
        else:
            print(f"  Loading checkpoint: {args.busi_ckpt}")
            try:
                breast_model, breast_cfg = fns["load_breast"](args.busi_ckpt, args.device)
            except FileNotFoundError as e:
                print(f"  [error] {e}")
                breast_model = None

            if breast_model is not None:
                print("  Thu thap du lieu BUSI...")
                busi_samples = collect_busi_dataset(busi_dir)
                print(f"  Tong so mau: {len(busi_samples)}")
                counts = {}
                for s in busi_samples:
                    lbl = s["gt_label"]
                    counts[lbl] = counts.get(lbl, 0) + 1
                for lbl, cnt in sorted(counts.items()):
                    print(f"    {lbl}: {cnt}")

                print("\n  Danh gia segmentation BUSI...")
                busi_seg = eval_segmentation(busi_samples, fns["run_breast"], breast_model, breast_cfg, "BUSI")

                print("  Danh gia classification BUSI...")
                busi_cls = eval_classification(busi_samples, fns["run_breast"], breast_model, breast_cfg, "BUSI")

                print(f"  Do inference time (n={args.timing_samples})...")
                busi_timing = measure_inference_time(
                    busi_samples, fns["run_breast"], breast_model, breast_cfg, fns,
                    n_samples=args.timing_samples, skip_xai=not args.run_xai_timing,
                )

                busi_params = count_parameters(breast_model)
                busi_flops = try_count_flops(breast_model, breast_cfg)

                output["busi"] = {
                    "checkpoint": args.busi_ckpt,
                    "device": breast_cfg.DEVICE,
                    "model_parameters": busi_params,
                    "flops": busi_flops,
                    "segmentation": busi_seg,
                    "classification": busi_cls,
                    "inference_time": busi_timing,
                }

                print_segmentation_summary(busi_seg, "BUSI (breast, 3-lop)")
                print_classification_summary(busi_cls, "BUSI (breast, 3-lop)")
                print_timing_summary(busi_timing, "BUSI (breast)")
                print(f"\n  Parameters: {busi_params['total_M']}M total")
                if busi_flops and "gmacs" in busi_flops:
                    print(f"  FLOPs (forward): {busi_flops['gmacs']} GMACs")
                elif busi_flops and "error" in busi_flops:
                    print(f"  FLOPs: {busi_flops['error']}")

    if not args.skip_tn3k:
        tn3k_dir = Path(args.tn3k_dir)
        print(f"\n[eval_vision] TN3K - Thu muc: {tn3k_dir}")
        if not tn3k_dir.exists():
            print(f"  [warn] Khong tim thay {tn3k_dir}. Dung --tn3k_dir de chi dinh dung duong dan.")
        else:
            print(f"  Loading checkpoint: {args.thyroid_ckpt}")
            try:
                thyroid_model, thyroid_cfg = fns["load_thyroid"](args.thyroid_ckpt, args.device)
            except FileNotFoundError as e:
                print(f"  [error] {e}")
                thyroid_model = None

            if thyroid_model is not None:
                print("  Thu thap du lieu TN3K...")
                tn3k_samples = collect_tn3k_dataset(tn3k_dir)
                print(f"  Tong so mau: {len(tn3k_samples)}")
                counts = {}
                for s in tn3k_samples:
                    lbl = s["gt_label"] or "unknown"
                    counts[lbl] = counts.get(lbl, 0) + 1
                for lbl, cnt in sorted(counts.items()):
                    print(f"    {lbl}: {cnt}")

                print("\n  Danh gia segmentation TN3K...")
                tn3k_seg = eval_segmentation(tn3k_samples, fns["run_thyroid"], thyroid_model, thyroid_cfg, "TN3K")

                print("  Danh gia classification TN3K...")
                tn3k_cls = eval_classification(tn3k_samples, fns["run_thyroid"], thyroid_model, thyroid_cfg, "TN3K")

                print(f"  Do inference time (n={args.timing_samples})...")
                tn3k_timing = measure_inference_time(
                    tn3k_samples, fns["run_thyroid"], thyroid_model, thyroid_cfg, fns,
                    n_samples=args.timing_samples, skip_xai=not args.run_xai_timing,
                )

                tn3k_params = count_parameters(thyroid_model)
                tn3k_flops = try_count_flops(thyroid_model, thyroid_cfg)

                output["tn3k"] = {
                    "checkpoint": args.thyroid_ckpt,
                    "device": thyroid_cfg.DEVICE,
                    "model_parameters": tn3k_params,
                    "flops": tn3k_flops,
                    "segmentation": tn3k_seg,
                    "classification": tn3k_cls,
                    "inference_time": tn3k_timing,
                    "note_parameters": (
                        "Thyroid UNet_MTL co cung kien truc backbone (EfficientNet-B4) voi BUSI. "
                        "Chi khac o cls_head cuoi (2 vs 3 output), nen so tham so rat gan bang BUSI."
                    ),
                }

                print_segmentation_summary(tn3k_seg, "TN3K (thyroid, 2-lop)")
                print_classification_summary(tn3k_cls, "TN3K (thyroid, 2-lop)")
                print_timing_summary(tn3k_timing, "TN3K (thyroid)")
                print(f"\n  Parameters: {tn3k_params['total_M']}M total")
                if tn3k_flops and "gmacs" in tn3k_flops:
                    print(f"  FLOPs (forward): {tn3k_flops['gmacs']} GMACs")
                elif tn3k_flops and "error" in tn3k_flops:
                    print(f"  FLOPs: {tn3k_flops['error']}")

    if output:
        results_path = out_dir / "vision_eval.json"
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"\n[eval_vision] Ket qua day du luu tai: {results_path}")
    else:
        print("\n[eval_vision] Khong co ket qua nao duoc tao ra. Kiem tra lai --busi_dir va --tn3k_dir.")

    print("\n[eval_vision] Hoan thanh.\n")


if __name__ == "__main__":
    main()
