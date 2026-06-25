"""
eval/eval_router.py
====================
Giai doan 1 - Danh gia Router (EfficientNet-B0).

Chay offline, khong can Docker. Load model truc tiep tu checkpoint.

Ket qua:
  - Bảng 1: Accuracy / F1 / Precision / Recall cho phan loai 2 lop
            (us_breast vs us_thyroid) tren anh in-distribution.
  - Bảng 2: AUROC + FPR@95TPR cho OOD detection (sweep threshold),
            bao gom diem tai threshold mac dinh OOD_THRESHOLD=0.6.
  - Confusion matrix (2x2) cho in-distribution.
  - Inference time (ms, batch=1) va tham so model.

Cau truc thu muc test_router can thiet:
  data/router/test_router/
    us_breast/    <- anh breast ultrasound (*.png / *.jpg)
    us_thyroid/   <- anh thyroid ultrasound
    ood/          <- anh ngoai phan phoi (X-ray, CT, non-medical, ...)

Chay:
  python eval/eval_router.py \
    --data_dir  data/router/test_router \
    --ckpt      models/checkpoints/router_effnet_b0.pth \
    --out_dir   eval/results/router \
    [--device   cpu|cuda]
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

from services.router.model import (
    ROUTER_CLASSES,
    OOD_THRESHOLD,
    load_router,
    run_routing,
)


IN_DIST_CLASSES = list(ROUTER_CLASSES.values())  # ['us_breast', 'us_thyroid']


def load_images_from_dir(directory: Path) -> list:
    """Returns list of (path, bytes) for all PNG/JPG in directory (non-recursive)."""
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    items = []
    for p in sorted(directory.iterdir()):
        if p.suffix.lower() in exts:
            items.append((p, p.read_bytes()))
    return items


def collect_dataset(data_dir: Path) -> tuple:
    """
    Walk data_dir expecting subdirs: us_breast/, us_thyroid/, ood/

    Returns:
        in_dist_samples:  list of (image_bytes, true_class_str)   -- breast/thyroid only
        ood_samples:      list of (image_bytes, label=1)           -- 1 = OOD positive
        in_dist_samples_with_ood: all samples for joint AUROC      -- (bytes, is_ood: int)
    """
    in_dist = []
    ood_bytes = []

    for cls_name in IN_DIST_CLASSES:
        cls_dir = data_dir / cls_name
        if not cls_dir.exists():
            print(f"[warn] directory not found, skipping: {cls_dir}")
            continue
        imgs = load_images_from_dir(cls_dir)
        print(f"  {cls_name}: {len(imgs)} images")
        for _, img_bytes in imgs:
            in_dist.append((img_bytes, cls_name))

    ood_dir = data_dir / "ood"
    if ood_dir.exists():
        ood_imgs = load_images_from_dir(ood_dir)
        print(f"  ood: {len(ood_imgs)} images")
        for _, img_bytes in ood_imgs:
            ood_bytes.append(img_bytes)
    else:
        print("[warn] ood/ subdirectory not found - OOD metrics will be skipped")

    return in_dist, ood_bytes


def eval_in_distribution(model, transform, device, samples: list, degraded: bool) -> dict:
    """
    Evaluate 2-class classification (us_breast vs us_thyroid).

    Uses run_routing() with ood_threshold=0.0 so every image gets a class label
    regardless of confidence (we measure classification quality, not OOD detection here).
    """
    y_true = []
    y_pred = []
    confidences = []
    latencies_ms = []

    cls_to_idx = {cls: i for i, cls in enumerate(IN_DIST_CLASSES)}

    for img_bytes, true_cls in samples:
        t0 = time.perf_counter()
        result = run_routing(
            model=model,
            transform=transform,
            device=device,
            image_bytes=img_bytes,
            ood_threshold=0.0,
            degraded=degraded,
        )
        latencies_ms.append((time.perf_counter() - t0) * 1000)

        pred_key = result["module_key"]
        if pred_key not in cls_to_idx:
            pred_key = IN_DIST_CLASSES[0]

        y_true.append(cls_to_idx[true_cls])
        y_pred.append(cls_to_idx[pred_key])
        confidences.append(result["confidence"])

    labels = list(range(len(IN_DIST_CLASSES)))
    report = classification_report(
        y_true, y_pred,
        labels=labels,
        target_names=IN_DIST_CLASSES,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    acc = accuracy_score(y_true, y_pred)

    return {
        "n_samples": len(samples),
        "accuracy": round(acc, 4),
        "macro_f1": round(report["macro avg"]["f1-score"], 4),
        "macro_precision": round(report["macro avg"]["precision"], 4),
        "macro_recall": round(report["macro avg"]["recall"], 4),
        "per_class": {
            cls: {
                "precision": round(report[cls]["precision"], 4),
                "recall":    round(report[cls]["recall"], 4),
                "f1":        round(report[cls]["f1-score"], 4),
                "support":   report[cls]["support"],
            }
            for cls in IN_DIST_CLASSES if cls in report
        },
        "confusion_matrix": {
            "labels": IN_DIST_CLASSES,
            "matrix": cm,
        },
        "mean_confidence": round(float(np.mean(confidences)), 4),
        "inference_time_ms": {
            "mean":   round(float(np.mean(latencies_ms)), 2),
            "median": round(float(np.median(latencies_ms)), 2),
            "p95":    round(float(np.percentile(latencies_ms, 95)), 2),
            "min":    round(float(np.min(latencies_ms)), 2),
            "max":    round(float(np.max(latencies_ms)), 2),
        },
    }


def eval_ood_detection(model, transform, device, in_dist_samples, ood_bytes, degraded) -> dict:
    """
    Binary OOD detection: 0 = in-distribution, 1 = OOD.

    Score = 1 - max_softmax_confidence (higher = more likely OOD).
    Sweep threshold to plot ROC; also report at the hardcoded threshold 0.6.
    """
    scores = []
    labels = []

    for img_bytes, _ in in_dist_samples:
        result = run_routing(
            model=model,
            transform=transform,
            device=device,
            image_bytes=img_bytes,
            ood_threshold=0.0,
            degraded=degraded,
        )
        scores.append(1.0 - result["confidence"])
        labels.append(0)

    for img_bytes in ood_bytes:
        result = run_routing(
            model=model,
            transform=transform,
            device=device,
            image_bytes=img_bytes,
            ood_threshold=0.0,
            degraded=degraded,
        )
        scores.append(1.0 - result["confidence"])
        labels.append(1)

    scores_np = np.array(scores)
    labels_np = np.array(labels)

    if labels_np.sum() == 0 or (labels_np == 0).sum() == 0:
        return {"error": "Need both in-distribution and OOD samples for AUROC"}

    auroc = roc_auc_score(labels_np, scores_np)

    fpr_arr, tpr_arr, thresholds = roc_curve(labels_np, scores_np)

    tpr_target = 0.95
    idx_95 = np.searchsorted(tpr_arr, tpr_target)
    fpr_at_95tpr = float(fpr_arr[min(idx_95, len(fpr_arr) - 1)])

    # Metrics at the hardcoded threshold OOD_THRESHOLD=0.6
    # ood_score > (1 - OOD_THRESHOLD) means confidence < OOD_THRESHOLD -> is_ood
    default_score_thresh = 1.0 - OOD_THRESHOLD
    preds_at_default = (scores_np > default_score_thresh).astype(int)
    tp = int(((preds_at_default == 1) & (labels_np == 1)).sum())
    fp = int(((preds_at_default == 1) & (labels_np == 0)).sum())
    fn = int(((preds_at_default == 0) & (labels_np == 1)).sum())
    tn = int(((preds_at_default == 0) & (labels_np == 0)).sum())

    precision_at_default = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall_at_default    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_at_default = (
        2 * precision_at_default * recall_at_default
        / (precision_at_default + recall_at_default)
        if (precision_at_default + recall_at_default) > 0 else 0.0
    )

    # Store ROC curve points (downsample to 100 pts for readable JSON)
    n_pts = min(100, len(fpr_arr))
    idx_pts = np.linspace(0, len(fpr_arr) - 1, n_pts, dtype=int)
    roc_curve_data = [
        {"fpr": round(float(fpr_arr[i]), 4), "tpr": round(float(tpr_arr[i]), 4),
         "threshold_score": round(float(thresholds[i]), 4)}
        for i in idx_pts
    ]

    return {
        "n_in_distribution": int((labels_np == 0).sum()),
        "n_ood": int((labels_np == 1).sum()),
        "auroc": round(float(auroc), 4),
        "fpr_at_95tpr": round(fpr_at_95tpr, 4),
        "at_default_threshold": {
            "ood_threshold": OOD_THRESHOLD,
            "score_threshold_used": round(default_score_thresh, 4),
            "precision": round(precision_at_default, 4),
            "recall":    round(recall_at_default, 4),
            "f1":        round(f1_at_default, 4),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        },
        "roc_curve": roc_curve_data,
    }


def count_parameters(model) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "total_M": round(total / 1e6, 2),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate the router model")
    parser.add_argument("--data_dir", default="data/router/test_router",
                        help="Root dir with us_breast/, us_thyroid/, ood/ subdirs")
    parser.add_argument("--ckpt", default="models/checkpoints/router_effnet_b0.pth",
                        help="Path to router checkpoint (.pth)")
    parser.add_argument("--out_dir", default="eval/results/router",
                        help="Directory to write results JSON and text report")
    parser.add_argument("--device", default=None,
                        help="'cuda' or 'cpu' (auto-detect if not set)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[eval_router] Loading model from: {args.ckpt}")
    model, transform, device, degraded = load_router(args.ckpt, args.device)
    if degraded:
        print("[eval_router] WARNING: running with random weights - results are meaningless")

    print(f"[eval_router] Device: {device}")
    print(f"\n[eval_router] Loading test data from: {data_dir}")
    in_dist_samples, ood_bytes = collect_dataset(data_dir)
    print(f"  Total in-distribution: {len(in_dist_samples)}")
    print(f"  Total OOD: {len(ood_bytes)}")

    if not in_dist_samples:
        print("[eval_router] No in-distribution images found. Check --data_dir.")
        sys.exit(1)

    print("\n[eval_router] Running in-distribution evaluation...")
    in_dist_results = eval_in_distribution(model, transform, device, in_dist_samples, degraded)

    ood_results = None
    if ood_bytes:
        print("[eval_router] Running OOD detection evaluation...")
        ood_results = eval_ood_detection(model, transform, device, in_dist_samples, ood_bytes, degraded)
    else:
        print("[eval_router] Skipping OOD metrics (no OOD images found).")

    params = count_parameters(model)

    output = {
        "checkpoint": args.ckpt,
        "device": device,
        "degraded": degraded,
        "model_parameters": params,
        "in_distribution_classification": in_dist_results,
        "ood_detection": ood_results,
    }

    results_path = out_dir / "router_eval.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n[eval_router] Results saved to: {results_path}")

    print("\n" + "=" * 60)
    print("IN-DISTRIBUTION CLASSIFICATION (us_breast vs us_thyroid)")
    print("=" * 60)
    print(f"  Samples : {in_dist_results['n_samples']}")
    print(f"  Accuracy: {in_dist_results['accuracy']:.4f}")
    print(f"  Macro F1: {in_dist_results['macro_f1']:.4f}")
    print(f"  Macro P : {in_dist_results['macro_precision']:.4f}")
    print(f"  Macro R : {in_dist_results['macro_recall']:.4f}")
    print(f"\n  Inference time (batch=1):")
    t = in_dist_results["inference_time_ms"]
    print(f"    mean={t['mean']}ms  median={t['median']}ms  p95={t['p95']}ms")
    print(f"\n  Model parameters: {params['total_M']}M")
    print(f"\n  Per-class:")
    for cls, m in in_dist_results["per_class"].items():
        print(f"    {cls}: P={m['precision']:.4f}  R={m['recall']:.4f}  "
              f"F1={m['f1']:.4f}  n={m['support']}")
    print(f"\n  Confusion matrix (rows=true, cols=pred):")
    labels_cm = in_dist_results["confusion_matrix"]["labels"]
    print(f"    {'':15s} " + "  ".join(f"{l:10s}" for l in labels_cm))
    for i, row in enumerate(in_dist_results["confusion_matrix"]["matrix"]):
        print(f"    {labels_cm[i]:15s} " + "  ".join(f"{v:10d}" for v in row))

    if ood_results:
        print("\n" + "=" * 60)
        print("OOD DETECTION")
        print("=" * 60)
        print(f"  In-dist: {ood_results['n_in_distribution']}  OOD: {ood_results['n_ood']}")
        print(f"  AUROC         : {ood_results['auroc']:.4f}")
        print(f"  FPR@95TPR     : {ood_results['fpr_at_95tpr']:.4f}")
        d = ood_results["at_default_threshold"]
        print(f"\n  At default threshold OOD_THRESHOLD={d['ood_threshold']}:")
        print(f"    Precision={d['precision']:.4f}  Recall={d['recall']:.4f}  F1={d['f1']:.4f}")
        print(f"    TP={d['tp']}  FP={d['fp']}  FN={d['fn']}  TN={d['tn']}")

    print(f"\n[eval_router] Done. Full results in: {results_path}\n")


if __name__ == "__main__":
    main()
