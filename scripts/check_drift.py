"""
scripts/check_drift.py
========================
Manual data drift check script -- NOT a continuously running service.
Run manually as needed, outputs HTML + JSON for manual review.

Goal:
    Detect early if a new batch of test images diverges from the training
    data distribution (BUSI/TN3K), or if confidence score distribution is
    abnormal (>= 0.999 hints at overfitting).

Tools used: Deepchecks (preferred) or Evidently (fallback).

Input:
    --ref_dir     : Reference image directory (BUSI/TN3K validation set).
    --cur_dir     : New image directory to check.
    --scores_json : JSON list of confidence scores from the most recent analysis run.
                    Format: [{"image_id": "...", "confidence": 0.87, "label": "benign"}, ...]
    --out_dir     : Output directory (default: reports/drift/).

Usage:
    python scripts/check_drift.py \\
        --ref_dir data/busi_val \\
        --cur_dir data/incoming \\
        --scores_json reports/scores.json \\
        --out_dir reports/drift
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Check data drift (Deepchecks / Evidently).")
    p.add_argument("--ref_dir",     default="data/busi_val",
                   help="Reference image directory (validation set)")
    p.add_argument("--cur_dir",     default="data/incoming",
                   help="New image directory to check")
    p.add_argument("--scores_json", default=None,
                   help="JSON file containing the most recent confidence scores")
    p.add_argument("--out_dir",     default="reports/drift",
                   help="Output directory for the report")
    p.add_argument("--backend",     choices=["deepchecks", "evidently", "auto"],
                   default="auto",
                   help="Tool to use. 'auto' tries deepchecks first, falls back to evidently")
    return p.parse_args()


def _load_images_as_array(directory: str):
    """
    Loads images from a directory -> numpy array [N, H, W, C] after resizing to 256x256.
    Returns an empty array if no images are found.
    """
    import numpy as np
    try:
        from PIL import Image
    except ImportError:
        print("ERROR: pip install Pillow")
        sys.exit(1)

    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    paths = [
        p for p in Path(directory).rglob("*")
        if p.suffix.lower() in exts
    ]
    if not paths:
        return np.empty((0, 256, 256, 3), dtype=np.uint8), []

    arrays = []
    for path in paths:
        img = Image.open(path).convert("RGB").resize((256, 256))
        arrays.append(np.array(img))
    return np.stack(arrays), [str(p) for p in paths]


def _extract_image_features(images_array) -> dict:
    """
    Computes features for drift checking:
        - mean brightness
        - contrast (brightness std)
        - aspect ratio (256x256 so always = 1, but kept for future extension)
        - file size distribution (bytes) -- a sign of compression artifacts
    """
    import numpy as np

    if images_array.shape[0] == 0:
        return {}

    brightness = images_array.mean(axis=(1, 2, 3))
    contrast   = images_array.std(axis=(1, 2, 3))

    return {
        "brightness_mean":   float(brightness.mean()),
        "brightness_std":    float(brightness.std()),
        "contrast_mean":     float(contrast.mean()),
        "contrast_std":      float(contrast.std()),
        "n_images":          int(images_array.shape[0]),
    }


def _check_confidence_anomaly(scores: list) -> dict:
    """
    Detects signs of overfitting / distribution shift via confidence scores:
        - rate of requests with confidence >= 0.999 (warning threshold)
        - overall distribution: mean, std, min, max
        - rate by label: has any label "collapsed" onto a single class

    Returns a result dict + the is_anomalous flag.
    """
    if not scores:
        return {"is_anomalous": False, "reason": "No scores to check."}

    import numpy as np

    confs  = np.array([s.get("confidence", 0) for s in scores])
    labels = [s.get("label", "unknown") for s in scores]

    high_conf_rate   = float((confs >= 0.999).mean())
    label_counts     = {}
    for lbl in labels:
        label_counts[lbl] = label_counts.get(lbl, 0) + 1
    dominant_label   = max(label_counts, key=label_counts.get) if label_counts else "none"
    dominant_rate    = label_counts.get(dominant_label, 0) / len(labels) if labels else 0

    is_anomalous = False
    reasons = []

    if high_conf_rate >= 0.3:
        is_anomalous = True
        reasons.append(
            f"{high_conf_rate:.0%} of requests have confidence >= 0.999 "
            f"-- a sign of overfitting or a distribution different from the training set."
        )

    if dominant_rate >= 0.9 and len(label_counts) > 1:
        is_anomalous = True
        reasons.append(
            f"Label '{dominant_label}' accounts for {dominant_rate:.0%} of all requests "
            f"-- the model may have collapsed onto a single class."
        )

    return {
        "is_anomalous":    is_anomalous,
        "reasons":         reasons,
        "high_conf_rate":  high_conf_rate,
        "conf_mean":       float(confs.mean()),
        "conf_std":        float(confs.std()),
        "conf_min":        float(confs.min()),
        "conf_max":        float(confs.max()),
        "label_counts":    label_counts,
        "n_samples":       len(scores),
    }


def _per_image_brightness_contrast(images_array) -> "tuple":
    """
    Computes brightness/contrast for EACH image (not collapsed into one mean value).
    Necessary for drift detection since drift tools need a distribution of many
    samples, not a single summary value.

    Returns (list[float] brightness, list[float] contrast).
    """
    if images_array.shape[0] == 0:
        return [], []
    brightness = images_array.mean(axis=(1, 2, 3)).tolist()
    contrast   = images_array.std(axis=(1, 2, 3)).tolist()
    return brightness, contrast


def _run_deepchecks(ref_images, cur_images, out_dir: str) -> str:
    """
    Runs Deepchecks ImagePropertyDrift, returns the path to the HTML report.

    Uses VisionData with a batch_loader generator returning BatchOutputFormat --
    the way to build VisionData from a raw numpy array without needing
    model/labels (ImagePropertyDrift only needs images). The deepchecks.vision
    API can change between versions; on import or runtime errors, check the
    deepchecks docs matching the installed version.
    """
    try:
        from deepchecks.vision.checks import ImagePropertyDrift
        from deepchecks.vision.vision_data import VisionData
        from deepchecks.vision.vision_data.batch_wrapper import BatchOutputFormat
    except ImportError:
        raise ImportError("deepchecks is not installed: pip install deepchecks[vision]")

    if ref_images.shape[0] == 0 or cur_images.shape[0] == 0:
        raise ValueError("Need at least 1 image in each set to check for drift.")

    def _make_loader(images_array, batch_size=16):
        """Generator returning BatchOutputFormat per image batch -- no labels/predictions needed."""
        n = images_array.shape[0]
        for start in range(0, n, batch_size):
            batch = images_array[start:start + batch_size]
            yield BatchOutputFormat(images=[img for img in batch])

    ref_data = VisionData(
        batch_loader=_make_loader(ref_images),
        task_type="other",
        reshuffle_data=False,
    )
    cur_data = VisionData(
        batch_loader=_make_loader(cur_images),
        task_type="other",
        reshuffle_data=False,
    )

    check  = ImagePropertyDrift()
    result = check.run(ref_data, cur_data)

    html_path = os.path.join(out_dir, "deepchecks_drift.html")
    result.save_as_html(html_path)
    return html_path


def _run_evidently(ref_images, cur_images, out_dir: str) -> str:
    """
    Runs Evidently DataDriftPreset on the PER-IMAGE brightness/contrast distribution.

    Does not use a 1-row summary (mean of means cannot reveal distributional
    drift) - each image is one row in the DataFrame so Evidently can compute
    KS-test / PSI correctly.
    """
    try:
        import pandas as pd
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset
    except ImportError:
        raise ImportError("evidently is not installed: pip install evidently pandas")

    ref_bright, ref_contrast = _per_image_brightness_contrast(ref_images)
    cur_bright, cur_contrast = _per_image_brightness_contrast(cur_images)

    if not ref_bright or not cur_bright:
        raise ValueError("Need at least 1 image in each set to check for drift.")

    ref_df = pd.DataFrame({"brightness": ref_bright, "contrast": ref_contrast})
    cur_df = pd.DataFrame({"brightness": cur_bright, "contrast": cur_contrast})

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref_df, current_data=cur_df)

    html_path = os.path.join(out_dir, "evidently_drift.html")
    report.save_html(html_path)
    return html_path


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("[check_drift] Running data drift check...")
    print(f"  Reference set : {args.ref_dir}")
    print(f"  Set to check  : {args.cur_dir}")
    print(f"  Output        : {args.out_dir}")

    report = {
        "generated_at": datetime.now().isoformat(),
        "ref_dir":       args.ref_dir,
        "cur_dir":       args.cur_dir,
        "scores_json":   args.scores_json,
    }

    print("\n[check_drift] Loading images...")
    ref_images, ref_paths = _load_images_as_array(args.ref_dir)
    cur_images, cur_paths = _load_images_as_array(args.cur_dir)
    print(f"  Reference set: {len(ref_paths)} images")
    print(f"  Set to check:  {len(cur_paths)} images")

    ref_features = _extract_image_features(ref_images)
    cur_features = _extract_image_features(cur_images)
    report["ref_features"] = ref_features
    report["cur_features"] = cur_features

    if ref_features and cur_features:
        brightness_diff = abs(
            cur_features["brightness_mean"] - ref_features["brightness_mean"]
        )
        contrast_diff = abs(
            cur_features["contrast_mean"] - ref_features["contrast_mean"]
        )
        report["feature_diff"] = {
            "brightness_diff": round(brightness_diff, 2),
            "contrast_diff":   round(contrast_diff, 2),
        }
        if brightness_diff > 30:
            print(
                f"  [WARNING] Large brightness difference: "
                f"ref={ref_features['brightness_mean']:.1f}, "
                f"cur={cur_features['brightness_mean']:.1f}"
            )
        if contrast_diff > 20:
            print(
                f"  [WARNING] Large contrast difference: "
                f"ref={ref_features['contrast_mean']:.1f}, "
                f"cur={cur_features['contrast_mean']:.1f}"
            )

    scores = []
    if args.scores_json and os.path.exists(args.scores_json):
        print(f"\n[check_drift] Checking confidence scores: {args.scores_json}")
        with open(args.scores_json) as f:
            scores = json.load(f)
        conf_check = _check_confidence_anomaly(scores)
        report["confidence_check"] = conf_check
        if conf_check["is_anomalous"]:
            print("  [WARNING] Anomaly detected in confidence scores:")
            for reason in conf_check["reasons"]:
                print(f"    - {reason}")
        else:
            print(f"  Confidence OK: mean={conf_check['conf_mean']:.3f}, "
                  f"high_conf_rate={conf_check['high_conf_rate']:.1%}")
    else:
        print("\n[check_drift] Skipping confidence check (--scores_json not provided).")

    html_path = None
    if len(ref_paths) > 0 and len(cur_paths) > 0:
        print(f"\n[check_drift] Running drift detection (backend={args.backend})...")

        backends = (
            ["deepchecks", "evidently"] if args.backend == "auto"
            else [args.backend]
        )
        for backend in backends:
            try:
                if backend == "deepchecks":
                    html_path = _run_deepchecks(ref_images, cur_images, args.out_dir)
                else:
                    html_path = _run_evidently(ref_images, cur_images, args.out_dir)
                print(f"  Backend used: {backend}")
                report["drift_backend"] = backend
                report["drift_html"]    = html_path
                break
            except ImportError as e:
                print(f"  {backend} not available: {e}")
            except Exception as e:
                print(f"  {backend} error: {e}")
    else:
        print("\n[check_drift] Not enough images to run drift detection. Confidence check only.")

    json_path = os.path.join(args.out_dir, "drift_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n[check_drift] Results:")
    print(f"  JSON report: {json_path}")
    if html_path:
        print(f"  HTML report: {html_path}")

    conf_check = report.get("confidence_check", {})
    if conf_check.get("is_anomalous"):
        print("\n[check_drift] CONCLUSION: Anomaly detected. Read the report for details.")
        sys.exit(1)
    else:
        print("\n[check_drift] CONCLUSION: No clear anomaly detected.")
        sys.exit(0)


if __name__ == "__main__":
    main()
