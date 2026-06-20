"""
scripts/check_drift.py
========================
Script kiem tra data drift thu cong -- KHONG phai service chay lien tuc.
Chay thu cong khi can, output ra HTML + JSON de doc thu cong.

Muc tieu:
    Phat hien som neu phan phoi anh test moi lech so voi du lieu train (BUSI/TN3K),
    hoac phan phoi confidence score bat thuong (>= 0.999 dai tro overfitting).

Tool su dung: Deepchecks (uu tien) hoac Evidently (fallback).

Input:
    --ref_dir     : Thu muc anh tham chieu (tap validate BUSI/TN3K).
    --cur_dir     : Thu muc anh moi can kiem tra.
    --scores_json : JSON list cac confidence score tu lan phan tich gan nhat.
                    Dinh dang: [{"image_id": "...", "confidence": 0.87, "label": "benign"}, ...]
    --out_dir     : Thu muc output (default: reports/drift/).

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
    p = argparse.ArgumentParser(description="Kiem tra data drift (Deepchecks / Evidently).")
    p.add_argument("--ref_dir",     default="data/busi_val",
                   help="Thu muc anh tham chieu (tap validate)")
    p.add_argument("--cur_dir",     default="data/incoming",
                   help="Thu muc anh moi can kiem tra")
    p.add_argument("--scores_json", default=None,
                   help="File JSON chua confidence scores gan nhat")
    p.add_argument("--out_dir",     default="reports/drift",
                   help="Thu muc output cho report")
    p.add_argument("--backend",     choices=["deepchecks", "evidently", "auto"],
                   default="auto",
                   help="Tool su dung. 'auto' thu deepchecks truoc, fallback evidently")
    return p.parse_args()


def _load_images_as_array(directory: str):
    """
    Load anh tu thu muc -> numpy array [N, H, W, C] sau khi resize ve 256x256.
    Tra ve array rong neu khong tim thay anh.
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
    Tinh cac features de kiem tra drift:
        - do sang trung binh (mean brightness)
        - do tuong phan (std brightness)
        - ty le aspect ratio (256x256 nen luon = 1, nhung dung de mo rong sau)
        - phan phoi kich thuoc file (byte) -- dau hieu ve compression artifact
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
    Phat hien dau hieu overfitting / distribution shift qua confidence score:
        - ty le request co confidence >= 0.999 (dat nguong canh bao)
        - phan phoi tong the: mean, std, min, max
        - ty le theo label: co label nao bi "collapse" ve 1 class khong

    Tra ve dict ket qua + flag is_anomalous.
    """
    if not scores:
        return {"is_anomalous": False, "reason": "Khong co score de kiem tra."}

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
            f"{high_conf_rate:.0%} request co confidence >= 0.999 "
            f"-- dau hieu overfitting hoac distribution khac tap train."
        )

    if dominant_rate >= 0.9 and len(label_counts) > 1:
        is_anomalous = True
        reasons.append(
            f"Label '{dominant_label}' chiem {dominant_rate:.0%} tong request "
            f"-- co the model bi collapse ve 1 class."
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
    Tinh brightness/contrast cho TUNG anh (khong gop thanh 1 so trung binh).
    Can thiet cho drift detection vi cac tool drift can phan phoi nhieu mau,
    khong phai 1 gia tri summary duy nhat.

    Tra ve (list[float] brightness, list[float] contrast).
    """
    if images_array.shape[0] == 0:
        return [], []
    brightness = images_array.mean(axis=(1, 2, 3)).tolist()
    contrast   = images_array.std(axis=(1, 2, 3)).tolist()
    return brightness, contrast


def _run_deepchecks(ref_images, cur_images, out_dir: str) -> str:
    """
    Chay Deepchecks ImagePropertyDrift, tra ve path report HTML.

    Dung VisionData voi batch_loader generator tra ve BatchOutputFormat -- cach build
    VisionData tu raw numpy array ma khong can model/labels (ImagePropertyDrift chi
    can anh). API deepchecks.vision co the thay doi giua cac phien ban; neu loi import
    hoac loi runtime, kiem tra lai doc deepchecks tuong ung voi version dang cai.
    """
    try:
        from deepchecks.vision.checks import ImagePropertyDrift
        from deepchecks.vision.vision_data import VisionData
        from deepchecks.vision.vision_data.batch_wrapper import BatchOutputFormat
    except ImportError:
        raise ImportError("deepchecks chua install: pip install deepchecks[vision]")

    if ref_images.shape[0] == 0 or cur_images.shape[0] == 0:
        raise ValueError("Can it nhat 1 anh trong moi tap de kiem tra drift.")

    def _make_loader(images_array, batch_size=16):
        """Generator tra ve BatchOutputFormat tung batch anh -- khong can labels/predictions."""
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
    Chay Evidently DataDriftPreset tren phan phoi brightness/contrast THEO TUNG ANH.

    Khong dung 1-row summary (mean cua mean khong the hien duoc drift ve phan phoi) -
    moi anh la 1 dong trong DataFrame de Evidently tinh duoc KS-test / PSI dung cach.
    """
    try:
        import pandas as pd
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset
    except ImportError:
        raise ImportError("evidently chua install: pip install evidently pandas")

    ref_bright, ref_contrast = _per_image_brightness_contrast(ref_images)
    cur_bright, cur_contrast = _per_image_brightness_contrast(cur_images)

    if not ref_bright or not cur_bright:
        raise ValueError("Can it nhat 1 anh trong moi tap de kiem tra drift.")

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

    print("[check_drift] Dang chay kiem tra data drift...")
    print(f"  Tap tham chieu : {args.ref_dir}")
    print(f"  Tap can kiem   : {args.cur_dir}")
    print(f"  Output         : {args.out_dir}")

    report = {
        "generated_at": datetime.now().isoformat(),
        "ref_dir":       args.ref_dir,
        "cur_dir":       args.cur_dir,
        "scores_json":   args.scores_json,
    }

    print("\n[check_drift] Load anh...")
    ref_images, ref_paths = _load_images_as_array(args.ref_dir)
    cur_images, cur_paths = _load_images_as_array(args.cur_dir)
    print(f"  Tap tham chieu: {len(ref_paths)} anh")
    print(f"  Tap can kiem:   {len(cur_paths)} anh")

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
                f"  [CANH BAO] Do sang lech nhieu: "
                f"ref={ref_features['brightness_mean']:.1f}, "
                f"cur={cur_features['brightness_mean']:.1f}"
            )
        if contrast_diff > 20:
            print(
                f"  [CANH BAO] Do tuong phan lech nhieu: "
                f"ref={ref_features['contrast_mean']:.1f}, "
                f"cur={cur_features['contrast_mean']:.1f}"
            )

    scores = []
    if args.scores_json and os.path.exists(args.scores_json):
        print(f"\n[check_drift] Kiem tra confidence scores: {args.scores_json}")
        with open(args.scores_json) as f:
            scores = json.load(f)
        conf_check = _check_confidence_anomaly(scores)
        report["confidence_check"] = conf_check
        if conf_check["is_anomalous"]:
            print("  [CANH BAO] Phat hien bat thuong trong confidence scores:")
            for reason in conf_check["reasons"]:
                print(f"    - {reason}")
        else:
            print(f"  Confidence OK: mean={conf_check['conf_mean']:.3f}, "
                  f"high_conf_rate={conf_check['high_conf_rate']:.1%}")
    else:
        print("\n[check_drift] Bo qua kiem tra confidence (--scores_json khong co).")

    html_path = None
    if len(ref_paths) > 0 and len(cur_paths) > 0:
        print(f"\n[check_drift] Chay drift detection (backend={args.backend})...")

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
                print(f"  Backend su dung: {backend}")
                report["drift_backend"] = backend
                report["drift_html"]    = html_path
                break
            except ImportError as e:
                print(f"  {backend} khong co: {e}")
            except Exception as e:
                print(f"  {backend} loi: {e}")
    else:
        print("\n[check_drift] Khong du anh de chay drift detection. Chi kiem tra confidence.")

    json_path = os.path.join(args.out_dir, "drift_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n[check_drift] Ket qua:")
    print(f"  JSON report: {json_path}")
    if html_path:
        print(f"  HTML report: {html_path}")

    conf_check = report.get("confidence_check", {})
    if conf_check.get("is_anomalous"):
        print("\n[check_drift] KET LUAN: Phat hien bat thuong. Doc report de xem chi tiet.")
        sys.exit(1)
    else:
        print("\n[check_drift] KET LUAN: Khong phat hien bat thuong ro ret.")
        sys.exit(0)


if __name__ == "__main__":
    main()
