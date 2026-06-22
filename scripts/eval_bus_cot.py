"""
scripts/eval_bus_cot.py
=========================
Offline pipeline evaluation script against BUS-CoT -- NOT a continuously
running service and NOT a CI gate. Run manually when classification +
ICD-10 mapping quality needs evaluating against expert ground truth.

Dataset: BUS-CoT (Yu et al., Scientific Data 2026, doi:10.1038/s41597-026-06702-9,
arXiv:2509.17046). 11,439 images / ~10,000 lesions, covering all 99 WHO
histopathology categories, available on Figshare (doi:10.6084/m9.figshare.30838715),
including the BUS-Lesion subset (5,163 images) for training/evaluation.

IMPORTANT -- check the license before downloading:
    The paper is an Open Access Data Descriptor and the authors state
    "data and code are publicly available", but this script does NOT
    automatically confirm the specific license terms of the Figshare
    deposit (CC0 / CC-BY / CC-BY-NC...). Before running --download or
    using the dataset for anything beyond internal research/evaluation,
    re-read the Figshare page (doi:10.6084/m9.figshare.30838715) to
    confirm the exact usage terms.

Granularity mismatch (worth noting when reading results):
    BUS-CoT has 99 histopathology categories, while this pipeline only has
    3 classes (normal/benign/malignant). The script collapses the 99
    categories down to 3 classes via BUS_COT_TO_PLATFORM_LABEL (see
    _collapse_histopathology_label) -- this is a methodological decision
    that needs to be explicit, not a silent default. If a category is not
    in the mapping table, the script assigns 'unknown' and does NOT count
    it toward accuracy (see _build_confusion_report).

Input:
    --bus_cot_json : BUS-Lesion annotation JSON file (after downloading from
                     Figshare). Expected format (a list of records):
                     [{"image_path": "...", "histopathology_category": "...",
                       "icd10": "...", "reasoning": {...}}, ...]
    --image_root   : Root directory containing images, joined with image_path in the JSON.
    --orchestrator_url : Orchestrator URL (default http://localhost:8000).
    --limit        : Only run the first N records (for quick testing, default: all).
    --out_dir      : Output directory for the report (default: reports/eval_bus_cot/).

Usage:
    python scripts/eval_bus_cot.py \\
        --bus_cot_json data/bus_cot/bus_lesion_annotations.json \\
        --image_root data/bus_cot/images \\
        --orchestrator_url http://localhost:8000 \\
        --out_dir reports/eval_bus_cot

Cost warning: running the full BUS-Lesion set (5,163 images) through the
live pipeline (vision + knowledge + LLM) can be slow and, if
LLM_BACKEND=google, incurs real API spend. Use --limit for quick checks.
"""

import argparse
import json
import os
import sys
import time

import httpx

# Maps the 99 BUS-CoT histopathology categories down to the platform's 3 classes.
# This is a methodological decision, not an exact 1-1 mapping -- e.g. the
# "uncertain malignant potential" (borderline) groups are classified as
# 'malignant' in the safe direction (a false negative is more dangerous than
# a false positive in screening). This table does NOT cover all 99
# categories yet -- extend it as real annotations are encountered.
BUS_COT_TO_PLATFORM_LABEL = {
    # Benign
    "fibroadenoma":                 "benign",
    "cyst":                         "benign",
    "fibrocystic_change":           "benign",
    "adenosis":                     "benign",
    "sclerosing_adenosis":          "benign",
    "intraductal_papilloma":        "benign",
    "lipoma":                       "benign",
    "hamartoma":                    "benign",
    "phyllodes_tumor_benign":       "benign",
    "normal_breast_tissue":         "normal",
    # Malignant (including uncertain malignant potential -- classified safely)
    "invasive_ductal_carcinoma":    "malignant",
    "invasive_lobular_carcinoma":   "malignant",
    "ductal_carcinoma_in_situ":     "malignant",
    "mucinous_carcinoma":           "malignant",
    "metaplastic_carcinoma":        "malignant",
    "phyllodes_tumor_malignant":    "malignant",
    "atypical_ductal_hyperplasia":  "malignant",   # uncertain potential, classified safely
    "lobular_carcinoma_in_situ":    "malignant",   # uncertain potential, classified safely
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate the pipeline against BUS-CoT (offline, not a CI gate)."
    )
    p.add_argument("--bus_cot_json", required=True,
                   help="BUS-Lesion annotation JSON file downloaded from Figshare")
    p.add_argument("--image_root", required=True,
                   help="Root directory containing images, joined with image_path in the JSON")
    p.add_argument("--orchestrator_url", default="http://localhost:8000",
                   help="URL of the running orchestrator")
    p.add_argument("--limit", type=int, default=None,
                   help="Only run the first N records (for quick testing)")
    p.add_argument("--out_dir", default="reports/eval_bus_cot",
                   help="Output directory for the report")
    p.add_argument("--timeout", type=int, default=180,
                   help="Timeout (seconds) for each /analyze request")
    return p.parse_args()


def _collapse_histopathology_label(category: str) -> str:
    """
    Maps a BUS-CoT histopathology category down to benign/malignant/normal.
    Returns 'unknown' if the category is not in BUS_COT_TO_PLATFORM_LABEL --
    these records are excluded from accuracy, not misassigned to a class.
    """
    key = category.strip().lower().replace(" ", "_").replace("-", "_")
    return BUS_COT_TO_PLATFORM_LABEL.get(key, "unknown")


def _load_bus_cot_records(json_path: str, image_root: str, limit: int = None) -> list:
    """
    Reads the annotation JSON file, joins image_path with image_root into an
    absolute path. Skips records missing image_path or whose file doesn't exist.
    """
    with open(json_path, encoding="utf-8") as f:
        raw = json.load(f)

    records = []
    for r in raw:
        rel_path = r.get("image_path")
        if not rel_path:
            continue
        abs_path = os.path.join(image_root, rel_path)
        if not os.path.exists(abs_path):
            print(f"  [skipped] image not found: {abs_path}")
            continue
        records.append({
            "image_path":   abs_path,
            "ground_truth_category": r.get("histopathology_category", "unknown"),
            "ground_truth_icd10":    r.get("icd10"),
            "ground_truth_reasoning": r.get("reasoning"),
        })
        if limit and len(records) >= limit:
            break
    return records


def _call_analyze(
    client: httpx.Client,
    orchestrator_url: str,
    image_path: str,
    question: str = "What are the findings in this ultrasound image?",
) -> dict:
    """
    Calls the orchestrator's /analyze -- same pattern as ui/app.py::call_orchestrator().
    Returns the ReportOutput dict, raises httpx.HTTPStatusError on failure.
    """
    with open(image_path, "rb") as f:
        resp = client.post(
            f"{orchestrator_url}/analyze",
            files={"image": (os.path.basename(image_path), f, "image/png")},
            data={"question": question},
        )
    resp.raise_for_status()
    return resp.json()


def _check_reasoning_structure(reasoning_text: str) -> dict:
    """
    Checks the structure (not semantics) of the CoT reasoning -- a first step
    before attempting a semantic comparison against BUS-CoT's stage-wise
    annotation (observation/feature/diagnosis/pathology), since semantic
    comparison is hard to automate and noisy (see TODO.md item 5).
    """
    if not reasoning_text:
        return {"has_reasoning": False}

    text_lower = reasoning_text.lower()
    return {
        "has_reasoning":           True,
        "mentions_spatial":        any(
            kw in text_lower for kw in ["location", "quadrant", "bbox", "area", "margin"]
        ),
        "mentions_confidence":     any(
            kw in text_lower for kw in ["confidence", "%", "score"]
        ),
        "mentions_severity":       any(
            kw in text_lower for kw in ["severity", "urgent", "significant", "critical", "incidental"]
        ),
        "length_chars":            len(reasoning_text),
    }


def _build_confusion_report(results: list) -> dict:
    """
    Computes accuracy + confusion matrix for classification (3-class collapse).
    Records with ground_truth 'unknown' (category not in the mapping table)
    are excluded from accuracy -- the exclusion rate is reported separately
    to show the mapping table's coverage.
    """
    from sklearn.metrics import accuracy_score, confusion_matrix

    usable = [r for r in results if r["ground_truth_label"] != "unknown" and r.get("predicted_label")]
    excluded_count = len(results) - len(usable)

    if not usable:
        return {
            "accuracy": None,
            "n_usable": 0,
            "n_excluded_unmapped": excluded_count,
            "note": "No record has a mapped ground truth -- extend BUS_COT_TO_PLATFORM_LABEL.",
        }

    y_true = [r["ground_truth_label"] for r in usable]
    y_pred = [r["predicted_label"] for r in usable]
    labels = ["normal", "benign", "malignant"]

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "accuracy":             float(accuracy_score(y_true, y_pred)),
        "n_usable":             len(usable),
        "n_excluded_unmapped":  excluded_count,
        "labels":               labels,
        "confusion_matrix":     cm.tolist(),
    }


def _build_icd10_report(results: list) -> dict:
    """
    Compares the platform's icd10_hint against the ICD code already annotated
    by BUS-CoT -- a direct comparison (apples-to-apples), without going through
    the collapse step like classification, since BUS-CoT already assigns
    ICD-10 per record.
    """
    usable = [r for r in results if r.get("ground_truth_icd10") and r.get("predicted_icd10")]
    if not usable:
        return {"icd10_match_rate": None, "n_usable": 0}

    matches = sum(
        1 for r in usable
        if r["ground_truth_icd10"].strip().upper() == r["predicted_icd10"].strip().upper()
    )
    return {
        "icd10_match_rate": matches / len(usable),
        "n_usable":          len(usable),
        "n_matches":         matches,
    }


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("[eval_bus_cot] This is an offline evaluation, NOT a CI gate.")
    print(f"  BUS-CoT JSON : {args.bus_cot_json}")
    print(f"  Image root   : {args.image_root}")
    print(f"  Orchestrator : {args.orchestrator_url}")

    print("\n[eval_bus_cot] Loading annotations...")
    records = _load_bus_cot_records(args.bus_cot_json, args.image_root, args.limit)
    print(f"  Loaded {len(records)} records with valid images.")

    if not records:
        print("[eval_bus_cot] No records to run. Check --bus_cot_json / --image_root again.")
        sys.exit(1)

    results = []
    with httpx.Client(timeout=args.timeout) as client:
        for i, rec in enumerate(records):
            print(f"  [{i + 1}/{len(records)}] {os.path.basename(rec['image_path'])}")
            ground_truth_label = _collapse_histopathology_label(rec["ground_truth_category"])

            try:
                t_start = time.perf_counter()
                report = _call_analyze(client, args.orchestrator_url, rec["image_path"])
                elapsed = time.perf_counter() - t_start
            except Exception as e:
                print(f"    [error] /analyze failed: {e}")
                results.append({
                    "image_path":           rec["image_path"],
                    "ground_truth_category": rec["ground_truth_category"],
                    "ground_truth_label":    ground_truth_label,
                    "ground_truth_icd10":    rec.get("ground_truth_icd10"),
                    "error":                 str(e),
                })
                continue

            t1 = report.get("tier_1_structured", {})
            cot_result = report.get("cot_result") or {}

            results.append({
                "image_path":            rec["image_path"],
                "ground_truth_category": rec["ground_truth_category"],
                "ground_truth_label":    ground_truth_label,
                "ground_truth_icd10":    rec.get("ground_truth_icd10"),
                "predicted_label":       t1.get("label"),
                "predicted_icd10":       t1.get("icd10_hint"),
                "predicted_confidence":  t1.get("confidence"),
                "latency_seconds":       round(elapsed, 2),
                "reasoning_structure":   _check_reasoning_structure(cot_result.get("reasoning")),
            })

    print("\n[eval_bus_cot] Computing metrics...")
    confusion_report = _build_confusion_report(results)
    icd10_report = _build_icd10_report(results)

    n_errors = sum(1 for r in results if r.get("error"))
    n_reasoning_present = sum(
        1 for r in results
        if r.get("reasoning_structure", {}).get("has_reasoning")
    )

    if confusion_report.get("accuracy") is not None:
        print(f"  Classification accuracy (3-class collapse): {confusion_report['accuracy']:.1%} "
              f"({confusion_report['n_usable']} usable records)")
    else:
        print("  Classification accuracy: could not be computed (see confusion_report.note)")

    if icd10_report.get("icd10_match_rate") is not None:
        print(f"  ICD-10 match rate: {icd10_report['icd10_match_rate']:.1%} "
              f"({icd10_report['n_usable']} usable records)")
    else:
        print("  ICD-10 match rate: could not be computed (missing ground_truth_icd10 or predicted_icd10)")

    print(f"  /analyze errors: {n_errors}/{len(results)}")
    print(f"  CoT reasoning present: {n_reasoning_present}/{len(results)}")

    full_report = {
        "generated_at":      time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_records":         len(results),
        "n_errors":          n_errors,
        "classification":    confusion_report,
        "icd10":             icd10_report,
        "n_reasoning_present": n_reasoning_present,
        "results":           results,
    }

    json_path = os.path.join(args.out_dir, "eval_bus_cot_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2, ensure_ascii=False)

    print(f"\n[eval_bus_cot] Full report: {json_path}")


if __name__ == "__main__":
    main()
