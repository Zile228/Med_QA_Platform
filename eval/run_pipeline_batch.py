"""
eval/run_pipeline_batch.py
=============================
Chay full pipeline qua API /analyze cho 1 thu muc anh, luu moi ket qua thanh
1 file JSON trong eval/results/pipeline_outputs/. Day la buoc tao du lieu dau
vao cho eval/eval_ragas.py --mode pipeline (faithfulness + answer_relevancy
tren output thuc te cua QA Agent).

YEU CAU: full Docker stack (orchestrator, router, vision, knowledge) phai
dang chay (`docker compose up`), vi /analyze can goi qua HTTP den tat ca
service do. Script nay KHONG chay duoc voi orchestrator dung 1 minh.

QUAN TRONG: ca --organ_hint va --modality_hint cua /analyze (xem
services/orchestrator/main.py) deu nhan gia tri 'breast' | 'thyroid' | None,
KHONG phai 'ultrasound' -- 2 hint nay duoc router dung cung nhau de quyet
dinh module, khong phai 1 chi dinh modality kieu DICOM. Neu chi co 1 gia tri
chac chan, chi can dien --organ_hint; --modality_hint co the de trong.

Ho tro 2 nguon anh:
  --image_dir   thu muc anh thuong (vd BUSI test set, hoac anh tu U2-Bench
                da export ra .png/.jpg). Neu co file label di kem
                (--labels_csv: cot 0 = ten file, cot 1 = gt_label) thi gt_label
                duoc luu kem vao JSON output de doi chieu sau.
                File co ten ket thuc bang "_mask" (vi du "benign (1)_mask.png")
                duoc bo qua tu dong -- day la segmentation mask cua BUSI, khong
                phai anh sieu am, router se reject chung la OOD.
  --organ_hint / --modality_hint  forward thang vao /analyze ('breast' |
                'thyroid' | None), dat cung 1 gia tri cho ca batch (vd toan
                bo anh trong thu muc la "breast"). Neu can mix nhieu organ
                trong 1 batch, dung --labels_csv voi them cot thu 3 = organ
                (xem _read_labels_csv).

Output: eval/results/pipeline_outputs/{image_stem}.json voi cau truc
  {"report": <ReportOutput dict tu /analyze>, "gt_label": "...", "image_path": "..."}
(dung dung key "report" ma eval_ragas.py.load_pipeline_outputs() doc).

Chay:
  python eval/run_pipeline_batch.py \\
    --image_dir     data/busi/test_busi/malignant \\
    --api_url       http://localhost:8000 \\
    --organ_hint    breast \\
    --out_dir       eval/results/pipeline_outputs
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import httpx


def _read_labels_csv(labels_csv: str) -> dict:
    """
    Doc file CSV nhan kem theo anh (khong header):
      cot 0 = ten file, cot 1 = gt_label, cot 2 (tuy chon) = organ.
    Tra ve dict: filename -> {"gt_label": ..., "organ": ... | None}.
    """
    mapping = {}
    with open(labels_csv, newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) < 2:
                continue
            name = row[0].strip()
            gt_label = row[1].strip()
            organ = row[2].strip() if len(row) > 2 and row[2].strip() else None
            mapping[name] = {"gt_label": gt_label, "organ": organ}
            mapping[Path(name).stem] = {"gt_label": gt_label, "organ": organ}
    return mapping


def run_batch(
    image_dir: str,
    api_url: str,
    out_dir: str,
    organ_hint: str = None,
    modality_hint: str = None,
    labels_csv: str = None,
    max_n: int = None,
    timeout: float = 120.0,
):
    image_dir_p = Path(image_dir)
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    label_map = _read_labels_csv(labels_csv) if labels_csv else {}

    images = sorted(
        p for p in image_dir_p.iterdir()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
        and not p.stem.endswith("_mask")
    )
    if max_n:
        images = images[:max_n]
    if not images:
        print(f"[run_pipeline_batch] Khong tim thay anh nao trong {image_dir_p}")
        return

    print(f"[run_pipeline_batch] {len(images)} anh -- API: {api_url}/analyze")

    client = httpx.Client(timeout=timeout)
    n_ok, n_fail = 0, 0
    for i, img_path in enumerate(images, start=1):
        out_path = out_dir_p / f"{img_path.stem}.json"
        if out_path.exists():
            print(f"  [{i}/{len(images)}] {img_path.name}: da co output, bo qua.")
            continue

        meta = label_map.get(img_path.name) or label_map.get(img_path.stem) or {}
        gt_label = meta.get("gt_label", "")
        organ = meta.get("organ") or organ_hint

        form_data = {}
        # Pass a deterministic image_id derived from the filename so that
        # eval_qa.py can call /chat with the same image_id that /analyze
        # cached in _context_cache.  Without this, /analyze generates a
        # random UUID, the JSON is saved with that UUID, but if the
        # orchestrator restarts (or its in-memory cache is cleared) before
        # eval_qa.py runs, every /chat call returns 404.
        form_data["image_id"] = img_path.stem
        if organ:
            form_data["organ_hint"] = organ
        if modality_hint:
            form_data["modality_hint"] = modality_hint

        print(f"  [{i}/{len(images)}] {img_path.name} (organ={organ or '?'})")
        try:
            with open(img_path, "rb") as f:
                files = {"image": (img_path.name, f, "application/octet-stream")}
                resp = client.post(f"{api_url}/analyze", data=form_data, files=files)
            resp.raise_for_status()
            report = resp.json()
        except Exception as e:
            print(f"    [skip] Loi goi /analyze: {e}")
            n_fail += 1
            continue

        record = {
            "report": report,
            "gt_label": gt_label,
            "image_path": str(img_path),
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
        n_ok += 1

    print(f"[run_pipeline_batch] Hoan tat: {n_ok} thanh cong, {n_fail} loi -> {out_dir_p}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Chay full pipeline qua /analyze cho 1 batch anh")
    p.add_argument("--image_dir", required=True)
    p.add_argument("--api_url", default="http://localhost:8000")
    p.add_argument("--out_dir", default="eval/results/pipeline_outputs")
    p.add_argument("--organ_hint", default=None, help="'breast' | 'thyroid', ap dung cho ca batch neu khong co labels_csv")
    p.add_argument("--modality_hint", default=None, help="'breast' | 'thyroid' | None -- giong gia tri cua organ_hint, KHONG phai 'ultrasound'")
    p.add_argument("--labels_csv", default=None, help="CSV khong header: ten_file,gt_label[,organ]")
    p.add_argument("--max_n", type=int, default=None)
    p.add_argument("--timeout", type=float, default=120.0)
    args = p.parse_args()

    t0 = time.time()
    run_batch(
        args.image_dir, args.api_url, args.out_dir,
        organ_hint=args.organ_hint, modality_hint=args.modality_hint,
        labels_csv=args.labels_csv, max_n=args.max_n, timeout=args.timeout,
    )
    print(f"[run_pipeline_batch] Thoi gian chay: {time.time() - t0:.1f}s")