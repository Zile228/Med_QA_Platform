"""
eval/run_pipeline_batch.py
=============================
Chay full pipeline qua API /analyze cho 1 batch anh, luu moi ket qua thanh
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

Ho tro 3 nguon anh (chon dung 1 trong --image_dir / --busi_dir / --tn3k_dir):

  --busi_dir DIR
      DIR la thu muc cha kieu "data/busi/test_busi" chua 3 thu muc con
      benign/, malignant/, normal/ -- gt_label duoc suy TU TEN THU MUC CON,
      khong can --labels_csv (dung cung convention voi
      eval_cot.py:collect_busi_dataset() va
      scripts/generate_finetune_data.py:collect_busi_train()). organ luon la
      "breast", tu dong set organ_hint="breast" cho moi anh, khong doc tu
      --organ_hint. File ket thuc bang "_mask" (vd "benign (1)_mask.png") bi
      bo qua tu dong vi la segmentation mask, khong phai anh sieu am.
      Dung --n_per_class de gioi han so anh LAY TU MOI THU MUC CON (vd
      --n_per_class 20 tren BUSI se ra toi da 20+20+20=60 anh, KHONG PHAI 20
      anh tong -- BUSI co 3 lop nen "20 mau/lop" != "20 mau/organ").

  --tn3k_dir DIR
      DIR la thu muc cha kieu "data/tn3k/test_tn3k" chua test-image/ va
      label4test.csv (khong header: cot 0 = ten file, cot 1 = nhan so
      0=benign/1=malignant) -- gt_label duoc suy tu CSV nay, khong can
      --labels_csv rieng (dung cung convention voi
      eval_cot.py:collect_tn3k_dataset()). organ luon la "thyroid", tu dong
      set organ_hint="thyroid" cho moi anh. TN3K chi co 2 lop (khong co
      "normal"), nen --n_per_class 20 tren TN3K ra toi da 20+20=40 anh.

  --image_dir DIR (che do cu, van giu nguyen hanh vi truoc)
      DIR la 1 thu muc anh PHANG (khong co thu muc con theo nhan), dung cho
      nguon anh tuy y (vd anh tu U2-Bench da export ra .png/.jpg). Neu co
      file label di kem (--labels_csv: cot 0 = ten file, cot 1 = gt_label,
      cot 3 tuy chon = organ) thi gt_label duoc luu kem vao JSON output de
      doi chieu sau; neu KHONG truyen --labels_csv, gt_label se la CHUOI
      RONG trong output (khong co nhan de doi chieu, faithfulness/
      answer_relevancy van tinh duoc binh thuong vi khong phu thuoc
      gt_label, nhung khong doi chieu duoc dung/sai chan doan).
      --organ_hint / --modality_hint forward thang vao /analyze, dat cung 1
      gia tri cho ca batch. Neu can mix nhieu organ trong 1 batch --image_dir
      phang, dung --labels_csv voi them cot thu 3 = organ (xem
      _read_labels_csv).
  File ten ket thuc bang "_mask" bi bo qua tu dong o CA 3 che do -- day la
  segmentation mask cua BUSI, khong phai anh sieu am, router se reject
  chung la OOD.

Output: eval/results/pipeline_outputs/{image_stem}.json voi cau truc
  {"report": <ReportOutput dict tu /analyze>, "gt_label": "...", "image_path": "..."}
(dung dung key "report" ma eval_ragas.py.load_pipeline_outputs() doc).

Vi du chay -- 20 mau/lop tren ca BUSI (60 anh, breast) va TN3K (40 anh, thyroid):
  python eval/run_pipeline_batch.py \\
    --busi_dir      data/busi/test_busi \\
    --api_url       http://localhost:8000 \\
    --n_per_class   20 \\
    --out_dir       eval/results/pipeline_outputs

  python eval/run_pipeline_batch.py \\
    --tn3k_dir      data/tn3k/test_tn3k \\
    --api_url       http://localhost:8000 \\
    --n_per_class   20 \\
    --out_dir       eval/results/pipeline_outputs

Che do cu (image_dir phang, van con dung duoc):
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


def _collect_flat_dir(image_dir: Path, labels_csv: str, organ_hint: str, max_n: int) -> list:
    """
    Che do cu: 1 thu muc anh phang, gt_label tu --labels_csv (hoac rong neu
    khong truyen). Tra ve list dict {path, gt_label, organ}.
    """
    label_map = _read_labels_csv(labels_csv) if labels_csv else {}
    images = sorted(
        p for p in image_dir.iterdir()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
        and not p.stem.endswith("_mask")
    )
    if max_n:
        images = images[:max_n]

    samples = []
    for p in images:
        meta = label_map.get(p.name) or label_map.get(p.stem) or {}
        samples.append({
            "path": p,
            "gt_label": meta.get("gt_label", ""),
            "organ": meta.get("organ") or organ_hint,
        })
    return samples


def _collect_busi_dir(busi_dir: Path, n_per_class: int) -> list:
    """
    gt_label suy tu ten thu muc con (benign/malignant/normal), organ luon la
    "breast". Cung convention voi eval_cot.py:collect_busi_dataset(). Neu
    n_per_class duoc truyen, gioi han so anh LAY TU MOI THU MUC CON (khong
    phai tong so anh cua ca 3 lop).
    """
    samples = []
    for gt_label in ("benign", "malignant", "normal"):
        cls_dir = busi_dir / gt_label
        if not cls_dir.exists():
            print(f"  [warn] Khong tim thay {cls_dir}, bo qua.")
            continue
        images = sorted(
            p for p in cls_dir.iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
            and not p.stem.endswith("_mask")
        )
        if n_per_class:
            images = images[:n_per_class]
        for p in images:
            samples.append({"path": p, "gt_label": gt_label, "organ": "breast"})
    return samples


def _collect_tn3k_dir(tn3k_dir: Path, n_per_class: int) -> list:
    """
    gt_label suy tu label4test.csv (0=benign, 1=malignant), organ luon la
    "thyroid". Cung convention voi eval_cot.py:collect_tn3k_dataset(). TN3K
    chi co 2 lop (khong co "normal"). Neu n_per_class duoc truyen, gioi han
    so anh LAY TU MOI LOP (khong phai tong so anh cua ca 2 lop).
    """
    csv_path = tn3k_dir / "label4test.csv"
    image_dir = tn3k_dir / "test-image"
    if not csv_path.exists():
        print(f"  [warn] Khong tim thay {csv_path}")
        return []
    if not image_dir.exists():
        print(f"  [warn] Khong tim thay {image_dir}")
        return []

    label_map = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
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
        print(f"  [warn] {csv_path} doc duoc 0 dong hop le, kiem tra lai dinh dang file.")
        return []

    by_class = {"benign": [], "malignant": []}
    n_missing = 0
    for p in sorted(image_dir.iterdir()):
        if p.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp"}:
            continue
        gt_label = label_map.get(p.name) or label_map.get(p.stem)
        if gt_label is None:
            n_missing += 1
            continue
        by_class[gt_label].append(p)
    if n_missing:
        print(f"  [warn] {n_missing} anh trong {image_dir} khong co nhan trong {csv_path}, da bo qua.")

    samples = []
    for gt_label, images in by_class.items():
        if n_per_class:
            images = images[:n_per_class]
        for p in images:
            samples.append({"path": p, "gt_label": gt_label, "organ": "thyroid"})
    return samples


def run_batch(
    samples: list,
    api_url: str,
    out_dir: str,
    modality_hint: str = None,
    timeout: float = 120.0,
):
    """
    samples: list dict {"path": Path, "gt_label": str, "organ": str | None},
    da duoc 1 trong 3 ham _collect_* o tren chuan bi san.
    """
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    if not samples:
        print("[run_pipeline_batch] Khong tim thay anh nao.")
        return

    print(f"[run_pipeline_batch] {len(samples)} anh -- API: {api_url}/analyze")

    client = httpx.Client(timeout=timeout)
    n_ok, n_fail = 0, 0
    for i, sample in enumerate(samples, start=1):
        img_path = sample["path"]
        gt_label = sample["gt_label"]
        organ = sample["organ"]

        out_path = out_dir_p / f"{img_path.stem}.json"
        if out_path.exists():
            print(f"  [{i}/{len(samples)}] {img_path.name}: da co output, bo qua.")
            continue

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

        print(f"  [{i}/{len(samples)}] {img_path.name} (organ={organ or '?'}, gt_label={gt_label or '?'})")
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
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--image_dir", default=None,
                      help="Che do cu: 1 thu muc anh phang, gt_label tu --labels_csv (rong neu khong truyen)")
    src.add_argument("--busi_dir", default=None,
                      help="Thu muc cha chua benign/malignant/normal (vd data/busi/test_busi), gt_label tu ten thu muc con, organ=breast")
    src.add_argument("--tn3k_dir", default=None,
                      help="Thu muc cha chua test-image/ + label4test.csv (vd data/tn3k/test_tn3k), gt_label tu CSV, organ=thyroid")
    p.add_argument("--api_url", default="http://localhost:8000")
    p.add_argument("--out_dir", default="eval/results/pipeline_outputs")
    p.add_argument("--organ_hint", default=None,
                    help="Chi ap dung voi --image_dir. 'breast' | 'thyroid', ap dung cho ca batch neu khong co labels_csv")
    p.add_argument("--modality_hint", default=None, help="'breast' | 'thyroid' | None -- giong gia tri cua organ_hint, KHONG phai 'ultrasound'")
    p.add_argument("--labels_csv", default=None,
                    help="Chi ap dung voi --image_dir. CSV khong header: ten_file,gt_label[,organ]")
    p.add_argument("--n_per_class", type=int, default=None,
                    help="Chi ap dung voi --busi_dir / --tn3k_dir. So anh LAY TU MOI LOP (khong phai tong). "
                         "VD: --n_per_class 20 tren BUSI (3 lop) ra toi da 60 anh, tren TN3K (2 lop) ra toi da 40 anh.")
    p.add_argument("--max_n", type=int, default=None,
                    help="Chi ap dung voi --image_dir. Gioi han TONG so anh (khac --n_per_class la gioi han MOI LOP).")
    p.add_argument("--timeout", type=float, default=120.0)
    args = p.parse_args()

    if args.image_dir:
        samples = _collect_flat_dir(Path(args.image_dir), args.labels_csv, args.organ_hint, args.max_n)
    elif args.busi_dir:
        samples = _collect_busi_dir(Path(args.busi_dir), args.n_per_class)
    else:
        samples = _collect_tn3k_dir(Path(args.tn3k_dir), args.n_per_class)

    t0 = time.time()
    run_batch(
        samples, args.api_url, args.out_dir,
        modality_hint=args.modality_hint, timeout=args.timeout,
    )
    print(f"[run_pipeline_batch] Thoi gian chay: {time.time() - t0:.1f}s")