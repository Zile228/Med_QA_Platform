"""
scripts/split_embedding_finetune_data.py  (v2 -- chia theo ty le record)

Chia du lieu fine-tune embedding thanh train/validation/test.

THAY DOI SO VOI v1:
v1 chia theo DON VI FILE (moi organ nhuong nguyen 1 file cho val, 1 file
cho test). Voi organ chi co 2 file (vd thyroid: 97 vs 200 record), cong
thuc "phai con it nhat 1 file cho train" buoc phai HY SINH TOAN BO test
cho organ do -> test set mat trang 1 organ. Day khong phai bug, la he qua
tat yeu cua viec chia theo file nguyen khoi khi so file qua it.

v2 chia theo TY LE RECORD, thuc hien RIENG cho tung source_file (khong
gop chung roi chia ngau nhien theo record, va cung khong chia nguyen ca
file). Voi moi file:
  - Sort record theo chunk_idx (thu tu vi tri that trong tai lieu goc).
  - Cat theo ty le: [dau file] -> val, [giua] -> train, [cuoi file] -> test.
  - Bo di (khong dua vao split nao) `buffer` record o MOI ranh gioi cat,
    de tranh leakage do overlap giua 2 chunk lien ke (RecursiveCharacter
    TextSplitter co overlap giua chunk lien tiep, xem build_vectordb.py).

Vi ty le ap dung TRONG TUNG FILE, moi file (moi tai lieu goc) deu gop mat
o ca 3 split -> khong con tinh huong 1 organ bi mat trang o test/val chi
vi no co it file. Day la khac biet cot loi so voi v1.

Luu y: chunk_idx trong du lieu la ID toan cuc, thua (dataset da duoc
subsample tu vectordb goc, khong phai moi chunk lien tiep deu co mat),
nen buffer tinh theo SO RECORD lien tiep trong tap da subsample, khong
phai theo khoang cach chunk_idx thuc te. Neu muon chat che hon (buffer
theo khoang cach chunk_idx thuc), xem tham so --buffer_by_chunk_gap.

Chay:
  python scripts/split_embedding_finetune_data.py \\
    --in_file   scripts/finetune_data/embedding_training.jsonl \\
    --out_dir   scripts/finetune_data/split \\
    [--val_frac 0.15] [--test_frac 0.15] [--buffer 3]
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path


def _load_records(in_file: Path) -> list:
    records = []
    with open(in_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _group_by_source(records: list) -> dict:
    """Returns {source_file: [record, ...]}."""
    grouped = defaultdict(list)
    for r in records:
        grouped[r["source_file"]].append(r)
    return grouped


def _split_one_file(recs: list, val_frac: float, test_frac: float, buffer: int) -> tuple:
    """
    Chia 1 source_file thanh (train, val, test) theo thu tu chunk_idx.

    Bo tri: [val_block][buffer][train_block][buffer][test_block]
    Val dat o dau, test o cuoi, de neu file co xu huong noi dung dau/cuoi
    khac biet (vd muc luc, tai lieu tham khao) thi ca val va test deu
    "an" mot phan dac thu do, thay vi don het vao 1 phia.

    Neu file qua nho (khong du cho ca buffer + it nhat 1 record moi
    split), giam buffer truoc, roi giam val/test, va in canh bao -- KHONG
    de mot split bi am so luong hay loi ngam.
    """
    n = len(recs)
    recs_sorted = sorted(recs, key=lambda r: r["chunk_idx"])

    n_val = round(n * val_frac)
    n_test = round(n * test_frac)

    # Dam bao van con cho train sau khi tru val + test + 2*buffer.
    # Neu khong du, giam buffer truoc (it anh huong toi ty le hon la giam
    # thang val/test), sau do moi giam n_val/n_test neu van khong du.
    b = buffer
    while b > 0 and n_val + n_test + 2 * b >= n:
        b -= 1
    while n_val + n_test + 2 * b >= n and (n_val > 0 or n_test > 0):
        if n_val >= n_test and n_val > 0:
            n_val -= 1
        elif n_test > 0:
            n_test -= 1
        else:
            break

    warn = None
    if b < buffer or n_val != round(n * val_frac) or n_test != round(n * test_frac):
        warn = (
            f"file qua nho (n={n}) de giu buffer={buffer} va val_frac={val_frac}, "
            f"test_frac={test_frac} nhu mong muon. Da giam xuong buffer={b}, "
            f"n_val={n_val}, n_test={n_test}."
        )

    val = recs_sorted[:n_val]
    train_start = n_val + b
    test_start = n - n_test
    train_end = max(train_start, test_start - b)
    train = recs_sorted[train_start:train_end]
    test = recs_sorted[test_start:] if n_test > 0 else []

    return train, val, test, warn


def split_records(records: list, val_frac: float, test_frac: float, buffer: int) -> tuple:
    """Returns (train, val, test) lists, chia rieng cho tung source_file."""
    grouped = _group_by_source(records)
    train, val, test = [], [], []
    per_file_stats = {}

    for source_file, recs in grouped.items():
        f_train, f_val, f_test, warn = _split_one_file(recs, val_frac, test_frac, buffer)
        train.extend(f_train)
        val.extend(f_val)
        test.extend(f_test)
        per_file_stats[source_file] = {
            "total": len(recs),
            "train": len(f_train),
            "val": len(f_val),
            "test": len(f_test),
            "dropped_buffer": len(recs) - len(f_train) - len(f_val) - len(f_test),
        }
        if warn:
            print(f"  [warn] source_file={source_file}: {warn}")

    return train, val, test, per_file_stats


def _write_jsonl(records: list, out_path: Path):
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _organ_breakdown(records: list) -> dict:
    counts = defaultdict(int)
    for r in records:
        counts[r["organ"]] += 1
    return dict(counts)


def main():
    parser = argparse.ArgumentParser(
        description="Chia du lieu fine-tune embedding thanh train/val/test theo ty le record trong tung file"
    )
    parser.add_argument("--in_file", default="scripts/finetune_data/embedding_training.jsonl")
    parser.add_argument("--out_dir", default="scripts/finetune_data/split")
    parser.add_argument("--val_frac", type=float, default=0.15)
    parser.add_argument("--test_frac", type=float, default=0.15)
    parser.add_argument(
        "--buffer", type=int, default=3,
        help="So record lien tiep bo di o moi ranh gioi cat, chong leakage do chunk overlap."
    )
    args = parser.parse_args()

    in_file = Path(args.in_file)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[split_embedding_finetune_data] Doc {in_file}...")
    records = _load_records(in_file)
    print(f"[split_embedding_finetune_data] {len(records)} record.")

    train, val, test, per_file_stats = split_records(
        records, args.val_frac, args.test_frac, args.buffer
    )

    _write_jsonl(train, out_dir / "train.jsonl")
    _write_jsonl(val, out_dir / "val.jsonl")
    _write_jsonl(test, out_dir / "test.jsonl")

    train_organ = _organ_breakdown(train)
    val_organ = _organ_breakdown(val)
    test_organ = _organ_breakdown(test)
    all_organs = sorted(set(train_organ) | set(val_organ) | set(test_organ))

    missing = []
    for split_name, organ_counts in [("train", train_organ), ("val", val_organ), ("test", test_organ)]:
        missing_here = [o for o in all_organs if organ_counts.get(o, 0) == 0]
        if missing_here:
            missing.append((split_name, missing_here))
            print(f"  [warn] split '{split_name}' thieu organ: {missing_here}")

    readme_lines = [
        "Phan chia train/val/test cho du lieu fine-tune embedding, theo TY LE RECORD",
        "trong tung source_file (khong chia nguyen ca file).",
        "Sinh boi scripts/split_embedding_finetune_data.py (v2) -- deterministic,",
        "chay lai tren cung file dau vao se ra cung 1 ket qua.",
        "",
        f"Nguon: {in_file}",
        f"val_frac={args.val_frac}  test_frac={args.test_frac}  buffer={args.buffer}",
        f"Tong so record dau vao: {len(records)}",
        f"train: {len(train)} record  -> {train_organ}",
        f"val:   {len(val)} record  -> {val_organ}",
        f"test:  {len(test)} record  -> {test_organ}",
        "",
        "Chi tiet theo source_file (total = train + val + test + dropped_buffer):",
    ]
    for s, st in sorted(per_file_stats.items()):
        readme_lines.append(
            f"  {s}: total={st['total']} train={st['train']} val={st['val']} "
            f"test={st['test']} dropped_buffer={st['dropped_buffer']}"
        )
    if missing:
        readme_lines.append("")
        readme_lines.append("CANH BAO thieu organ:")
        for split_name, organs in missing:
            readme_lines.append(f"  [{split_name}] thieu: {organs}")
    (out_dir / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")

    print(f"[split_embedding_finetune_data] train={len(train)} val={len(val)} test={len(test)}")
    print(f"[split_embedding_finetune_data] Output: {out_dir}/{{train,val,test}}.jsonl, README.md")


if __name__ == "__main__":
    main()