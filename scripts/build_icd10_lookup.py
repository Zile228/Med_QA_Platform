"""
scripts/build_icd10_lookup.py

Offline script, run once to parse the ICD-10-CM tabular PDF into a
code -> description JSON lookup.

ICD-10 is a structured reference (each code maps to one fixed description),
not free-flowing clinical narrative, so it is not embedded into FAISS like
the other clinical PDFs (see ALLOWED_PDF_FILENAMES in build_vectordb.py).
An exact key lookup is faster and 100% accurate, unlike semantic search
which depends on unstable cosine similarity for short, context-free lines.

Parsing approach:
Each ICD-10 code line in the source PDF starts with the code itself
(e.g. "A01.03 Typhoid pneumonia" or "A01 Typhoid and paratyphoid fevers"),
followed by its description on the same line. Range headers such as
"A00-A09" (block titles with no inline description) are skipped, since
they are not addressable codes and the loop below only matches single-code
lines via CODE_LINE_RE.

Usage:
    python scripts/build_icd10_lookup.py
    python scripts/build_icd10_lookup.py --pdf_path path/to/icd10.pdf --out_path path/to/out.json
"""

import os
import re
import sys
import json
import argparse


CODE_LINE_RE = re.compile(r"^([A-Z][0-9]{2}(?:\.[0-9A-Z]{1,4})?)\s+(.+)$")
RANGE_HEADER_RE = re.compile(r"^[A-Z][0-9]{2}-[A-Z][0-9]{2}$")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--pdf_path",
        default="services/orchestrator/rag/docs/icd10cm-tabular-2022-April-1.pdf",
        help="Path to the ICD-10-CM tabular PDF",
    )
    p.add_argument(
        "--out_path",
        default="services/orchestrator/icd10_lookup.json",
        help="Output path for the code -> description JSON lookup",
    )
    return p.parse_args()


def parse_icd10_pdf(pdf_path: str) -> dict:
    """
    Extracts a code -> description mapping from the ICD-10-CM tabular PDF.

    Returns an empty dict if the PDF cannot be opened, so the caller can
    report a clear error instead of crashing on an AttributeError.
    """
    try:
        import fitz
    except ImportError:
        print("ERROR: pip install pymupdf")
        sys.exit(1)

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"[build_icd10_lookup] Could not open {pdf_path}: {e}")
        return {}

    lookup = {}
    for page in doc:
        text = page.get_text("text")
        for raw_line in text.split("\n"):
            line = raw_line.strip()
            if not line or RANGE_HEADER_RE.match(line):
                continue
            match = CODE_LINE_RE.match(line)
            if not match:
                continue
            code, description = match.group(1), match.group(2).strip()
            # Keep the first occurrence: the tabular list defines each code
            # once, but a code can also appear inside an "Excludes"/"Use
            # additional code" cross-reference line elsewhere in the PDF.
            if code not in lookup:
                lookup[code] = description
    doc.close()
    return lookup


def main():
    args = parse_args()

    if not os.path.exists(args.pdf_path):
        print(f"[build_icd10_lookup] PDF not found at {args.pdf_path}")
        sys.exit(1)

    print(f"[build_icd10_lookup] Parsing {args.pdf_path} ...")
    lookup = parse_icd10_pdf(args.pdf_path)

    if not lookup:
        print("[build_icd10_lookup] No ICD-10 codes parsed - check the PDF format.")
        sys.exit(1)

    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)
    with open(args.out_path, "w", encoding="utf-8") as f:
        json.dump(lookup, f, indent=2, ensure_ascii=False, sort_keys=True)

    print(f"[build_icd10_lookup] Parsed {len(lookup)} codes")
    print(f"[build_icd10_lookup] Saved -> {args.out_path}")


if __name__ == "__main__":
    main()
