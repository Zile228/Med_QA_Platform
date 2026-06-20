"""
scripts/build_vectordb.py
==========================
Offline script - chay 1 lan de index PDF lam sang vao FAISS.

Moi chunk duoc luu kem metadata: source_file, page_number, organ.
organ duoc gan tu ten file PDF:
    - ten chua "breast" hoac "birads" -> "breast"
    - ten chua "thyroid" hoac "tirads" -> "thyroid"
    - con lai -> "general"

Metadata nay duoc FAISSStore dung de loc theo organ (organ_filter)
va tra ve citation (file + trang) thay vi placeholder string.

Usage:
    python scripts/build_vectordb.py
    python scripts/build_vectordb.py --docs_dir rag/docs --out_dir rag/vectordb
"""

import os
import sys
import pickle
import argparse
import glob


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--docs_dir", default="services/orchestrator/rag/docs",
                   help="Thu muc chua PDF lam sang")
    p.add_argument("--out_dir",  default="services/orchestrator/rag/vectordb",
                   help="Output dir cho FAISS index + chunks.pkl")
    p.add_argument("--chunk_size", type=int, default=400,
                   help="So ky tu moi chunk")
    p.add_argument("--overlap",    type=int, default=50,
                   help="Overlap giua cac chunk")
    p.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2",
                   help="Embedding model")
    return p.parse_args()


def _detect_organ(filename: str) -> str:
    """Gan organ dua tren ten file PDF nguon."""
    name = filename.lower()
    if "breast" in name or "birads" in name or "bi-rads" in name:
        return "breast"
    if "thyroid" in name or "tirads" in name or "ti-rads" in name:
        return "thyroid"
    return "general"


def extract_text_by_page(pdf_path: str) -> list:
    """
    Trich xuat text tung trang tu PDF, tra ve list dict {page, text}.
    Tra ve list rong neu khong doc duoc.
    """
    try:
        import PyPDF2
        pages = []
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page_num, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                if text.strip():
                    pages.append({"page": page_num, "text": text})
        return pages
    except Exception as e:
        print(f"  [WARN] Khong doc duoc {pdf_path}: {e}")
        return []


def chunk_page(text: str, chunk_size: int, overlap: int) -> list:
    """Chia text cua 1 trang thanh cac chunk theo chunk_size va overlap."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if len(chunk) > 50:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    pdf_files = glob.glob(os.path.join(args.docs_dir, "**/*.pdf"), recursive=True)
    if not pdf_files:
        print(f"[build_vectordb] Khong tim thay PDF trong {args.docs_dir}")
        print("Dat file PDF lam sang vao thu muc docs/ roi chay lai.")
        sys.exit(0)

    print(f"[build_vectordb] Tim thay {len(pdf_files)} PDF files")

    all_chunks = []
    all_metadata = []

    for pdf_path in pdf_files:
        filename = os.path.basename(pdf_path)
        organ = _detect_organ(filename)
        print(f"  Processing: {filename} (organ={organ})")

        pages = extract_text_by_page(pdf_path)
        file_chunk_count = 0
        for page_info in pages:
            page_num = page_info["page"]
            chunks = chunk_page(page_info["text"], args.chunk_size, args.overlap)
            for chunk in chunks:
                all_chunks.append(chunk)
                all_metadata.append({
                    "source_file": filename,
                    "page_number": page_num,
                    "organ": organ,
                })
            file_chunk_count += len(chunks)

        print(f"    -> {file_chunk_count} chunks, {len(pages)} trang")

    print(f"[build_vectordb] Tong so chunk: {len(all_chunks)}")

    if not all_chunks:
        print("[build_vectordb] Khong co chunk nao - kiem tra lai PDF files.")
        sys.exit(1)

    print(f"[build_vectordb] Embedding voi {args.model}...")
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("ERROR: pip install sentence-transformers")
        sys.exit(1)

    embedder = SentenceTransformer(args.model)
    embeddings = embedder.encode(
        all_chunks,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    print(f"[build_vectordb] Embeddings shape: {embeddings.shape}")

    try:
        import faiss
    except ImportError:
        print("ERROR: pip install faiss-cpu")
        sys.exit(1)

    dim = embeddings.shape[1]
    # IndexFlatIP: inner product tuong duong cosine khi vector da normalize
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    index_path    = os.path.join(args.out_dir, "index.faiss")
    chunks_path   = os.path.join(args.out_dir, "chunks.pkl")
    metadata_path = os.path.join(args.out_dir, "metadata.pkl")

    faiss.write_index(index, index_path)
    with open(chunks_path, "wb") as f:
        pickle.dump(all_chunks, f)
    with open(metadata_path, "wb") as f:
        pickle.dump(all_metadata, f)

    print(f"[build_vectordb] Saved index    -> {index_path} ({index.ntotal} vectors)")
    print(f"[build_vectordb] Saved chunks   -> {chunks_path}")
    print(f"[build_vectordb] Saved metadata -> {metadata_path}")
    print("Done! Restart orchestrator service de load index moi.")


if __name__ == "__main__":
    main()
