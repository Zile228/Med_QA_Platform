"""
scripts/build_vectordb.py
==========================
Offline script - chạy 1 lần để index PDF lâm sàng vào FAISS.

Usage:
    python scripts/build_vectordb.py
    python scripts/build_vectordb.py --docs_dir rag/docs --out_dir rag/vectordb

Requires:
    pip install faiss-cpu sentence-transformers pypdf2
"""

import os
import sys
import pickle
import argparse
import glob

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--docs_dir", default="services/orchestrator/rag/docs",
                   help="Thư mục chứa PDF lâm sàng")
    p.add_argument("--out_dir",  default="services/orchestrator/rag/vectordb",
                   help="Output dir cho FAISS index + chunks.pkl")
    p.add_argument("--chunk_size", type=int, default=400,
                   help="Số ký tự mỗi chunk")
    p.add_argument("--overlap",    type=int, default=50,
                   help="Overlap giữa các chunk")
    p.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2",
                   help="Embedding model")
    return p.parse_args()


def extract_text_from_pdf(pdf_path: str) -> str:
    try:
        import PyPDF2
        text = ""
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() or ""
        return text
    except Exception as e:
        print(f"  [WARN] Không đọc được {pdf_path}: {e}")
        return ""


def chunk_text(text: str, chunk_size: int, overlap: int) -> list:
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end].strip())
        start += chunk_size - overlap
    return [c for c in chunks if len(c) > 50]


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Thu thap danh sach PDF
    pdf_files = glob.glob(os.path.join(args.docs_dir, "**/*.pdf"), recursive=True)
    if not pdf_files:
        print(f"[build_vectordb] Không tìm thấy PDF trong {args.docs_dir}")
        print("Đặt file PDF lâm sàng (BI-RADS, ACR Thyroid, ...) vào thư mục docs/ rồi chạy lại.")
        sys.exit(0)

    print(f"[build_vectordb] Tìm thấy {len(pdf_files)} PDF files")

    # Trich xuat text va chia chunk
    all_chunks = []
    for pdf_path in pdf_files:
        print(f"  Processing: {os.path.basename(pdf_path)}")
        text = extract_text_from_pdf(pdf_path)
        if text:
            chunks = chunk_text(text, args.chunk_size, args.overlap)
            all_chunks.extend(chunks)
            print(f"    -> {len(chunks)} chunks")

    print(f"[build_vectordb] Total chunks: {len(all_chunks)}")

    if not all_chunks:
        print("[build_vectordb] Không có chunk nào - kiểm tra lại PDF files.")
        sys.exit(1)

    # Embed cac chunk thanh vector
    print(f"[build_vectordb] Embedding với {args.model}...")
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

    # Tao FAISS index
    try:
        import faiss
    except ImportError:
        print("ERROR: pip install faiss-cpu")
        sys.exit(1)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)   # Inner product, tuong duong cosine khi vector da normalize
    index.add(embeddings)

    # Luu index va chunk ra disk
    index_path  = os.path.join(args.out_dir, "index.faiss")
    chunks_path = os.path.join(args.out_dir, "chunks.pkl")
    faiss.write_index(index, index_path)
    with open(chunks_path, "wb") as f:
        pickle.dump(all_chunks, f)

    print(f"[build_vectordb] Saved index -> {index_path} ({index.ntotal} vectors)")
    print(f"[build_vectordb] Saved chunks -> {chunks_path}")
    print("Done! Restart orchestrator service để load index mới.")


if __name__ == "__main__":
    main()
