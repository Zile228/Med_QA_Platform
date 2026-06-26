"""
scripts/build_vectordb.py

Offline script, run once to index clinical PDFs into FAISS.

Each PDF is converted to Markdown (page by page, with a page marker before
each page) so that section headings and tables survive the conversion.
Text is then split first by Markdown heading, then by token count within
each heading section. This keeps chunks aligned with the document structure
instead of cutting mid-sentence at a fixed character offset.

Each chunk's metadata stores source_file, page_number (start page),
page_end, section_heading, and organ. organ is assigned from the PDF
filename:
    - name contains "breast" or "birads" -> "breast"
    - name contains "thyroid" or "tirads" -> "thyroid"
    - otherwise -> "general"

Usage:
    python scripts/build_vectordb.py
    python scripts/build_vectordb.py --docs_dir rag/docs --out_dir rag/vectordb
"""

import os
import re
import sys
import time
import pickle
import argparse
import glob
import multiprocessing as mp


PAGE_MARKER_RE = re.compile(r"<!--page:(\d+)-->")

# pymupdf4llm wraps OCR'd image/table text with these markers and uses
# <br> as its line break; neither has semantic value for embedding.
PICTURE_TEXT_MARKER_RE = re.compile(
    r"\*{0,2}-{3,} (?:Start|End) of picture text -{3,}\*{0,2}"
)
BR_TAG_RE = re.compile(r"<br>")

# Average seconds per page, used only to print an ETA before processing.
# Conservative estimate so the printed ETA does not undersell large PDFs.
SECONDS_PER_PAGE_ESTIMATE = 1.0

# Hard timeout per PDF. A page that gets the OCR engine stuck must not
# block the rest of the batch.
PDF_TIMEOUT_SECONDS = 30 * 60


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--docs_dir", default="services/orchestrator/rag/docs",
                   help="Directory containing clinical PDFs")
    p.add_argument("--out_dir",  default="services/orchestrator/rag/vectordb",
                   help="Output dir for the FAISS index + chunks.pkl")
    p.add_argument("--chunk_tokens", type=int, default=200,
                   help="Target number of tokens per chunk")
    p.add_argument("--overlap_tokens", type=int, default=30,
                   help="Token overlap between consecutive chunks")
    p.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2",
                   help="Embedding model")
    return p.parse_args()


def _detect_organ(filename: str) -> str:
    """Assigns the organ based on the source PDF filename."""
    name = filename.lower()
    if "breast" in name or "birads" in name or "bi-rads" in name:
        return "breast"
    if "thyroid" in name or "tirads" in name or "ti-rads" in name:
        return "thyroid"
    return "general"


def _count_pages(pdf_path: str) -> int:
    """Returns the page count, or 0 if the PDF cannot be opened at all."""
    try:
        import fitz
        doc = fitz.open(pdf_path)
        n = len(doc)
        doc.close()
        return n
    except Exception as e:
        print(f"  [WARN] Could not open {pdf_path} to count pages: {e}")
        return 0


def _markdown_worker(pdf_path: str, result_queue: mp.Queue):
    """Runs the actual conversion in a subprocess so it can be killed on timeout."""
    try:
        import pymupdf4llm
        page_dicts = pymupdf4llm.to_markdown(pdf_path, page_chunks=True)
        parts = []
        for i, page_dict in enumerate(page_dicts, start=1):
            text = (page_dict.get("text") or "").strip()
            if text:
                parts.append(f"<!--page:{i}-->\n{text}")
        result_queue.put(("ok", "\n\n".join(parts)))
    except Exception as e:
        result_queue.put(("error", str(e)))


def pdf_to_marked_markdown(pdf_path: str) -> str:
    """
    Converts a PDF to a single Markdown string, with a <!--page:N--> marker
    inserted before the content of each page so page boundaries can be
    recovered after chunking.

    Runs the conversion in a separate process with a hard timeout, so a
    single page that hangs the OCR engine cannot block the rest of the
    batch. Returns an empty string if the PDF cannot be read, times out,
    or produces no extractable content.
    """
    page_count = _count_pages(pdf_path)
    if page_count == 0:
        return ""

    eta_seconds = page_count * SECONDS_PER_PAGE_ESTIMATE
    print(f"    {page_count} pages, estimated time: {eta_seconds / 60:.1f} min "
          f"(hard timeout: {PDF_TIMEOUT_SECONDS / 60:.0f} min)")

    result_queue = mp.Queue()
    proc = mp.Process(target=_markdown_worker, args=(pdf_path, result_queue))
    start = time.time()
    proc.start()
    proc.join(timeout=PDF_TIMEOUT_SECONDS)

    if proc.is_alive():
        proc.terminate()
        proc.join()
        print(f"  [WARN] {os.path.basename(pdf_path)} exceeded "
              f"{PDF_TIMEOUT_SECONDS / 60:.0f} min timeout -- skipping.")
        return ""

    elapsed = time.time() - start
    if result_queue.empty():
        print(f"  [WARN] {os.path.basename(pdf_path)} produced no result "
              f"(process likely crashed) after {elapsed:.0f}s -- skipping.")
        return ""

    status, payload = result_queue.get()
    if status == "error":
        print(f"  [WARN] Could not read {pdf_path}: {payload}")
        return ""

    print(f"    Done in {elapsed:.0f}s")
    return payload


def split_into_sections(marked_markdown: str) -> list:
    """
    Splits the marked markdown into sections by heading (H1-H4).

    Returns a list of dicts: {"heading": str | None, "text": str}.
    Content before the first heading is kept as a section with heading=None.
    """
    from langchain_text_splitters import MarkdownHeaderTextSplitter

    headers_to_split_on = [("#", "h1"), ("##", "h2"), ("###", "h3"), ("####", "h4")]
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False,
    )
    docs = splitter.split_text(marked_markdown)

    sections = []
    for doc in docs:
        heading = None
        for level in ("h4", "h3", "h2", "h1"):
            if level in doc.metadata:
                heading = doc.metadata[level]
                break
        sections.append({"heading": heading, "text": doc.page_content})
    return sections


def _extract_page_range(text_with_markers: str, carry_page: int) -> tuple:
    """
    Finds the first and last <!--page:N--> markers inside the text.

    carry_page is the last known page number from the previous section,
    used as a fallback when a section contains no marker of its own
    (e.g. a short section squeezed between two markers by the splitter).

    Returns (page_start, page_end, next_carry_page).
    """
    pages = [int(m) for m in PAGE_MARKER_RE.findall(text_with_markers)]
    if not pages:
        return carry_page, carry_page, carry_page
    return min(pages), max(pages), max(pages)


def chunk_section(
    text: str,
    embedder,
    chunk_tokens: int,
    overlap_tokens: int,
) -> list:
    """
    Splits one section's text into token-bounded chunks, breaking at
    paragraph/sentence/word boundaries rather than at a fixed character
    offset. Page markers and heading lines are stripped from the chunk
    text returned, but are used beforehand to compute page numbers.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    tokenizer = embedder.tokenizer

    def token_len(s: str) -> int:
        return len(tokenizer.encode(s, add_special_tokens=False))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_tokens,
        chunk_overlap=overlap_tokens,
        length_function=token_len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    raw_chunks = splitter.split_text(text)

    cleaned = []
    for raw_chunk in raw_chunks:
        without_markers = PAGE_MARKER_RE.sub("", raw_chunk)
        without_markers = PICTURE_TEXT_MARKER_RE.sub("", without_markers)
        without_markers = BR_TAG_RE.sub(" ", without_markers)
        without_markers = without_markers.strip()
        if len(without_markers) > 30:
            cleaned.append((raw_chunk, without_markers))
    return cleaned


def process_pdf(pdf_path: str, embedder, chunk_tokens: int, overlap_tokens: int) -> tuple:
    """
    Runs the full pipeline for one PDF: extract -> sectionize -> chunk.

    Returns (chunks: list[str], metadata: list[dict]). Returns ([], [])
    if the PDF could not be processed, so the caller can skip it without
    crashing the whole batch.
    """
    filename = os.path.basename(pdf_path)
    organ = _detect_organ(filename)

    marked_markdown = pdf_to_marked_markdown(pdf_path)
    if not marked_markdown.strip():
        print(f"  [WARN] No extractable content in {filename} -- skipping.")
        return [], []

    sections = split_into_sections(marked_markdown)

    chunks = []
    metadata = []
    carry_page = 1
    for section in sections:
        section_chunks = chunk_section(section["text"], embedder, chunk_tokens, overlap_tokens)
        for raw_chunk, clean_text in section_chunks:
            page_start, page_end, carry_page = _extract_page_range(raw_chunk, carry_page)
            chunks.append(clean_text)
            metadata.append({
                "source_file": filename,
                "page_number": page_start,
                "page_end": page_end,
                "section_heading": section["heading"],
                "organ": organ,
            })
    return chunks, metadata


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    pdf_files = glob.glob(os.path.join(args.docs_dir, "**/*.pdf"), recursive=True)
    if not pdf_files:
        print(f"[build_vectordb] No PDF found in {args.docs_dir}")
        print("Place clinical PDF files in the docs/ directory then run again.")
        sys.exit(0)

    print(f"[build_vectordb] Found {len(pdf_files)} PDF files")

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("ERROR: pip install sentence-transformers")
        sys.exit(1)

    print(f"[build_vectordb] Loading embedding model {args.model}...")
    embedder = SentenceTransformer(args.model)

    all_chunks = []
    all_metadata = []

    for pdf_path in pdf_files:
        filename = os.path.basename(pdf_path)
        print(f"  Processing: {filename}")
        try:
            chunks, metadata = process_pdf(pdf_path, embedder, args.chunk_tokens, args.overlap_tokens)
        except Exception as e:
            print(f"  [WARN] Unexpected error processing {filename}: {e} -- skipping.")
            continue

        all_chunks.extend(chunks)
        all_metadata.extend(metadata)
        print(f"    -> {len(chunks)} chunks (organ={_detect_organ(filename)})")

    print(f"[build_vectordb] Total chunks: {len(all_chunks)}")

    if not all_chunks:
        print("[build_vectordb] No chunks found - check the PDF files again.")
        sys.exit(1)

    print("[build_vectordb] Embedding chunks...")
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
    # IndexFlatIP: inner product is equivalent to cosine when vectors are normalized
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
    print("Done! Restart the orchestrator service to load the new index.")


if __name__ == "__main__":
    main()
