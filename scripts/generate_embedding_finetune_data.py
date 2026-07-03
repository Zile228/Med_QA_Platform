"""
scripts/generate_embedding_finetune_data.py

Sinh du lieu fine-tune cho embedding model (all-MiniLM-L6-v2) tu chinh cac
chunk da lam sach trong vectordb hien tai. Thuc hien TODO_FINETUNE_RAG.md
Phan 1.2 (sinh cap query/positive_chunk) va Phan 1.3 (chon hard negative).

Ky thuat: voi moi chunk, dung LLM sinh 1-2 cau hoi ma chunk do la cau tra
loi phu hop nhat (nguoc voi HyDE: sinh cau hoi tu doan van co san, thay vi
sinh cau tra loi gia dinh tu cau hoi).

Hard negative 2 loai cho moi cap (query, positive_chunk):
  loai 1 (nham organ): 1-2 chunk ngau nhien co organ khac (breast<->thyroid).
  loai 2 (cung organ, khac section): 1 chunk cung source_file nhung khac
    section_heading, de model hoc phan biet trong noi bo 1 tai lieu.

Input: services/orchestrator/rag/vectordb/chunks.pkl, metadata.pkl (danh so
song song, idx cua chunks[i] khop metadata[i] -- xem scripts/build_vectordb.py
process_pdf()).

Output: JSONL, moi dong:
  {"query": str, "positive_chunk": str, "organ": str, "source_file": str,
   "hard_negatives": [str, ...]}

Chay:
  python scripts/generate_embedding_finetune_data.py \\
    --chunks_path   services/orchestrator/rag/vectordb/chunks.pkl \\
    --metadata_path services/orchestrator/rag/vectordb/metadata.pkl \\
    --out_file      scripts/finetune_data/embedding_training.jsonl \\
    [--questions_per_chunk 1] \\
    [--min_chunk_chars 100] \\
    [--rate_limit 10] \\
    [--max_retries 3] \\
    [--retry_base_delay 2.0] \\
    [--resume] \\
    [--max_chunks N]

LLM_BACKEND phai la "google" hoac "openai" (can generate() text on dinh cho
hang nghin chunk). "ollama"/"local_hf"/"remote" van chay duoc ve mat ky
thuat nhung khong duoc test o quy mo nay.
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Script nay chay standalone, .env khong tu doc nhu trong container.
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

import pickle
import os
import threading
import time


class RateLimitedLLMClient:
    """
    Wrap llm_client, dam bao khong goi .generate() qua N lan/phut.
    Dung sliding-window 60s, cong them retry-with-exponential-backoff.

    Cung logic voi scripts/generate_finetune_data.py, khong import truc
    tiep tu do vi module kia keo theo toan bo dependency vision/torch.
    """

    def __init__(
        self,
        inner_client,
        max_calls_per_minute: int = 10,
        max_retries: int = 3,
        retry_base_delay: float = 2.0,
    ):
        self._inner = inner_client
        self._max_calls = max_calls_per_minute
        self._max_retries = max_retries
        self._retry_base_delay = retry_base_delay
        self._timestamps = []
        self._lock = threading.Lock()

    def _wait_for_slot(self):
        """Block (sleep) cho den khi co cho trong sliding window 60s."""
        if not self._max_calls or self._max_calls <= 0:
            return
        while True:
            with self._lock:
                now = time.monotonic()
                self._timestamps = [t for t in self._timestamps if now - t < 60]
                if len(self._timestamps) < self._max_calls:
                    self._timestamps.append(now)
                    return
                sleep_time = 60 - (now - self._timestamps[0]) + 0.05
            if sleep_time > 0:
                print(
                    f"    [rate_limit] da dat {self._max_calls} request/phut, "
                    f"cho {sleep_time:.1f}s truoc khi goi tiep...",
                    flush=True,
                )
                time.sleep(sleep_time)

    def generate(self, *args, **kwargs):
        last_exc = None
        for attempt in range(self._max_retries + 1):
            self._wait_for_slot()
            try:
                return self._inner.generate(*args, **kwargs)
            except Exception as e:  # noqa: BLE001 - bat moi loai loi tu API/network
                last_exc = e
                if attempt >= self._max_retries:
                    break
                delay = self._retry_base_delay * (2 ** attempt)
                print(
                    f"    [retry] llm_client.generate() loi (lan {attempt + 1}"
                    f"/{self._max_retries + 1}): {type(e).__name__}: {e} "
                    f"-> cho {delay:.1f}s roi thu lai...",
                    flush=True,
                )
                time.sleep(delay)
        raise last_exc

    def __getattr__(self, name):
        return getattr(self._inner, name)


QUESTION_GEN_SYSTEM_PROMPT = """You are helping build a retrieval training set for a clinical breast/thyroid ultrasound RAG system.
Given a passage from a clinical guideline, write realistic questions that a clinician using this system would ask, for which this passage is the best available answer.
Rules:
- Questions must be answerable using only the given passage.
- Do not quote the passage verbatim; write questions in your own words.
- Do not mention "the passage", "the text", or "the document" in the question.
- Return ONLY a JSON array of strings, nothing else. No markdown fences, no preamble."""


def _build_question_gen_prompt(chunk_text: str, organ: str, n_questions: int) -> str:
    return (
        f"Organ context: {organ}\n\n"
        f"Passage:\n{chunk_text}\n\n"
        f"Write {n_questions} question(s) for which this passage is the best answer. "
        f'Return a JSON array of {n_questions} string(s), e.g. ["question 1"].'
    )


def _parse_questions(raw: str, n_expected: int) -> list:
    """
    Parses the LLM's JSON array response. Returns [] on any parse failure
    so the caller can skip the chunk instead of crashing the whole run.
    """
    clean = raw.strip()
    if clean.startswith("```"):
        clean = clean.strip("`")
        if clean.lower().startswith("json"):
            clean = clean[4:]
    clean = clean.strip()
    try:
        parsed = json.loads(clean)
    except Exception:
        return []
    if not isinstance(parsed, list):
        return []
    questions = [str(q).strip() for q in parsed if str(q).strip()]
    return questions[:n_expected] if questions else []


def _load_vectordb(chunks_path: str, metadata_path: str) -> tuple:
    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)
    if len(chunks) != len(metadata):
        raise ValueError(
            f"chunks.pkl co {len(chunks)} phan tu nhung metadata.pkl co "
            f"{len(metadata)} phan tu -- 2 file khong khop, kiem tra lai nguon du lieu."
        )
    return chunks, metadata


def _select_hard_negatives(
    idx: int,
    chunks: list,
    metadata: list,
    by_organ: dict,
    by_source: dict,
    rng: random.Random,
    n_organ_negatives: int = 2,
    n_section_negatives: int = 1,
) -> list:
    """
    Chon hard negative cho chunk o vi tri idx.

    Loai 1 (nham organ): chunk ngau nhien co organ khac (breast<->thyroid).
    Loai 2 (cung organ, khac section): chunk cung source_file nhung khac
    section_heading.

    Khong dung random negative tu toan vectordb lam nguon chinh, vi qua de
    phan biet va khong phan anh dung loai nhieu he thong dang gap phai
    (xem TODO_FINETUNE_RAG.md Phan 1.3).
    """
    meta = metadata[idx]
    organ = meta.get("organ")
    source_file = meta.get("source_file")
    section_heading = meta.get("section_heading")

    negatives = []

    other_organs = [o for o in by_organ if o != organ]
    other_organ_pool = []
    for o in other_organs:
        other_organ_pool.extend(by_organ[o])
    if other_organ_pool:
        k = min(n_organ_negatives, len(other_organ_pool))
        for neg_idx in rng.sample(other_organ_pool, k):
            negatives.append(chunks[neg_idx])

    same_source_pool = [
        i for i in by_source.get(source_file, [])
        if i != idx and metadata[i].get("section_heading") != section_heading
    ]
    if same_source_pool:
        k = min(n_section_negatives, len(same_source_pool))
        for neg_idx in rng.sample(same_source_pool, k):
            negatives.append(chunks[neg_idx])

    return negatives


def _load_done_indices(out_file: Path) -> set:
    """Doc chunk_idx da xu ly tu file JSONL cu, ho tro --resume."""
    done = set()
    if not out_file.exists():
        return done
    with open(out_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                if "chunk_idx" in record:
                    done.add(record["chunk_idx"])
            except Exception:
                continue
    return done


def _append_jsonl(out_file: Path, record: dict):
    with open(out_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Sinh du lieu fine-tune embedding tu chunk trong vectordb da lam sach"
    )
    parser.add_argument(
        "--chunks_path", default="services/orchestrator/rag/vectordb/chunks.pkl"
    )
    parser.add_argument(
        "--metadata_path", default="services/orchestrator/rag/vectordb/metadata.pkl"
    )
    parser.add_argument(
        "--out_file", default="scripts/finetune_data/embedding_training.jsonl"
    )
    parser.add_argument("--questions_per_chunk", type=int, default=1)
    parser.add_argument(
        "--min_chunk_chars", type=int, default=100,
        help="Bo qua chunk ngan hon nguong nay (vd header trang, khong du "
             "ngu canh de sinh cau hoi co y nghia).",
    )
    parser.add_argument("--n_organ_negatives", type=int, default=2)
    parser.add_argument("--n_section_negatives", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rate_limit", type=int, default=10)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--retry_base_delay", type=float, default=2.0)
    parser.add_argument(
        "--resume", action="store_true",
        help="Bo qua chunk da co trong --out_file cu (theo chunk_idx).",
    )
    parser.add_argument(
        "--max_chunks", type=int, default=None,
        help="Gioi han TONG so chunk xu ly, lay tuan tu theo thu tu trong "
             "chunks.pkl. Thien vi ve file dung dau vectordb -- neu can lay "
             "deu theo tung source_file, dung --max_chunks_per_source thay the.",
    )
    parser.add_argument(
        "--max_chunks_per_source", type=int, default=None,
        help="Gioi han so chunk MOI source_file (lay ngau nhien theo --seed), "
             "dam bao ca 5 PDF deu co mat trong output du ngan sach nho. "
             "Uu tien hon --max_chunks neu ca 2 cung duoc truyen.",
    )
    args = parser.parse_args()

    out_file = Path(args.out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"[generate_embedding_finetune_data] Doc {args.chunks_path} / {args.metadata_path}...")
    chunks, metadata = _load_vectordb(args.chunks_path, args.metadata_path)
    print(f"[generate_embedding_finetune_data] {len(chunks)} chunk trong vectordb.")

    by_organ = {}
    by_source = {}
    for i, meta in enumerate(metadata):
        by_organ.setdefault(meta.get("organ"), []).append(i)
        by_source.setdefault(meta.get("source_file"), []).append(i)

    eligible = [
        i for i, c in enumerate(chunks)
        if len(c.strip()) >= args.min_chunk_chars
    ]
    n_skipped_short = len(chunks) - len(eligible)
    print(
        f"[generate_embedding_finetune_data] {len(eligible)} chunk du dai "
        f"(>= {args.min_chunk_chars} ky tu), {n_skipped_short} chunk bi bo qua vi qua ngan."
    )

    rng = random.Random(args.seed)

    if args.max_chunks_per_source:
        by_source_eligible = {}
        for i in eligible:
            by_source_eligible.setdefault(metadata[i]["source_file"], []).append(i)
        sampled = []
        for source_file, idxs in by_source_eligible.items():
            k = min(args.max_chunks_per_source, len(idxs))
            sampled.extend(rng.sample(idxs, k))
        eligible = sorted(sampled)
        print(
            f"[generate_embedding_finetune_data] --max_chunks_per_source={args.max_chunks_per_source}: "
            f"con {len(eligible)} chunk, lay deu tu {len(by_source_eligible)} source_file."
        )
    elif args.max_chunks:
        eligible = eligible[: args.max_chunks]
        print(f"[generate_embedding_finetune_data] --max_chunks: gioi han con {len(eligible)} chunk.")

    done_indices = _load_done_indices(out_file) if args.resume else set()
    if args.resume and done_indices:
        print(f"[generate_embedding_finetune_data] --resume: {len(done_indices)} chunk da co trong {out_file}, se bo qua.")

    backend = os.getenv("LLM_BACKEND", "ollama").lower()
    if backend not in ("google", "openai"):
        print(
            f"[generate_embedding_finetune_data] WARNING: LLM_BACKEND='{backend}' "
            "chua duoc kiem chung o quy mo hang nghin chunk. Khuyen nghi 'google' hoac 'openai'."
        )

    from services.orchestrator.llm_client import get_llm_client
    llm_client = get_llm_client()
    llm_client = RateLimitedLLMClient(
        llm_client,
        max_calls_per_minute=args.rate_limit,
        max_retries=args.max_retries,
        retry_base_delay=args.retry_base_delay,
    )

    n_written = 0
    n_skipped_resume = 0
    n_dropped_parse = 0
    n_total = len(eligible)

    for pos, idx in enumerate(eligible, start=1):
        if idx in done_indices:
            n_skipped_resume += 1
            continue

        chunk_text = chunks[idx]
        meta = metadata[idx]
        organ = meta.get("organ")
        source_file = meta.get("source_file")

        print(f"  [{pos}/{n_total}] chunk_idx={idx} organ={organ} source={source_file}")

        prompt = _build_question_gen_prompt(chunk_text, organ, args.questions_per_chunk)
        try:
            raw = llm_client.generate(prompt, QUESTION_GEN_SYSTEM_PROMPT)
        except Exception as e:
            print(f"    [skip] LLM call that bai: {type(e).__name__}: {e}")
            n_dropped_parse += 1
            continue

        questions = _parse_questions(raw, args.questions_per_chunk)
        if not questions:
            print(f"    [skip] Khong parse duoc JSON array cau hoi tu response.")
            n_dropped_parse += 1
            continue

        hard_negatives = _select_hard_negatives(
            idx, chunks, metadata, by_organ, by_source, rng,
            n_organ_negatives=args.n_organ_negatives,
            n_section_negatives=args.n_section_negatives,
        )

        for query in questions:
            record = {
                "query": query,
                "positive_chunk": chunk_text,
                "organ": organ,
                "source_file": source_file,
                "hard_negatives": hard_negatives,
                "chunk_idx": idx,
            }
            _append_jsonl(out_file, record)
            n_written += 1

    print(
        f"\n[generate_embedding_finetune_data] Hoan tat. "
        f"{n_written} cap ghi moi, {n_dropped_parse} chunk bi loai (LLM/parse loi), "
        f"{n_skipped_resume} chunk bo qua (--resume)."
    )
    print(f"[generate_embedding_finetune_data] Output: {out_file}")


if __name__ == "__main__":
    main()