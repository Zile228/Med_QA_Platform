"""
services/orchestrator/rag/faiss_store.py
FAISS-based RAG retrieval for clinical documents.

Workflow:
  1. Build (offline):   scripts/build_vectordb.py runs once, indexes PDFs -> disk
  2. Retrieve (online): FAISSStore.retrieve(...) -> List[str]

Metadata (source_file, page_number, page_end, section_heading, organ) is
loaded from metadata.pkl if present. An index without metadata.pkl still
works, falling back to a "general" placeholder for every chunk.

organ_filter only matches "breast" or "thyroid" exactly; any other value
(including "chest", "unknown", or "general" itself) returns no results
instead of silently falling back -- see _filter_by_organ() for the reasoning.

Public API:
    FAISSStore(index_path, docs_path)
    FAISSStore.retrieve(query, k, organ_filter) -> List[str]
    FAISSStore.retrieve_with_meta(query, k, organ_filter) -> List[dict]
    FAISSStore.is_ready() -> bool
"""

import os
import pickle
from typing import List, Optional


class FAISSStore:
    """
    Loads a FAISS index from disk -> retrieves top-k chunks for a query.

    If the index does not exist yet -> is_ready() = False -> the orchestrator
    falls back to an empty context (the LLM still runs, no crash).
    """

    def __init__(
        self,
        index_path: str = None,
        chunks_path: str = None,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        self.index_path = index_path or os.getenv(
            "FAISS_INDEX_PATH",
            "services/orchestrator/rag/vectordb/index.faiss"
        )
        self.chunks_path = chunks_path or self.index_path.replace(
            "index.faiss", "chunks.pkl"
        )
        self.metadata_path = self.index_path.replace("index.faiss", "metadata.pkl")
        self.embedding_model_name    = embedding_model
        self.cross_encoder_model_name = cross_encoder_model
        self._index = None
        self._chunks: List[str] = []
        self._metadata: List[dict] = []
        self._embedder = None
        self._cross_encoder = None

        self._try_load()

    def _try_load(self):
        """Loads the FAISS index, chunks, and metadata from disk. Does not throw if missing."""
        try:
            import faiss
            if not os.path.exists(self.index_path):
                print(f"[rag] Index does not exist yet: {self.index_path}")
                print("[rag] Run scripts/build_vectordb.py to build the index.")
                return

            self._index = faiss.read_index(self.index_path)

            if os.path.exists(self.chunks_path):
                with open(self.chunks_path, "rb") as f:
                    self._chunks = pickle.load(f)

            if os.path.exists(self.metadata_path):
                with open(self.metadata_path, "rb") as f:
                    self._metadata = pickle.load(f)
            else:
                # Old index without metadata: create a placeholder to avoid crashing
                self._metadata = [
                    {"source_file": "unknown", "page_number": 0, "organ": "general"}
                    for _ in self._chunks
                ]

            self._load_embedder()
            self._load_cross_encoder()
            print(
                f"[rag] FAISS index loaded - "
                f"{self._index.ntotal} vectors, {len(self._chunks)} chunks"
            )

        except ImportError:
            print("[rag] faiss-cpu is not installed -> RAG disabled.")
        except Exception as e:
            print(f"[rag] Load failed: {e} -> RAG disabled.")

    def _load_embedder(self):
        """Load the embedding model."""
        try:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(self.embedding_model_name)
        except ImportError:
            print("[rag] sentence-transformers is not installed -> RAG disabled.")

    def _load_cross_encoder(self):
        """
        Loads the cross-encoder model once at startup, caches it in self._cross_encoder.
        If not installed, self._cross_encoder = None and rerank() falls back to the original order.
        """
        try:
            from sentence_transformers import CrossEncoder
            self._cross_encoder = CrossEncoder(self.cross_encoder_model_name)
            print(f"[rag] CrossEncoder loaded: {self.cross_encoder_model_name}")
        except ImportError:
            print("[rag] cross-encoder is not installed -> rerank disabled, using raw FAISS score order.")
        except Exception as e:
            print(f"[rag] CrossEncoder load error: {e} -> rerank disabled.")

    def is_ready(self) -> bool:
        return (
            self._index is not None
            and len(self._chunks) > 0
            and self._embedder is not None
        )

    def _embed_query(self, query: str):
        return self._embedder.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        )

    def _filter_by_organ(self, indices, distances, organ_filter: Optional[str]):
        """
        Keeps chunks whose metadata.organ matches organ_filter exactly.
        If organ_filter is None, returns everything (no filtering).

        After build_vectordb.py's allow-list (see ALLOWED_PDF_FILENAMES),
        every indexed chunk is tagged organ="breast" or organ="thyroid";
        the "general" placeholder is no longer expected to have real
        clinical content behind it. organ_filter values outside
        {"breast", "thyroid"} (e.g. "chest", "unknown") therefore return
        no results instead of silently falling back to "general" -- a
        caller should treat that as "no clinical guideline coverage for
        this organ" rather than receive unrelated chunks.
        """
        if organ_filter is None:
            return [
                (int(idx), float(dist))
                for idx, dist in zip(indices, distances)
                if 0 <= idx < len(self._chunks)
            ]

        if organ_filter not in ("breast", "thyroid"):
            return []

        result = []
        for idx, dist in zip(indices, distances):
            idx = int(idx)
            if not (0 <= idx < len(self._chunks)):
                continue
            meta = self._metadata[idx] if idx < len(self._metadata) else {}
            chunk_organ = meta.get("organ", "general")
            if chunk_organ == organ_filter:
                result.append((idx, float(dist)))
        return result

    def retrieve(
        self,
        query: str,
        k: int = 3,
        organ_filter: Optional[str] = None,
    ) -> List[str]:
        """
        Retrieve the top-k chunks relevant to the query.

        Args:
            query:        clinical question or finding description
            k:            number of chunks to return
            organ_filter: 'breast' | 'thyroid' | None (no filtering)

        Returns:
            List[str] - text chunks, empty list if RAG is not ready
        """
        metas = self.retrieve_with_meta(query, k, organ_filter)
        return [m["chunk"] for m in metas]

    def retrieve_with_meta(
        self,
        query: str,
        k: int = 3,
        organ_filter: Optional[str] = None,
    ) -> List[dict]:
        """
        Retrieve the top-k chunks with full metadata.

        Returns a list of dicts:
            {"chunk": str, "source_file": str, "page_number": int,
             "page_end": int, "section_heading": str | None,
             "organ": str, "score": float}
        """
        if not self.is_ready():
            return []

        try:
            query_vec = self._embed_query(query)
            # Search for more than k to have enough candidates after organ filtering
            search_k = min(k * 5, self._index.ntotal)
            distances, indices = self._index.search(query_vec, search_k)

            filtered = self._filter_by_organ(indices[0], distances[0], organ_filter)

            results = []
            seen_texts = set()
            for idx, dist in filtered:
                text = self._chunks[idx]
                if text in seen_texts:
                    continue
                seen_texts.add(text)
                meta = self._metadata[idx] if idx < len(self._metadata) else {}
                page_number = meta.get("page_number", 0)
                results.append({
                    "chunk":           text,
                    "source_file":     meta.get("source_file", "unknown"),
                    "page_number":     page_number,
                    "page_end":        meta.get("page_end", page_number),
                    "section_heading": meta.get("section_heading"),
                    "organ":           meta.get("organ", "general"),
                    "score":           round(dist, 4),
                })
                if len(results) >= k:
                    break

            return results

        except Exception as e:
            print(f"[rag] Retrieve error: {e}")
            return []

    def rerank(self, query: str, candidates: List[dict], top_n: int = 3) -> List[dict]:
        """
        Re-ranks the chunk list using the cached cross-encoder, keeps the top_n highest.

        candidates: list of dicts from retrieve_with_meta (must have a "chunk" field).
        Falls back to the original FAISS score order if the cross-encoder isn't loaded.
        """
        if not candidates:
            return candidates

        if self._cross_encoder is None:
            return candidates[:top_n]

        try:
            pairs = [(query, c["chunk"]) for c in candidates]
            scores = self._cross_encoder.predict(pairs)
            ranked = sorted(
                zip(scores, candidates),
                key=lambda x: x[0],
                reverse=True,
            )
            return [item for _, item in ranked[:top_n]]
        except Exception as e:
            print(f"[rag] Rerank error: {e}")
            return candidates[:top_n]

    def retrieve_with_scores(self, query: str, k: int = 3) -> List[dict]:
        """
        Like retrieve() but includes scores.
        Used for debugging, not used in the main pipeline.
        """
        return self.retrieve_with_meta(query, k)
