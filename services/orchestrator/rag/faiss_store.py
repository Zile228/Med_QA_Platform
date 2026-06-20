"""
services/orchestrator/rag/faiss_store.py
==========================================
FAISS-based RAG retrieval cho clinical documents.

Workflow:
  1. Build (offline):   scripts/build_vectordb.py chay 1 lan, index PDFs -> disk
  2. Retrieve (online): FAISSStore.retrieve(...) -> List[str]

Metadata (source_file, page_number, organ) duoc load tu metadata.pkl neu co.
Index cu (khong co metadata.pkl) van hoat dong -- fallback tra ve chunk text thuan.

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
    Load FAISS index tu disk -> retrieve top-k chunks cho query.

    Neu index chua ton tai -> is_ready() = False -> orchestrator
    fallback sang empty context (LLM van chay, khong crash).
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
        """Load FAISS index, chunks va metadata tu disk. Khong throw neu chua co."""
        try:
            import faiss
            if not os.path.exists(self.index_path):
                print(f"[rag] Index chua ton tai: {self.index_path}")
                print("[rag] Chay scripts/build_vectordb.py de build index.")
                return

            self._index = faiss.read_index(self.index_path)

            if os.path.exists(self.chunks_path):
                with open(self.chunks_path, "rb") as f:
                    self._chunks = pickle.load(f)

            if os.path.exists(self.metadata_path):
                with open(self.metadata_path, "rb") as f:
                    self._metadata = pickle.load(f)
            else:
                # Index cu khong co metadata: tao placeholder de khong crash
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
            print("[rag] faiss-cpu chua install -> RAG disabled.")
        except Exception as e:
            print(f"[rag] Load failed: {e} -> RAG disabled.")

    def _load_embedder(self):
        """Load embedding model."""
        try:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(self.embedding_model_name)
        except ImportError:
            print("[rag] sentence-transformers chua install -> RAG disabled.")

    def _load_cross_encoder(self):
        """
        Load cross-encoder model 1 lan khi khoi dong, cache vao self._cross_encoder.
        Neu chua install, self._cross_encoder = None va rerank() fallback ve thu tu goc.
        """
        try:
            from sentence_transformers import CrossEncoder
            self._cross_encoder = CrossEncoder(self.cross_encoder_model_name)
            print(f"[rag] CrossEncoder loaded: {self.cross_encoder_model_name}")
        except ImportError:
            print("[rag] cross-encoder chua install -> rerank disabled, dung thu tu FAISS score.")
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
        Giu lai cac chunk co metadata.organ khop voi organ_filter hoac la "general".
        Neu organ_filter la None, tra ve tat ca (khong loc).
        """
        if organ_filter is None:
            return [
                (int(idx), float(dist))
                for idx, dist in zip(indices, distances)
                if 0 <= idx < len(self._chunks)
            ]

        result = []
        for idx, dist in zip(indices, distances):
            idx = int(idx)
            if not (0 <= idx < len(self._chunks)):
                continue
            meta = self._metadata[idx] if idx < len(self._metadata) else {}
            chunk_organ = meta.get("organ", "general")
            if chunk_organ in (organ_filter, "general"):
                result.append((idx, float(dist)))
        return result

    def retrieve(
        self,
        query: str,
        k: int = 3,
        organ_filter: Optional[str] = None,
    ) -> List[str]:
        """
        Retrieve top-k chunks lien quan den query.

        Args:
            query:        clinical question hoac finding description
            k:            so chunks tra ve
            organ_filter: 'breast' | 'thyroid' | None (khong loc)

        Returns:
            List[str] - text chunks, empty list neu RAG khong san sang
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
        Retrieve top-k chunks kem day du metadata.

        Tra ve list dict:
            {"chunk": str, "source_file": str, "page_number": int, "organ": str}
        """
        if not self.is_ready():
            return []

        try:
            query_vec = self._embed_query(query)
            # Tim nhieu hon k de co du ung vien sau khi loc theo organ
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
                results.append({
                    "chunk":       text,
                    "source_file": meta.get("source_file", "unknown"),
                    "page_number": meta.get("page_number", 0),
                    "organ":       meta.get("organ", "general"),
                    "score":       round(dist, 4),
                })
                if len(results) >= k:
                    break

            return results

        except Exception as e:
            print(f"[rag] Retrieve error: {e}")
            return []

    def rerank(self, query: str, candidates: List[dict], top_n: int = 3) -> List[dict]:
        """
        Re-rank danh sach chunk bang cross-encoder da duoc cache, giu lai top_n cao nhat.

        candidates: list dict tu retrieve_with_meta (co field "chunk").
        Fallback ve thu tu FAISS score goc neu cross-encoder chua load.
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
        Nhu retrieve() nhung kem score.
        Dung cho debugging, khong dung trong pipeline chinh.
        """
        return self.retrieve_with_meta(query, k)
