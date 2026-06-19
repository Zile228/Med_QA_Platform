"""
services/orchestrator/rag/faiss_store.py
==========================================
FAISS-based RAG retrieval cho clinical documents.

Workflow:
  1. Build (offline):   scripts/build_vectordb.py chạy 1 lần, index PDFs -> disk
  2. Retrieve (online): FAISSStore.retrieve(query, k=3) -> List[str]

Embedding: sentence-transformers/all-MiniLM-L6-v2 (lightweight, no GPU needed)

Public API:
    FAISSStore(index_path, docs_path)
    FAISSStore.retrieve(query: str, k: int) -> List[str]
    FAISSStore.is_ready() -> bool
"""

import os
import pickle
from typing import List, Optional


class FAISSStore:
    """
    Load FAISS index từ disk -> retrieve top-k chunks cho query.

    Nếu index chưa tồn tại -> is_ready() = False -> orchestrator
    fallback sang empty context (LLM vẫn chạy, không crash).
    """

    def __init__(
        self,
        index_path: str = None,
        chunks_path: str = None,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.index_path = index_path or os.getenv(
            "FAISS_INDEX_PATH",
            "services/orchestrator/rag/vectordb/index.faiss"
        )
        self.chunks_path = chunks_path or self.index_path.replace(
            "index.faiss", "chunks.pkl"
        )
        self.embedding_model_name = embedding_model
        self._index = None
        self._chunks: List[str] = []
        self._embedder = None

        self._try_load()

    def _try_load(self):
        """Load FAISS index va chunks tu disk. Khong throw neu chua co."""
        
        try:
            import faiss
            if not os.path.exists(self.index_path):
                print(f"[rag] Index chưa tồn tại: {self.index_path}")
                print("[rag] Chạy scripts/build_vectordb.py để build index.")
                return

            self._index = faiss.read_index(self.index_path)

            if os.path.exists(self.chunks_path):
                with open(self.chunks_path, "rb") as f:
                    self._chunks = pickle.load(f)

            self._load_embedder()
            print(
                f"[rag] FAISS index loaded - "
                f"{self._index.ntotal} vectors, {len(self._chunks)} chunks"
            )

        except ImportError:
            print("[rag] faiss-cpu chưa install -> RAG disabled.")
        except Exception as e:
            print(f"[rag] Load failed: {e} -> RAG disabled.")

    def _load_embedder(self):
        """Load embedding model."""
        
        try:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(self.embedding_model_name)
        except ImportError:
            print("[rag] sentence-transformers chưa install -> RAG disabled.")

    def is_ready(self) -> bool:
        return (
            self._index is not None
            and len(self._chunks) > 0
            and self._embedder is not None
        )

    def retrieve(self, query: str, k: int = 3) -> List[str]:
        """
        Retrieve top-k chunks liên quan đến query.

        Args:
            query: clinical question hoặc finding description
            k:     số chunks trả về

        Returns:
            List[str] - text chunks, empty list nếu RAG không sẵn sàng
        """
        if not self.is_ready():
            return []

        try:
            query_vec = self._embedder.encode(
                [query], convert_to_numpy=True, normalize_embeddings=True
            )
            distances, indices = self._index.search(query_vec, k)
            results = []
            for idx in indices[0]:
                if 0 <= idx < len(self._chunks):
                    results.append(self._chunks[idx])
            return results

        except Exception as e:
            print(f"[rag] Retrieve error: {e}")
            return []

    def retrieve_with_scores(self, query: str, k: int = 3) -> List[dict]:
        """
        Như retrieve() nhưng kèm score (L2 distance).
        Dùng cho debugging - orchestrator dùng retrieve() thường.
        """
        if not self.is_ready():
            return []

        try:
            query_vec = self._embedder.encode(
                [query], convert_to_numpy=True, normalize_embeddings=True
            )
            distances, indices = self._index.search(query_vec, k)
            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if 0 <= idx < len(self._chunks):
                    results.append({
                        "chunk": self._chunks[idx],
                        "distance": round(float(dist), 4),
                    })
            return results

        except Exception as e:
            print(f"[rag] Retrieve error: {e}")
            return []
