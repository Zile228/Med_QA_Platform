"""
services/orchestrator/icd10_lookup.py

Exact-key lookup for ICD-10-CM code descriptions, loaded from a JSON file
built offline by scripts/build_icd10_lookup.py. This is intentionally
separate from FAISSStore: ICD-10 is a structured reference where one code
maps to exactly one fixed description, so an exact key lookup is faster
and 100% accurate, unlike semantic search over an embedding index which
depends on unstable cosine similarity for short, context-free lines.

Public API:
    ICD10Lookup(json_path) -> ICD10Lookup
    ICD10Lookup.describe(code) -> str | None
    ICD10Lookup.is_ready() -> bool
"""

import os
import json
from typing import Optional


class ICD10Lookup:
    """
    Loads a code -> description JSON mapping from disk and exposes a
    lookup by exact ICD-10 code.

    If the JSON file does not exist yet -> is_ready() = False -> describe()
    always returns None (no crash, callers can treat it as "unavailable").
    """

    def __init__(self, json_path: str = None):
        self.json_path = json_path or os.getenv(
            "ICD10_LOOKUP_PATH",
            "services/orchestrator/icd10_lookup.json",
        )
        self._lookup: dict = {}
        self._try_load()

    def _try_load(self):
        """Loads the code -> description JSON from disk. Does not throw if missing."""
        if not os.path.exists(self.json_path):
            print(f"[icd10_lookup] Lookup file not found: {self.json_path}")
            print("[icd10_lookup] Run scripts/build_icd10_lookup.py to build it.")
            return
        try:
            with open(self.json_path, "r", encoding="utf-8") as f:
                self._lookup = json.load(f)
            print(f"[icd10_lookup] Loaded {len(self._lookup)} ICD-10 codes")
        except Exception as e:
            print(f"[icd10_lookup] Load failed: {e} -> lookup disabled.")
            self._lookup = {}

    def is_ready(self) -> bool:
        return len(self._lookup) > 0

    def describe(self, code: Optional[str]) -> Optional[str]:
        """
        Returns the description for an exact ICD-10 code, or None if the
        code is missing, blank, or not found in the lookup.
        """
        if not code:
            return None
        return self._lookup.get(code.strip().upper())
