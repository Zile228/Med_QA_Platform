"""
Test cho section 4: pipeline song song (vision + rag), merge_node, qa_agent_node.
"""

import asyncio
import os
import sys
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from services.orchestrator.graph import (
    make_rag_node,
    make_merge_node,
    make_qa_agent_node,
    AsyncSequentialFallback,
    OrchestratorState,
)


# Helpers

def _base_state(**overrides) -> dict:
    state = {
        "image_bytes":   b"fake",
        "question":      "What are the findings?",
        "image_id":      "test_123",
        "modality_hint": None,
        "organ_hint":    None,
        "routing":       {"modality": "ultrasound", "organ": "breast",
                          "module_key": "us_breast", "confidence": 0.9,
                          "hint_conflict": False, "hint_resolution_note": None},
        "model_output":  {"top_label": "benign", "confidence": 0.8,
                          "all_scores": {"benign": 0.8}, "mask_png_base64": "",
                          "original_size": [512, 512]},
        "knowledge":     {"severity": "significant", "severity_level": 2,
                          "icd10_hint": "N63.0", "risk_category": "BI-RADS 3",
                          "description": "Benign", "confidence_calibration_note": None},
        "spatial":       {"location_quadrant": "upper-outer", "bbox": [10, 10, 50, 50],
                          "area_cm2": 1.0, "aspect_ratio": 1.1, "circularity": 0.9,
                          "centroid": [30, 30], "width_px": 40, "height_px": 40,
                          "location_confidence": "high"},
        "rag_chunks":    [],
        "report":        None,
        "error":         None,
    }
    state.update(overrides)
    return state


# RAG node

class MockRAGStore:
    """RAG store gia lap -- ghi lai query duoc truyen vao."""
    def __init__(self, chunks=None, latency=0.0):
        self.chunks = chunks or ["chunk A", "chunk B"]
        self.latency = latency
        self.last_query = None

    def is_ready(self):
        return True

    def retrieve(self, query: str, k: int = 3):
        self.last_query = query
        if self.latency:
            time.sleep(self.latency)
        return self.chunks[:k]


def test_rag_node_query_uses_user_question_not_classification_label():
    """
    rag_node phai dung question goc cua user lam query, KHONG dung
    organ/top_label (vi luc do vision chua chay xong).
    """
    store = MockRAGStore()
    rag_node = make_rag_node(store)

    state = _base_state(question="Is this lesion malignant?")
    result = asyncio.run(rag_node(state))

    assert store.last_query == "Is this lesion malignant?"
    assert "benign" not in (store.last_query or "")
    assert "us_breast" not in (store.last_query or "")
    assert result["rag_chunks"] == ["chunk A", "chunk B"]


def test_rag_node_returns_empty_when_store_not_ready():
    class NotReadyStore:
        def is_ready(self):
            return False
        def retrieve(self, q, k=3):
            raise RuntimeError("Should not be called")

    rag_node = make_rag_node(NotReadyStore())
    state = _base_state()
    result = asyncio.run(rag_node(state))
    assert result["rag_chunks"] == []


def test_rag_node_skips_when_error_already_set():
    store = MockRAGStore()
    rag_node = make_rag_node(store)
    state = _base_state(error="upstream error")
    result = asyncio.run(rag_node(state))
    assert store.last_query is None    # retrieve khong bi goi


# Merge node

def test_merge_node_passes_when_both_branches_ok():
    merge_node = make_merge_node()
    state = _base_state(rag_chunks=["doc1"])
    result = asyncio.run(merge_node(state))
    assert result["error"] is None


def test_merge_node_short_circuits_on_branch_error():
    merge_node = make_merge_node()
    state = _base_state(error="vision failed", knowledge=None)
    result = asyncio.run(merge_node(state))
    assert result["error"] is not None


def test_merge_node_sets_error_when_knowledge_missing():
    merge_node = make_merge_node()
    state = _base_state(knowledge=None)
    result = asyncio.run(merge_node(state))
    assert result["error"] is not None


# Concurrency: vision va rag chay song song

def test_vision_and_rag_nodes_run_concurrently():
    """
    Moi nhanh gia lap 0.3s latency. Neu song song thi tong thoi gian ~ 0.3s.
    Neu tuan tu thi ~ 0.6s. Assert tong < 0.5s chung minh song song that su.
    """
    LATENCY = 0.3
    store = MockRAGStore(latency=LATENCY)

    async def slow_image_branch(state):
        await asyncio.sleep(LATENCY)
        state["knowledge"] = _base_state()["knowledge"]
        state["spatial"]   = _base_state()["spatial"]
        return state

    async def run():
        state = _base_state()
        t0 = time.perf_counter()
        image_result, rag_result = await asyncio.gather(
            slow_image_branch(state.copy()),
            make_rag_node(store)(state.copy()),
        )
        elapsed = time.perf_counter() - t0
        return elapsed, image_result, rag_result

    elapsed, _, rag_result = asyncio.run(run())
    assert elapsed < LATENCY * 1.8, (
        f"Hai nhanh mat {elapsed:.2f}s -- qua lau, co ve chay tuan tu"
    )
    assert rag_result["rag_chunks"]


def test_merge_node_waits_for_both_branches():
    """
    Nhanh nhanh xong truoc nhung merge_node van phai doi nhanh cham.
    Kiem tra qua sequential fallback: ca hai ket qua deu co trong state cuoi.
    """
    fast_chunks = ["fast_doc"]
    slow_chunks = ["slow_doc"]

    class SlowStore:
        def is_ready(self):
            return True
        def retrieve(self, q, k=3):
            time.sleep(0.15)
            return slow_chunks

    store = SlowStore()

    async def run():
        state = _base_state(rag_chunks=[])
        # Gia lap: rag chay cham, sau do merge
        rag_result = await make_rag_node(store)(state.copy())
        state["rag_chunks"] = rag_result["rag_chunks"]
        merged = await make_merge_node()(state)
        return merged

    result = asyncio.run(run())
    assert result["rag_chunks"] == slow_chunks
    assert result["error"] is None


# AsyncSequentialFallback

def test_sequential_fallback_runs_both_rag_and_image_branches():
    """
    Khi langgraph khong install, AsyncSequentialFallback van phai chay
    ca rag_node lan image/knowledge branch -- khong bo nhanh nao.
    """
    rag_store = MockRAGStore(chunks=["guideline_1"])

    async def fake_route(state):
        state["routing"] = _base_state()["routing"]
        return state

    async def fake_vision(state):
        state["model_output"] = _base_state()["model_output"]
        return state

    async def fake_knowledge(state):
        state["knowledge"] = _base_state()["knowledge"]
        state["spatial"]   = _base_state()["spatial"]
        return state

    class MockLLM:
        def generate(self, prompt, system=None):
            return "TIER 2 findings\nTIER 3 suggestion"

    fb = AsyncSequentialFallback(
        image_nodes=[fake_route, fake_vision, fake_knowledge],
        rag_node=make_rag_node(rag_store),
        merge_node=make_merge_node(),
        qa_agent_node=make_qa_agent_node(MockLLM(), rag_store),
    )

    state = _base_state(
        routing=None, model_output=None, knowledge=None, spatial=None
    )
    result = asyncio.run(fb.ainvoke(state))

    assert result.get("error") is None
    assert result.get("rag_chunks") == ["guideline_1"]
    assert result.get("report") is not None


def test_sequential_fallback_propagates_error_and_skips_llm():
    """
    Neu nhanh image loi, merge va qa_agent_node khong duoc goi.
    """
    called = {"qa": False}

    async def fail_route(state):
        state["error"] = "route failed"
        return state

    async def should_not_run(state):
        called["qa"] = True
        return state

    class MockLLM:
        def generate(self, prompt, system=None):
            called["qa"] = True
            return ""

    fb = AsyncSequentialFallback(
        image_nodes=[fail_route],
        rag_node=make_rag_node(MockRAGStore()),
        merge_node=make_merge_node(),
        qa_agent_node=make_qa_agent_node(MockLLM(), MockRAGStore()),
    )

    state = _base_state(routing=None, model_output=None, knowledge=None, spatial=None)
    result = asyncio.run(fb.ainvoke(state))

    assert result["error"] == "route failed"
    assert called["qa"] is False
