"""
Test cho pipeline song song (vision + rag + cot), merge_node, consistency_guard, qa_agent_node.
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
    make_consistency_guard_node,
    make_qa_agent_node,
    make_cot_node,
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
                          "original_size": [512, 512], "bottleneck_features": {}},
        "knowledge":     {"severity": "significant", "severity_level": 2,
                          "icd10_hint": "N63.0", "risk_category": "BI-RADS 3",
                          "description": "Benign", "confidence_calibration_note": None},
        "spatial":       {"location_quadrant": "upper-outer", "bbox": [10, 10, 50, 50],
                          "area_cm2": 1.0, "aspect_ratio": 1.1, "circularity": 0.9,
                          "centroid": [30, 30], "width_px": 40, "height_px": 40,
                          "location_confidence": "high"},
        "cot_result":    None,
        "rag_chunks":    [],
        "rag_meta":      [],
        "consensus":     None,
        "report":        None,
        "error":         None,
    }
    state.update(overrides)
    return state


# Mock RAG store

class MockRAGStore:
    """RAG store gia lap - ghi lai query duoc truyen vao."""

    def __init__(self, chunks=None, latency=0.0):
        self.chunks = chunks or ["chunk A", "chunk B"]
        self.latency = latency
        self.last_query = None

    def is_ready(self):
        return True

    def retrieve(self, query: str, k: int = 3, organ_filter=None):
        self.last_query = query
        if self.latency:
            time.sleep(self.latency)
        return self.chunks[:k]

    def retrieve_with_meta(self, query: str, k: int = 3, organ_filter=None):
        self.last_query = query
        if self.latency:
            time.sleep(self.latency)
        return [
            {"chunk": c, "source_file": "test.pdf", "page_number": 1, "organ": "general"}
            for c in self.chunks[:k]
        ]

    def rerank(self, query, candidates, top_n=3):
        return candidates[:top_n]


# RAG node

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
        def retrieve(self, q, k=3, organ_filter=None):
            raise RuntimeError("Should not be called")
        def retrieve_with_meta(self, q, k=3, organ_filter=None):
            raise RuntimeError("Should not be called")

    rag_node = make_rag_node(NotReadyStore())
    state = _base_state()
    result = asyncio.run(rag_node(state))
    assert result["rag_chunks"] == []


def test_rag_node_skips_when_error_already_set():
    store = MockRAGStore()
    rag_node = make_rag_node(store)
    state = _base_state(error="upstream error")
    asyncio.run(rag_node(state))
    assert store.last_query is None    # retrieve khong bi goi


# CoT node -- regression test cho bug spatial=None gay AttributeError

class MockLLMForCoT:
    def generate(self, prompt, system=None):
        return ('{"severity": "incidental", "severity_level": 1, '
                '"icd10_hint": "N63.0", "risk_category": "low", "reasoning": "test"}')


def test_cot_node_handles_spatial_none_without_crashing():
    """
    Regression test: cot_reasoning_node chay song song voi knowledge_node.
    Neu cot_node toi truoc khi knowledge_node kip set state['spatial'], gia tri
    nay van con la None (gia tri khoi tao trong OrchestratorState) -- KHONG phai {}.
    state.get('spatial', {}) khong bat duoc truong hop nay vi key 'spatial' van
    ton tai voi gia tri None. Truoc khi sua, dieu nay gay AttributeError ben trong
    _build_cot_prompt khi goi spatial.get(...) tren None, lam crash toan bo /analyze
    voi 500 Internal Server Error.
    """
    cot_node = make_cot_node(MockLLMForCoT())
    state = _base_state(
        spatial=None,      # mo phong dung truong hop knowledge_node chua chay xong
        knowledge=None,
    )
    result = asyncio.run(cot_node(state))
    assert result["cot_result"]["severity"] == "incidental"


def test_cot_node_handles_routing_none_without_crashing():
    """Tuong tu test tren nhung cho field routing=None."""
    cot_node = make_cot_node(MockLLMForCoT())
    state = _base_state(routing=None)
    result = asyncio.run(cot_node(state))
    assert result["cot_result"]["severity"] == "incidental"


def test_cot_node_preserves_error_when_already_set():
    cot_node = make_cot_node(MockLLMForCoT())
    state = _base_state(error="vision failed")
    result = asyncio.run(cot_node(state))
    assert result == {"cot_result": None}


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


# Consistency guard

def test_consistency_guard_consensus_true_when_levels_match():
    store = MockRAGStore()
    guard = make_consistency_guard_node(store)
    # Mapper: level 2, CoT: level 2 -> dong thuan
    state = _base_state(
        cot_result={"severity": "significant", "severity_level": 2,
                    "icd10_hint": "N63.0", "risk_category": "BI-RADS 3",
                    "reasoning": "test"},
    )
    result = asyncio.run(guard(state))
    assert result["consensus"] is True


def test_consistency_guard_consensus_false_when_levels_differ_much():
    store = MockRAGStore()
    guard = make_consistency_guard_node(store)
    # Mapper: level 2, CoT: level 4 -> bat dong (chenh 2)
    state = _base_state(
        cot_result={"severity": "critical", "severity_level": 4,
                    "icd10_hint": "C50.9", "risk_category": "BI-RADS 5",
                    "reasoning": "test"},
    )
    result = asyncio.run(guard(state))
    assert result["consensus"] is False


def test_consistency_guard_consensus_none_when_cot_undetermined():
    store = MockRAGStore()
    guard = make_consistency_guard_node(store)
    state = _base_state(
        cot_result={"severity": "undetermined", "severity_level": 0,
                    "icd10_hint": "R93.8", "risk_category": "undetermined",
                    "reasoning": "parse error"},
    )
    result = asyncio.run(guard(state))
    assert result["consensus"] is None


def test_consistency_guard_handles_knowledge_none_without_crashing():
    """
    Regression test: neu fan-in chua hoan tat dung va consistency_guard chay
    truoc khi knowledge_node kip set state, state['knowledge'] van la None
    (gia tri khoi tao). second_retrieve va consistency_guard phai khong crash
    trong truong hop nay, chi tra ve consensus=None thay vi raise AttributeError.
    """
    store = MockRAGStore()
    guard = make_consistency_guard_node(store)
    state = _base_state(
        knowledge=None,
        routing=None,
        model_output=None,
        cot_result={"severity": "incidental", "severity_level": 1,
                    "icd10_hint": "N63.0", "risk_category": "low", "reasoning": "x"},
    )
    result = asyncio.run(guard(state))
    # mapper_level mac dinh 0, cot_level=1 -> chenh lech 1 -> consensus True
    assert result["consensus"] is True


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


# AsyncSequentialFallback

def test_sequential_fallback_runs_both_rag_and_image_branches():
    """
    Khi langgraph khong install, AsyncSequentialFallback van phai chay
    ca rag_node lan image/knowledge branch - khong bo nhanh nao.
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
        cot_node=make_rag_node(rag_store),      # stub nhanh CoT bang rag_node cho don gian
        rag_node=make_rag_node(rag_store),
        merge_node=make_merge_node(),
        consistency_guard_node=make_consistency_guard_node(rag_store),
        qa_agent_node=make_qa_agent_node(MockLLM(), rag_store),
    )

    state = _base_state(
        routing=None, model_output=None, knowledge=None, spatial=None
    )
    result = asyncio.run(fb.ainvoke(state))

    assert result.get("error") is None
    assert result.get("report") is not None


def test_sequential_fallback_propagates_error_and_skips_llm():
    """Neu nhanh image loi, merge va qa_agent_node khong duoc goi."""
    called = {"qa": False}

    async def fail_route(state):
        state["error"] = "route failed"
        return state

    class MockLLM:
        def generate(self, prompt, system=None):
            called["qa"] = True
            return ""

    rag_store = MockRAGStore()
    fb = AsyncSequentialFallback(
        image_nodes=[fail_route],
        cot_node=make_rag_node(rag_store),
        rag_node=make_rag_node(rag_store),
        merge_node=make_merge_node(),
        consistency_guard_node=make_consistency_guard_node(rag_store),
        qa_agent_node=make_qa_agent_node(MockLLM(), rag_store),
    )

    state = _base_state(routing=None, model_output=None, knowledge=None, spatial=None)
    result = asyncio.run(fb.ainvoke(state))

    assert result["error"] == "route failed"
    assert called["qa"] is False
