"""
Tests for the parallel pipeline (vision + rag + cot), merge_node, consistency_guard, qa_agent_node.
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
        "image_id":      "test_123",
        "modality_hint": None,
        "organ_hint":    None,
        "routing":       {"modality": "ultrasound", "organ": "breast",
                          "module_key": "us_breast", "confidence": 0.9,
                          "hint_conflict": False, "hint_resolution_note": None},
        "model_output":  {"top_label": "benign", "confidence": 0.8,
                          "all_scores": {"benign": 0.8}, "mask_png_base64": "",
                          "original_size": [512, 512],
                          "bottleneck_enriched": {},
                          "gradcam_png_base64": "",
                          "gradcam_mask_overlap": {},
                          "texture_features": {},
                          "uncertainty": {},
                          "filtered_findings": []},
        "knowledge":     {"severity": "significant", "severity_level": 2,
                          "icd10_hint": "N63.0", "risk_category": "BI-RADS 3",
                          "description": "Benign", "confidence_calibration_note": None},
        "spatial":       {"location_quadrant": "upper-outer", "bbox": [10, 10, 50, 50],
                          "area_cm2": 1.0, "pixel_spacing_reliable": False,
                          "aspect_ratio": 1.1, "aspect_ratio_interpretation": "intermediate",
                          "circularity": 0.9,
                          "centroid": [30, 30], "width_px": 40, "height_px": 40,
                          "location_confidence": "high"},
        "cot_result":    None,
        "rag_chunks":    [],
        "rag_meta":      [],
        "consensus":     None,
        "icd10_agreement": None,
        "label_agreement": None,
        "hard_conflict": None,
        "visual_flags":  [],
        "risk_modifier": 0,
        "report":        None,
        "error":         None,
    }
    state.update(overrides)
    return state


# Mock RAG store

class MockRAGStore:
    """Mock RAG store - records the query passed in."""

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

def test_rag_node_query_uses_modality_organ():
    """
    In two_stage mode, rag_node runs before vision finishes and uses
    "{modality} {organ}" from the routing result as the initial broad query.
    The enriched query (with top_label + icd10) happens in consistency_guard.
    """
    store = MockRAGStore()
    rag_node = make_rag_node(store)

    state = _base_state()
    result = asyncio.run(rag_node(state))

    assert store.last_query == "ultrasound breast"
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
    assert store.last_query is None    # retrieve was not called


# CoT node -- regression test for the spatial=None bug causing AttributeError

class MockLLMForCoT:
    def generate(self, prompt, system=None):
        return ('{"cot_label": "benign", "severity": "incidental", "severity_level": 1, '
                '"icd10_hint": "N63.0", "risk_category": "low", "reasoning": "test"}')


def test_cot_node_handles_spatial_none_without_crashing():
    """
    Regression test: cot_reasoning_node runs in parallel with knowledge_node.
    If cot_node runs before knowledge_node has a chance to set state['spatial'],
    that value is still None (the initial value in OrchestratorState) -- NOT {}.
    state.get('spatial', {}) does not catch this case since the key 'spatial'
    still exists with value None. Before the fix, this caused an AttributeError
    inside _build_cot_prompt when calling spatial.get(...) on None, crashing the
    entire /analyze with a 500 Internal Server Error.
    """
    cot_node = make_cot_node(MockLLMForCoT())
    state = _base_state(
        spatial=None,      # simulates the exact case where knowledge_node hasn't run yet
        knowledge=None,
    )
    result = asyncio.run(cot_node(state))
    assert result["cot_result"]["severity"] == "incidental"


def test_cot_node_handles_routing_none_without_crashing():
    """Same as the test above but for field routing=None."""
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
    # Mapper: level 2, CoT: level 2 -> agreement
    state = _base_state(
        cot_result={"cot_label": "benign", "severity": "significant", "severity_level": 2,
                    "icd10_hint": "N63.0", "risk_category": "BI-RADS 3",
                    "reasoning": "test"},
    )
    result = asyncio.run(guard(state))
    assert result["consensus"] is True


def test_consistency_guard_consensus_false_when_levels_differ_much():
    store = MockRAGStore()
    guard = make_consistency_guard_node(store)
    # Mapper: level 2, CoT: level 4 -> disagreement (differ by 2)
    state = _base_state(
        cot_result={"cot_label": "malignant", "severity": "critical", "severity_level": 4,
                    "icd10_hint": "C50.9", "risk_category": "BI-RADS 5",
                    "reasoning": "test"},
    )
    result = asyncio.run(guard(state))
    assert result["consensus"] is False


def test_icd10_agreement_false_when_codes_differ_even_if_levels_match():
    """
    Mapper and CoT have the same severity_level but different icd10_hint
    (benign vs malignant) -> consensus is still True (severity only) but
    icd10_agreement must be False since this is an independent clinical question.
    """
    store = MockRAGStore()
    guard = make_consistency_guard_node(store)
    # Mapper: level 2, icd10 N63.0 (benign). CoT: level 2, icd10 C50.9 (malignant).
    state = _base_state(
        cot_result={"cot_label": "malignant", "severity": "significant", "severity_level": 2,
                    "icd10_hint": "C50.9", "risk_category": "BI-RADS 5",
                    "reasoning": "test"},
    )
    result = asyncio.run(guard(state))
    assert result["consensus"] is True
    assert result["icd10_agreement"] is False


def test_icd10_agreement_true_when_codes_match():
    store = MockRAGStore()
    guard = make_consistency_guard_node(store)
    state = _base_state(
        cot_result={"cot_label": "benign", "severity": "significant", "severity_level": 2,
                    "icd10_hint": "N63.0", "risk_category": "BI-RADS 3",
                    "reasoning": "test"},
    )
    result = asyncio.run(guard(state))
    assert result["icd10_agreement"] is True


class MockLLMForQA:
    def generate(self, prompt, system=None):
        return "TIER 2 -- test description\nTIER 3 -- test suggestion"


def test_qa_agent_node_surfaces_both_icd10_codes_when_disagreement():
    """
    When icd10_agreement is False, the final report must contain both ICD-10
    codes (not just keep the mapper's code) and tier_1_structured.icd10_hint
    must be the concatenation of both, exactly as required in TODO.md item 3.
    """
    qa_agent_node = make_qa_agent_node(MockLLMForQA(), MockRAGStore())
    state = _base_state(
        consensus=True,
        icd10_agreement=False,
        cot_result={"cot_label": "malignant", "severity": "significant", "severity_level": 2,
                    "icd10_hint": "C50.9", "risk_category": "BI-RADS 5",
                    "reasoning": "test"},
    )
    result = asyncio.run(qa_agent_node(state))
    report = result["report"]

    assert report["icd10_agreement"] is False
    assert "N63.0" in report["tier_1_structured"]["icd10_hint"]
    assert "C50.9" in report["tier_1_structured"]["icd10_hint"]
    assert report["mapper_result"]["icd10_hint"] == "N63.0"
    assert report["cot_result"]["icd10_hint"] == "C50.9"


def test_qa_agent_node_single_icd10_when_codes_agree():
    qa_agent_node = make_qa_agent_node(MockLLMForQA(), MockRAGStore())
    state = _base_state(
        consensus=True,
        icd10_agreement=True,
        cot_result={"cot_label": "benign", "severity": "significant", "severity_level": 2,
                    "icd10_hint": "N63.0", "risk_category": "BI-RADS 3",
                    "reasoning": "test"},
    )
    result = asyncio.run(qa_agent_node(state))
    report = result["report"]

    assert report["tier_1_structured"]["icd10_hint"] == "N63.0"


def test_consistency_guard_consensus_none_when_cot_undetermined():
    store = MockRAGStore()
    guard = make_consistency_guard_node(store)
    state = _base_state(
        cot_result={"cot_label": "unknown", "severity": "undetermined", "severity_level": 0,
                    "icd10_hint": "R93.8", "risk_category": "undetermined",
                    "reasoning": "parse error"},
    )
    result = asyncio.run(guard(state))
    assert result["consensus"] is None
    assert result["icd10_agreement"] is None


def test_consistency_guard_handles_knowledge_none_without_crashing():
    """
    Regression test: if the fan-in hasn't completed correctly and consistency_guard
    runs before knowledge_node has a chance to set state, state['knowledge'] is
    still None (the initial value). second_retrieve and consistency_guard must
    not crash in this case, just return consensus=None instead of raising AttributeError.
    """
    store = MockRAGStore()
    guard = make_consistency_guard_node(store)
    state = _base_state(
        knowledge=None,
        routing=None,
        model_output=None,
        cot_result={"cot_label": "benign", "severity": "incidental", "severity_level": 1,
                    "icd10_hint": "N63.0", "risk_category": "low", "reasoning": "x"},
    )
    result = asyncio.run(guard(state))
    # mapper_level defaults to 0, cot_level=1 -> differs by 1 -> consensus True
    assert result["consensus"] is True
    # knowledge=None so there is no icd10_hint to compare -> icd10_agreement None
    assert result["icd10_agreement"] is None


# Concurrency: vision and rag run in parallel

def test_vision_and_rag_nodes_run_concurrently():
    """
    Each branch simulates 0.3s latency. If run in parallel, total time is ~0.3s.
    If sequential, ~0.6s. Asserting total < 0.5s proves real parallelism.
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
        f"The two branches took {elapsed:.2f}s -- too long, looks sequential"
    )
    assert rag_result["rag_chunks"]


# AsyncSequentialFallback

def test_sequential_fallback_runs_both_rag_and_image_branches():
    """
    When langgraph is not installed, AsyncSequentialFallback must still run
    both the rag_node and the image/spatial/knowledge branch - no branch skipped.
    """
    rag_store = MockRAGStore(chunks=["guideline_1"])

    async def fake_route(state):
        state["routing"] = _base_state()["routing"]
        return state

    async def fake_vision(state):
        state["model_output"] = _base_state()["model_output"]
        return state

    async def fake_spatial(state):
        state["spatial"] = _base_state()["spatial"]
        return state

    async def fake_knowledge(state):
        state["knowledge"] = _base_state()["knowledge"]
        return state

    class MockLLM:
        def generate(self, prompt, system=None):
            return "TIER 2 findings\nTIER 3 suggestion"

    async def fake_birads(state):
        return {"birads_description": None}

    fb = AsyncSequentialFallback(
        image_nodes=[fake_route, fake_vision, fake_spatial, fake_knowledge],
        cot_node=make_rag_node(rag_store),      # stub the CoT branch with rag_node for simplicity
        rag_node=make_rag_node(rag_store),
        birads_node=fake_birads,
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


def test_sequential_fallback_cot_does_not_see_knowledge_before_reasoning():
    """
    Regression test for a real bug: cot_node must run on a state snapshot
    taken BEFORE knowledge_node executes, exactly like the real LangGraph
    edges (spatial -> {knowledge, cot_reasoning} in parallel). If cot_node
    sees state["knowledge"] already populated, it is no longer an
    independent assessment -- it could anchor on the mapper's answer,
    silently defeating the whole point of comparing two independent
    opinions (consensus / icd10_agreement).
    """
    rag_store = MockRAGStore()

    async def fake_route(state):
        state["routing"] = {"organ": "breast", "modality": "ultrasound"}
        return state

    async def fake_vision(state):
        state["model_output"] = {"top_label": "malignant", "confidence": 0.9}
        return state

    async def fake_spatial(state):
        state["spatial"] = {"area_cm2": 1.0}
        return state

    async def fake_knowledge(state):
        state["knowledge"] = {"severity": "critical", "icd10_hint": "C50.9"}
        return state

    captured = {}

    async def fake_cot(state):
        captured["saw_knowledge"] = state.get("knowledge") is not None
        captured["saw_spatial"] = state.get("spatial") is not None
        state["cot_result"] = {"severity": "undetermined"}
        return state

    class MockLLM:
        def generate(self, prompt, system=None):
            return "TIER 2 findings\nTIER 3 suggestion"

    async def fake_birads(state):
        return {"birads_description": None}

    fb = AsyncSequentialFallback(
        image_nodes=[fake_route, fake_vision, fake_spatial, fake_knowledge],
        cot_node=fake_cot,
        rag_node=make_rag_node(rag_store),
        birads_node=fake_birads,
        merge_node=make_merge_node(),
        consistency_guard_node=make_consistency_guard_node(rag_store),
        qa_agent_node=make_qa_agent_node(MockLLM(), rag_store),
    )

    state = _base_state(routing=None, model_output=None, knowledge=None, spatial=None)
    result = asyncio.run(fb.ainvoke(state))

    assert captured["saw_knowledge"] is False, (
        "cot_node saw state['knowledge'] before reasoning -- it must run "
        "independently of the mapper, exactly like the real LangGraph edges."
    )
    assert captured["saw_spatial"] is True, (
        "cot_node should still see spatial -- it needs spatial features to reason."
    )
    # knowledge must still end up in the final state once both branches finish
    assert result.get("knowledge") is not None


def test_sequential_fallback_propagates_error_and_skips_llm():
    """If the image branch fails, merge and qa_agent_node must not be called."""
    called = {"qa": False}

    async def fail_route(state):
        state["error"] = "route failed"
        return state

    class MockLLM:
        def generate(self, prompt, system=None):
            called["qa"] = True
            return ""

    async def fake_birads(state):
        return {"birads_description": None}

    rag_store = MockRAGStore()
    fb = AsyncSequentialFallback(
        image_nodes=[fail_route],
        cot_node=make_rag_node(rag_store),
        rag_node=make_rag_node(rag_store),
        birads_node=fake_birads,
        merge_node=make_merge_node(),
        consistency_guard_node=make_consistency_guard_node(rag_store),
        qa_agent_node=make_qa_agent_node(MockLLM(), rag_store),
    )

    state = _base_state(routing=None, model_output=None, knowledge=None, spatial=None)
    result = asyncio.run(fb.ainvoke(state))

    assert result["error"] == "route failed"
    assert called["qa"] is False
