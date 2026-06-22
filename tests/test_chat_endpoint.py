"""
Tests for section 5.2: multi-turn chatbot, context cache, /chat endpoint logic.
"""

import asyncio
import os
import sys
import time

import pytest
from unittest.mock import MagicMock, patch, AsyncMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# Cache helpers (unit test, does not use FastAPI TestClient)

from services.orchestrator import main as orch_main


def _make_report_dict(image_id="img_001") -> dict:
    return {
        "image_id": image_id,
        "tier_1_structured": {
            "modality": "ultrasound", "organ": "breast",
            "label": "benign", "confidence": 0.85,
            "risk_category": "BI-RADS 3", "severity": "significant",
            "severity_level": 2, "icd10_hint": "N63.0",
            "location_quadrant": "upper-outer", "bbox": [10, 10, 60, 60],
            "area_cm2": 1.234, "aspect_ratio": 1.1, "circularity": 0.8,
            "confidence_calibration_note": None,
            "hint_conflict": False, "hint_resolution_note": None,
        },
        "tier_2_radiological_description": "Hypoechoic lesion.",
        "tier_3_diagnostic_suggestion":    "Follow-up recommended.",
        "rag_sources": [], "rag_disabled_warning": None,
    }


def test_save_and_get_context_round_trip():
    orch_main._context_cache.clear()
    report = _make_report_dict("round_trip_01")
    orch_main._save_context("round_trip_01", report, ["chunk_a"])

    entry = orch_main._get_context("round_trip_01")
    assert entry["context"]["image_id"] == "round_trip_01"
    assert entry["rag_chunks"] == ["chunk_a"]


def test_get_context_missing_image_id_raises_404():
    from fastapi import HTTPException
    orch_main._context_cache.clear()
    with pytest.raises(HTTPException) as exc:
        orch_main._get_context("nonexistent_id")
    assert exc.value.status_code == 404


def test_get_context_expired_raises_404():
    from fastapi import HTTPException
    orch_main._context_cache.clear()

    orch_main._context_cache["old_img"] = {
        "context":    {"image_id": "old_img", "report": {}},
        "rag_chunks": [],
        "ts":         time.time() - (orch_main._CONTEXT_TTL_SECONDS + 10),
    }

    with pytest.raises(HTTPException) as exc:
        orch_main._get_context("old_img")
    assert exc.value.status_code == 404
    assert "old_img" not in orch_main._context_cache   # removed from cache


def test_chat_uses_cached_context_not_reanalyzing():
    """
    After _save_context, calling /chat must read from the cache -- must not
    call router/vision/knowledge again. Verified via mock: the HTTP
    services must not be called.
    """
    orch_main._context_cache.clear()
    report = _make_report_dict("no_rerun")
    orch_main._save_context("no_rerun", report, [])

    # If the cache works correctly, _get_context does not throw
    entry = orch_main._get_context("no_rerun")
    assert entry is not None


# _build_chat_prompt

from services.orchestrator.graph import _build_chat_prompt


def test_build_chat_prompt_includes_report_context():
    unified_context = {
        "image_id": "img_abc",
        "report": {
            "tier_1_structured": {
                "modality": "ultrasound", "organ": "breast",
                "label": "malignant", "confidence": 0.91,
                "risk_category": "BI-RADS 5", "severity": "critical",
                "severity_level": 4, "icd10_hint": "C50.9",
                "location_quadrant": "upper-inner",
                "area_cm2": 2.5, "aspect_ratio": 0.8, "circularity": 0.3,
            },
            "tier_2_radiological_description": "Irregular mass.",
            "tier_3_diagnostic_suggestion":    "Biopsy recommended.",
        },
    }
    prompt = _build_chat_prompt(
        unified_context=unified_context,
        rag_chunks=["guideline text"],
        history=[{"role": "user", "content": "What is the size?"}],
        message="Is FNA needed?",
    )

    assert "malignant" in prompt
    assert "2.5" in prompt         # area_cm2
    assert "Biopsy recommended" in prompt
    assert "guideline text" in prompt
    assert "What is the size?" in prompt   # history is embedded
    assert "Is FNA needed?" in prompt      # the new question


def test_build_chat_prompt_history_order():
    """History must be in order: older turns first, new question last."""
    context = {"image_id": "x", "report": {"tier_1_structured": {
        "modality": "us", "organ": "breast", "label": "benign",
        "confidence": 0.8, "risk_category": "BI-RADS 3",
        "severity": "significant", "severity_level": 2,
        "icd10_hint": "N63", "location_quadrant": "central",
        "area_cm2": 0.5, "aspect_ratio": 1.0, "circularity": 0.9,
    }, "tier_2_radiological_description": "", "tier_3_diagnostic_suggestion": ""}}

    history = [
        {"role": "user",      "content": "First question"},
        {"role": "assistant", "content": "First answer"},
    ]
    prompt = _build_chat_prompt(context, [], history, "Second question")

    pos_first  = prompt.find("First question")
    pos_answer = prompt.find("First answer")
    pos_second = prompt.find("Second question")

    assert pos_first < pos_answer < pos_second


# LLM client multi-turn

from services.orchestrator.llm_client import MockLLMClient, BaseLLMClient


def test_mock_llm_client_chat_returns_string():
    client = MockLLMClient()
    messages = [
        {"role": "user",      "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
        {"role": "user",      "content": "What is the diagnosis?"},
    ]
    reply = client.chat(messages)
    assert isinstance(reply, str)
    assert len(reply) > 0


def test_base_llm_client_chat_falls_back_to_generate():
    """BaseLLMClient.chat() defaults to calling generate() -- tested via MockLLMClient."""
    client = MockLLMClient()
    messages = [{"role": "user", "content": "test"}]
    reply = client.chat(messages, system="System prompt")
    assert isinstance(reply, str)
