"""
Tests for scripts/generate_embedding_finetune_data.py: parsing, hard
negative selection, resume logic. Does not call a real LLM.
"""

import importlib.util
import json
import os
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

_MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "generate_embedding_finetune_data.py"
_spec = importlib.util.spec_from_file_location("generate_embedding_finetune_data", _MODULE_PATH)
gefd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gefd)


# _parse_questions

def test_parse_questions_valid_json_array():
    raw = '["What is the margin criterion for BI-RADS 4a?"]'
    result = gefd._parse_questions(raw, n_expected=1)
    assert result == ["What is the margin criterion for BI-RADS 4a?"]


def test_parse_questions_strips_markdown_fence():
    raw = '```json\n["Question one?", "Question two?"]\n```'
    result = gefd._parse_questions(raw, n_expected=2)
    assert result == ["Question one?", "Question two?"]


def test_parse_questions_truncates_to_n_expected():
    raw = '["Q1?", "Q2?", "Q3?"]'
    result = gefd._parse_questions(raw, n_expected=1)
    assert result == ["Q1?"]


def test_parse_questions_invalid_json_returns_empty():
    raw = "This is not JSON at all."
    result = gefd._parse_questions(raw, n_expected=1)
    assert result == []


def test_parse_questions_json_object_not_array_returns_empty():
    raw = '{"question": "not an array"}'
    result = gefd._parse_questions(raw, n_expected=1)
    assert result == []


def test_parse_questions_empty_strings_filtered():
    raw = '["", "  ", "Real question?"]'
    result = gefd._parse_questions(raw, n_expected=3)
    assert result == ["Real question?"]


# _select_hard_negatives

def _make_fixture():
    chunks = [
        "breast chunk A section1",
        "breast chunk B section1",
        "breast chunk C section2",
        "thyroid chunk D section1",
        "thyroid chunk E section2",
    ]
    metadata = [
        {"organ": "breast", "source_file": "breast.pdf", "section_heading": "s1"},
        {"organ": "breast", "source_file": "breast.pdf", "section_heading": "s1"},
        {"organ": "breast", "source_file": "breast.pdf", "section_heading": "s2"},
        {"organ": "thyroid", "source_file": "thyroid.pdf", "section_heading": "s1"},
        {"organ": "thyroid", "source_file": "thyroid.pdf", "section_heading": "s2"},
    ]
    by_organ = {}
    by_source = {}
    for i, meta in enumerate(metadata):
        by_organ.setdefault(meta["organ"], []).append(i)
        by_source.setdefault(meta["source_file"], []).append(i)
    return chunks, metadata, by_organ, by_source


def test_hard_negatives_include_other_organ():
    chunks, metadata, by_organ, by_source = _make_fixture()
    rng = random.Random(0)
    negatives = gefd._select_hard_negatives(
        idx=0, chunks=chunks, metadata=metadata, by_organ=by_organ, by_source=by_source,
        rng=rng, n_organ_negatives=2, n_section_negatives=1,
    )
    assert any(n.startswith("thyroid") for n in negatives)


def test_hard_negatives_include_same_source_different_section():
    chunks, metadata, by_organ, by_source = _make_fixture()
    rng = random.Random(0)
    negatives = gefd._select_hard_negatives(
        idx=0, chunks=chunks, metadata=metadata, by_organ=by_organ, by_source=by_source,
        rng=rng, n_organ_negatives=0, n_section_negatives=1,
    )
    assert negatives == ["breast chunk C section2"]


def test_hard_negatives_excludes_self_and_same_section():
    chunks, metadata, by_organ, by_source = _make_fixture()
    rng = random.Random(0)
    for _ in range(20):
        negatives = gefd._select_hard_negatives(
            idx=0, chunks=chunks, metadata=metadata, by_organ=by_organ, by_source=by_source,
            rng=rng, n_organ_negatives=0, n_section_negatives=1,
        )
        assert chunks[0] not in negatives
        assert "breast chunk B section1" not in negatives


def test_hard_negatives_empty_pool_returns_empty_list():
    chunks = ["only chunk"]
    metadata = [{"organ": "breast", "source_file": "only.pdf", "section_heading": "s1"}]
    by_organ = {"breast": [0]}
    by_source = {"only.pdf": [0]}
    rng = random.Random(0)
    negatives = gefd._select_hard_negatives(
        idx=0, chunks=chunks, metadata=metadata, by_organ=by_organ, by_source=by_source,
        rng=rng, n_organ_negatives=2, n_section_negatives=1,
    )
    assert negatives == []


# _load_vectordb

def test_load_vectordb_mismatched_lengths_raises(tmp_path):
    import pickle
    chunks_path = tmp_path / "chunks.pkl"
    metadata_path = tmp_path / "metadata.pkl"
    with open(chunks_path, "wb") as f:
        pickle.dump(["a", "b"], f)
    with open(metadata_path, "wb") as f:
        pickle.dump([{"organ": "breast"}], f)
    with pytest.raises(ValueError):
        gefd._load_vectordb(str(chunks_path), str(metadata_path))


def test_load_vectordb_matched_lengths_ok(tmp_path):
    import pickle
    chunks_path = tmp_path / "chunks.pkl"
    metadata_path = tmp_path / "metadata.pkl"
    with open(chunks_path, "wb") as f:
        pickle.dump(["a", "b"], f)
    with open(metadata_path, "wb") as f:
        pickle.dump([{"organ": "breast"}, {"organ": "thyroid"}], f)
    chunks, metadata = gefd._load_vectordb(str(chunks_path), str(metadata_path))
    assert chunks == ["a", "b"]
    assert len(metadata) == 2


# _load_done_indices / resume

def test_load_done_indices_missing_file_returns_empty(tmp_path):
    out_file = tmp_path / "does_not_exist.jsonl"
    assert gefd._load_done_indices(out_file) == set()


def test_load_done_indices_reads_chunk_idx(tmp_path):
    out_file = tmp_path / "out.jsonl"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(json.dumps({"chunk_idx": 3, "query": "q1"}) + "\n")
        f.write(json.dumps({"chunk_idx": 7, "query": "q2"}) + "\n")
    assert gefd._load_done_indices(out_file) == {3, 7}


def test_load_done_indices_skips_malformed_lines(tmp_path):
    out_file = tmp_path / "out.jsonl"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("not valid json\n")
        f.write(json.dumps({"chunk_idx": 1}) + "\n")
    assert gefd._load_done_indices(out_file) == {1}


# RateLimitedLLMClient

class _FlakyClient:
    def __init__(self, fail_times):
        self.fail_times = fail_times
        self.calls = 0

    def generate(self, *args, **kwargs):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise RuntimeError("simulated transient failure")
        return "ok"


def test_rate_limited_client_retries_then_succeeds():
    inner = _FlakyClient(fail_times=2)
    client = gefd.RateLimitedLLMClient(
        inner, max_calls_per_minute=0, max_retries=3, retry_base_delay=0.01,
    )
    result = client.generate("prompt")
    assert result == "ok"
    assert inner.calls == 3


def test_rate_limited_client_raises_after_max_retries():
    inner = _FlakyClient(fail_times=10)
    client = gefd.RateLimitedLLMClient(
        inner, max_calls_per_minute=0, max_retries=2, retry_base_delay=0.01,
    )
    with pytest.raises(RuntimeError):
        client.generate("prompt")
    assert inner.calls == 3


def test_rate_limited_client_getattr_passthrough():
    class _Inner:
        def generate(self, *a, **k):
            return "ok"
        some_attr = "hello"

    client = gefd.RateLimitedLLMClient(_Inner(), max_calls_per_minute=0)
    assert client.some_attr == "hello"
