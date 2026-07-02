"""
Tests for scripts/split_embedding_finetune_data.py: source_file-based
train/val/test split, with no chunk leaking across splits.
"""

import importlib.util
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

_MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "split_embedding_finetune_data.py"
_spec = importlib.util.spec_from_file_location("split_embedding_finetune_data", _MODULE_PATH)
sefd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sefd)


def _make_records():
    # breast: 3 source files with distinct sizes; thyroid: 2 source files.
    records = []
    for i in range(50):
        records.append({"organ": "breast", "source_file": "breast_big.pdf", "query": f"q{i}", "chunk_idx": i})
    for i in range(10):
        records.append({"organ": "breast", "source_file": "breast_small.pdf", "query": f"q{i}", "chunk_idx": 1000 + i})
    for i in range(5):
        records.append({"organ": "breast", "source_file": "breast_tiny.pdf", "query": f"q{i}", "chunk_idx": 2000 + i})
    for i in range(30):
        records.append({"organ": "thyroid", "source_file": "thyroid_big.pdf", "query": f"q{i}", "chunk_idx": 3000 + i})
    for i in range(8):
        records.append({"organ": "thyroid", "source_file": "thyroid_small.pdf", "query": f"q{i}", "chunk_idx": 4000 + i})
    return records


def test_split_no_source_file_appears_in_two_splits():
    records = _make_records()
    train, val, test, split_map = sefd.split_records(records, val_frac_per_organ=1, test_frac_per_organ=1)

    train_sources = {r["source_file"] for r in train}
    val_sources = {r["source_file"] for r in val}
    test_sources = {r["source_file"] for r in test}

    assert not (train_sources & val_sources)
    assert not (train_sources & test_sources)
    assert not (val_sources & test_sources)


def test_split_smallest_files_go_to_test_then_val():
    records = _make_records()
    train, val, test, split_map = sefd.split_records(records, val_frac_per_organ=1, test_frac_per_organ=1)

    assert split_map["breast_tiny.pdf"] == "test"
    assert split_map["breast_small.pdf"] == "val"
    assert split_map["breast_big.pdf"] == "train"


def test_split_both_organs_present_in_test_when_possible():
    records = _make_records()
    train, val, test, split_map = sefd.split_records(records, val_frac_per_organ=1, test_frac_per_organ=1)

    test_organs = {r["organ"] for r in test}
    assert "breast" in test_organs
    assert "thyroid" in test_organs


def test_split_train_never_empty_for_an_organ_with_few_sources():
    records = _make_records()
    train, val, test, split_map = sefd.split_records(records, val_frac_per_organ=1, test_frac_per_organ=1)

    train_organs = {r["organ"] for r in train}
    assert "breast" in train_organs
    assert "thyroid" in train_organs


def test_split_organ_with_two_sources_reduces_val_not_train():
    # thyroid only has 2 sources: with val=1,test=1 requested, train would
    # be empty unless the split reduces one of them.
    records = _make_records()
    train, val, test, split_map = sefd.split_records(records, val_frac_per_organ=1, test_frac_per_organ=1)

    thyroid_splits = {s: split_map[s] for s in split_map if s.startswith("thyroid")}
    assert "train" in thyroid_splits.values()


def test_split_all_records_preserved():
    records = _make_records()
    train, val, test, split_map = sefd.split_records(records, val_frac_per_organ=1, test_frac_per_organ=1)
    assert len(train) + len(val) + len(test) == len(records)


def test_split_deterministic_across_runs():
    records = _make_records()
    result_a = sefd.split_records(records, val_frac_per_organ=1, test_frac_per_organ=1)
    result_b = sefd.split_records(records, val_frac_per_organ=1, test_frac_per_organ=1)
    assert result_a[3] == result_b[3]


def test_load_records_roundtrip(tmp_path):
    import json
    in_file = tmp_path / "data.jsonl"
    records = [{"query": "q1", "organ": "breast"}, {"query": "q2", "organ": "thyroid"}]
    with open(in_file, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    loaded = sefd._load_records(in_file)
    assert loaded == records


def test_load_records_skips_blank_lines(tmp_path):
    in_file = tmp_path / "data.jsonl"
    in_file.write_text('{"query": "q1"}\n\n{"query": "q2"}\n', encoding="utf-8")
    loaded = sefd._load_records(in_file)
    assert len(loaded) == 2
