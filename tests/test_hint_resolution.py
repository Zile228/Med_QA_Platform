"""
Test cho section 3: resolve_with_hint() va router /route endpoint voi hint.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from services.router.model import resolve_with_hint


# Gia tri trong so mac dinh -- test phai assert con so cu the
_W = 0.7


def test_resolve_with_hint_no_hint_returns_router_argmax():
    probs = {"us_breast": 0.8, "us_thyroid": 0.2}
    result = resolve_with_hint(probs, hint_module_key=None)
    assert result["final_module_key"] == "us_breast"
    assert result["hint_conflict"] is False
    assert result["final_decision_source"] == "router"
    assert result["hint_resolution_note"] is None


def test_resolve_with_hint_matches_router_no_conflict():
    probs = {"us_breast": 0.9, "us_thyroid": 0.1}
    result = resolve_with_hint(probs, hint_module_key="us_breast")
    assert result["hint_conflict"] is False
    assert result["final_module_key"] == "us_breast"


def test_resolve_with_hint_conflicts_low_router_confidence_hint_wins():
    """
    probs gan deu, hint=thyroid.
    Tinh tay: score(thyroid) = 0.7*0.45 + 0.3*1.0 = 0.615
              score(breast)  = 0.7*0.55 + 0.3*0.0 = 0.385
    -> thyroid thang.
    """
    probs = {"us_breast": 0.55, "us_thyroid": 0.45}
    result = resolve_with_hint(probs, hint_module_key="us_thyroid", router_weight=_W)

    assert result["final_module_key"] == "us_thyroid"
    assert result["hint_conflict"] is True

    ws = result["weighted_scores"]
    assert abs(ws["us_thyroid"] - (_W * 0.45 + (1 - _W) * 1.0)) < 1e-3
    assert abs(ws["us_breast"]  - (_W * 0.55 + (1 - _W) * 0.0)) < 1e-3


def test_resolve_with_hint_conflicts_high_router_confidence_router_wins():
    """
    Router rat chac (0.95), hint=thyroid.
    score(breast)  = 0.7*0.95 = 0.665
    score(thyroid) = 0.7*0.05 + 0.3*1.0 = 0.335
    -> breast thang, nhung hint_conflict van la True.
    """
    probs = {"us_breast": 0.95, "us_thyroid": 0.05}
    result = resolve_with_hint(probs, hint_module_key="us_thyroid", router_weight=_W)

    assert result["final_module_key"] == "us_breast"
    assert result["hint_conflict"] is True     # co conflict du router thang

    ws = result["weighted_scores"]
    assert abs(ws["us_breast"]  - (_W * 0.95)) < 1e-3
    assert abs(ws["us_thyroid"] - (_W * 0.05 + (1 - _W) * 1.0)) < 1e-3


def test_resolve_with_hint_invalid_not_in_router_classes():
    """
    hint khong nam trong router_probs -- van chay duoc, khong crash.
    Class khong co trong probs se khong co trong weighted_scores.
    """
    probs = {"us_breast": 0.7, "us_thyroid": 0.3}
    # "us_xray" khong co trong probs -> score cua no la 0.3*1.0 = 0.3
    # nhung khong co trong weighted_scores vi loop qua probs
    result = resolve_with_hint(probs, hint_module_key="us_xray", router_weight=_W)
    # Hai class cu van duoc score, xray khong co trong tap -> breast thang
    assert result["final_module_key"] in {"us_breast", "us_thyroid"}
    assert result["hint_conflict"] is True    # "us_xray" != router top "us_breast"


def test_final_decision_source_user_hint_when_hint_wins():
    probs = {"us_breast": 0.55, "us_thyroid": 0.45}
    result = resolve_with_hint(probs, hint_module_key="us_thyroid", router_weight=_W)
    assert result["final_decision_source"] == "user_hint"


def test_final_decision_source_weighted_when_router_wins_over_hint():
    probs = {"us_breast": 0.95, "us_thyroid": 0.05}
    result = resolve_with_hint(probs, hint_module_key="us_thyroid", router_weight=_W)
    # Router thang -> source la "weighted" (hint co nhung khong du manh)
    assert result["final_decision_source"] == "weighted"
