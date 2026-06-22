"""
Tests for section 3: resolve_with_hint() and the router /route endpoint with a hint.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from services.router.model import resolve_with_hint


# Default weight value -- the tests must assert specific numbers
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
    probs nearly even, hint=thyroid.
    By hand: score(thyroid) = 0.7*0.45 + 0.3*1.0 = 0.615
             score(breast)  = 0.7*0.55 + 0.3*0.0 = 0.385
    -> thyroid wins.
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
    Router is very confident (0.95), hint=thyroid.
    score(breast)  = 0.7*0.95 = 0.665
    score(thyroid) = 0.7*0.05 + 0.3*1.0 = 0.335
    -> breast wins, but hint_conflict is still True.
    """
    probs = {"us_breast": 0.95, "us_thyroid": 0.05}
    result = resolve_with_hint(probs, hint_module_key="us_thyroid", router_weight=_W)

    assert result["final_module_key"] == "us_breast"
    assert result["hint_conflict"] is True     # there is a conflict even though the router wins

    ws = result["weighted_scores"]
    assert abs(ws["us_breast"]  - (_W * 0.95)) < 1e-3
    assert abs(ws["us_thyroid"] - (_W * 0.05 + (1 - _W) * 1.0)) < 1e-3


def test_resolve_with_hint_invalid_not_in_router_classes():
    """
    hint not present in router_probs -- still runs fine, no crash.
    A class absent from probs will also be absent from weighted_scores.
    """
    probs = {"us_breast": 0.7, "us_thyroid": 0.3}
    # "us_xray" is not in probs -> its score would be 0.3*1.0 = 0.3
    # but it is absent from weighted_scores since the loop iterates over probs
    result = resolve_with_hint(probs, hint_module_key="us_xray", router_weight=_W)
    # Both existing classes are still scored, xray isn't in the set -> breast wins
    assert result["final_module_key"] in {"us_breast", "us_thyroid"}
    assert result["hint_conflict"] is True    # "us_xray" != router top "us_breast"


def test_final_decision_source_user_hint_when_hint_wins():
    probs = {"us_breast": 0.55, "us_thyroid": 0.45}
    result = resolve_with_hint(probs, hint_module_key="us_thyroid", router_weight=_W)
    assert result["final_decision_source"] == "user_hint"


def test_final_decision_source_weighted_when_router_wins_over_hint():
    probs = {"us_breast": 0.95, "us_thyroid": 0.05}
    result = resolve_with_hint(probs, hint_module_key="us_thyroid", router_weight=_W)
    # Router wins -> source is "weighted" (hint exists but isn't strong enough)
    assert result["final_decision_source"] == "weighted"
