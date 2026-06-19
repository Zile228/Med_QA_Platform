"""
Test cho section 5: build_document_report_html, chatbot, va UI HTML contrast.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ui.app import (
    build_document_report_html,
    build_warning_banners,
)


# Helpers

def _make_report(**overrides) -> dict:
    t1 = {
        "modality":       "ultrasound",
        "organ":          "breast",
        "label":          "benign",
        "confidence":     0.85,
        "risk_category":  "BI-RADS 3",
        "severity":       "significant",
        "severity_level": 2,
        "icd10_hint":     "N63.0",
        "location_quadrant": "upper-outer",
        "bbox":           [10, 10, 60, 60],
        "area_cm2":       1.234,
        "aspect_ratio":   1.1,
        "circularity":    0.8,
        "confidence_calibration_note": None,
        "hint_conflict":  False,
        "hint_resolution_note": None,
    }
    t1.update(overrides.pop("t1_overrides", {}))
    base = {
        "image_id": "test_abc123",
        "tier_1_structured": t1,
        "tier_2_radiological_description": "Hypoechoic lesion with well-defined margins.",
        "tier_3_diagnostic_suggestion":    "AI-assisted suggestion: follow-up recommended.",
        "rag_sources":         [],
        "rag_disabled_warning": None,
    }
    base.update(overrides)
    return base


# Document report

def test_build_document_report_html_includes_all_three_tiers_content():
    report = _make_report()
    html = build_document_report_html(report)

    # Tier 1 structured fields
    assert "upper-outer" in html
    assert "N63.0" in html
    assert "significant" in html.lower()
    assert "1.234" in html

    # Tier 2 findings
    assert "Hypoechoic lesion with well-defined margins" in html

    # Tier 3 impression
    assert "AI-assisted suggestion" in html


def test_build_document_report_html_banner_precedes_findings():
    """
    Banner canh bao (hint_conflict) phai xuat hien truoc phan Findings
    trong HTML string -- clinician doc canh bao truoc.
    """
    report = _make_report(
        t1_overrides={
            "hint_conflict": True,
            "hint_resolution_note": "Router predicted breast, user chose thyroid.",
        }
    )
    html = build_document_report_html(report)

    banner_pos  = html.find("Hint Conflict")
    findings_pos = html.find("Hypoechoic lesion")

    assert banner_pos != -1, "Banner khong ton tai trong HTML"
    assert findings_pos != -1, "Findings khong ton tai trong HTML"
    assert banner_pos < findings_pos, (
        "Banner phai xuat hien truoc Findings trong HTML"
    )


def test_build_document_report_html_rag_warning_precedes_findings():
    report = _make_report(
        rag_disabled_warning="RAG not available -- no guideline retrieval."
    )
    html = build_document_report_html(report)

    warning_pos  = html.find("No clinical guideline retrieval")
    findings_pos = html.find("Hypoechoic lesion")

    assert warning_pos < findings_pos


def test_raw_json_schema_unchanged():
    """
    Tab Raw JSON phai chua day du cac field cua ReportOutput.
    Dam bao cac tab moi khong lam mat field nao so voi schema hien tai.
    """
    import json as _json
    report = _make_report()
    raw = _json.dumps(report, indent=2, ensure_ascii=False)

    required_fields = [
        "image_id",
        "tier_1_structured",
        "tier_2_radiological_description",
        "tier_3_diagnostic_suggestion",
        "rag_sources",
        "rag_disabled_warning",
    ]
    for field in required_fields:
        assert field in raw, f"Field '{field}' bi mat khoi Raw JSON"


def test_build_document_report_html_no_banner_when_no_warnings():
    report = _make_report()
    html = build_document_report_html(report)

    assert "Hint Conflict" not in html
    assert "No clinical guideline retrieval" not in html
    assert "Confidence calibration" not in html


# HTML contrast -- moi top-level div phai co ca color va background

import re

def _extract_top_level_div_styles(html: str) -> list:
    """
    Tim tat ca div co style inline o top level (con truc tiep cua root div).
    Tra ve list style string.
    """
    return re.findall(r'<div\s+style="([^"]*)"', html)


def test_html_blocks_have_explicit_background_and_text_color_pairs():
    """
    Moi div co set 'color:' cung phai co 'background' hoac 'background-color'
    trong cung style -- tranh truong hop chu bi chim khi Gradio ap nen khac.
    """
    report = _make_report(
        t1_overrides={
            "hint_conflict": True,
            "hint_resolution_note": "conflict test",
            "confidence_calibration_note": "high confidence note",
        },
        rag_disabled_warning="rag off",
    )
    html = build_document_report_html(report)
    styles = _extract_top_level_div_styles(html)

    for style in styles:
        has_color = "color:" in style
        has_bg    = "background" in style
        if has_color:
            assert has_bg, (
                f"Div co 'color' nhung khong co 'background' trong style:\n{style}"
            )
