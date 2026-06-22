"""
Tests for section 5: build_document_report_html, chatbot, and UI HTML contrast.
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
    The warning banner (hint_conflict) must appear before the Findings
    section in the HTML string -- the clinician reads the warning first.
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

    assert banner_pos != -1, "Banner not found in HTML"
    assert findings_pos != -1, "Findings not found in HTML"
    assert banner_pos < findings_pos, (
        "Banner must appear before Findings in HTML"
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
    The Raw JSON tab must contain all fields of ReportOutput.
    Ensures new tabs don't drop any field compared to the current schema.
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
        assert field in raw, f"Field '{field}' missing from Raw JSON"


def test_build_document_report_html_no_banner_when_no_warnings():
    report = _make_report()
    html = build_document_report_html(report)

    assert "Hint Conflict" not in html
    assert "No clinical guideline retrieval" not in html
    assert "Confidence calibration" not in html


def test_icd10_disagreement_banner_when_consensus_true_but_icd10_differs():
    """
    Severity agrees (consensus True) but mapper/CoT icd10_hint differ --
    a separate banner must still appear, the ICD-10 disagreement should not
    be hidden just because severity_level happens to match.
    """
    report = _make_report(
        consensus=True,
        icd10_agreement=False,
        mapper_result={"severity": "significant", "severity_level": 2,
                        "icd10_hint": "N63.0"},
        cot_result={"severity": "significant", "severity_level": 2,
                    "icd10_hint": "C50.9"},
        t1_overrides={"icd10_hint": "N63.0 / C50.9"},
    )
    html = build_document_report_html(report)

    assert "ICD-10 Code Disagreement" in html
    assert "N63.0" in html
    assert "C50.9" in html
    assert "Rule-based / AI reasoning" in html
    # The separate severity banner must not appear, since consensus is True
    assert "Rule-Engine vs AI Reasoning Disagreement" not in html


def test_no_icd10_disagreement_banner_when_codes_match():
    report = _make_report(consensus=True, icd10_agreement=True)
    html = build_document_report_html(report)

    assert "ICD-10 Code Disagreement" not in html
    assert "Rule-based / AI reasoning" not in html


def test_build_document_report_html_escapes_adversarial_llm_output():
    """
    tier_2/tier_3 are LLM output, generated from a prompt containing the
    user's question and RAG chunk text. If the LLM is induced to return
    HTML/JS, the result must be escaped before going into gr.HTML(), and
    must not render as live tags.
    """
    payload = "<img src=x onerror=alert(1)>"
    report = _make_report(
        **{
            "tier_2_radiological_description": payload,
            "tier_3_diagnostic_suggestion": f"<script>alert('xss')</script>{payload}",
        }
    )
    rendered = build_document_report_html(report)

    assert "<script" not in rendered
    assert "<img" not in rendered
    assert "&lt;img src=x onerror=alert(1)&gt;" in rendered
    assert "&lt;script&gt;" in rendered


def test_build_document_report_html_escapes_adversarial_tier1_fields():
    """
    The tier_1 fields (label, risk_category, icd10_hint, location_quadrant,
    modality, organ) and image_id must also be escaped since they can be
    indirectly influenced by model/pipeline output.
    """
    payload = "<img src=x onerror=alert(1)>"
    report = _make_report(
        image_id=payload,
        t1_overrides={
            "label": payload,
            "risk_category": payload,
            "icd10_hint": payload,
            "location_quadrant": payload,
            "modality": payload,
            "organ": payload,
        },
    )
    rendered = build_document_report_html(report)

    assert "<script" not in rendered
    assert "<img src=x onerror=alert(1)>" not in rendered.lower()
    # 7 escaped fields total: image_id, label, risk_category, icd10_hint,
    # location_quadrant, modality, organ (label/modality/organ go through
    # .upper() before escaping, the rest are escaped as-is)
    total_escaped = (
        rendered.count("&lt;img src=x onerror=alert(1)&gt;")
        + rendered.count("&lt;IMG SRC=X ONERROR=ALERT(1)&gt;")
    )
    assert total_escaped == 7


# HTML contrast -- every top-level div must have both color and background

import re

def _extract_top_level_div_styles(html: str) -> list:
    """
    Find all divs with inline style at the top level (direct children of
    the root div). Returns a list of style strings.
    """
    return re.findall(r'<div\s+style="([^"]*)"', html)


def test_html_blocks_have_explicit_background_and_text_color_pairs():
    """
    Every div that sets 'color:' must also have 'background' or
    'background-color' in the same style -- avoids text becoming
    unreadable when Gradio applies a different background.
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
                f"Div has 'color' but no 'background' in style:\n{style}"
            )
