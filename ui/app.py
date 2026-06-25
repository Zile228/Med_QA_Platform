"""
ui/app.py
==========
Gradio UI - demo interface for Med-Platform.

2-column x 2-row layout:
    Top left:     image upload + hint dropdown + Analyze button
    Bottom left:  annotated image (bbox overlay)
    Top right:    tabs [Document Report] [Raw JSON]
    Bottom right: multi-turn chatbot (the first question also goes through here)

Run:
    python ui/app.py
    or docker compose up ui
"""

import os
import io
import json
import html
import httpx
import gradio as gr
from PIL import Image, ImageDraw

ORCHESTRATOR_URL = os.getenv("ORCHESTRATOR_URL", "http://orchestrator:8000")
REQUEST_TIMEOUT  = int(os.getenv("UI_TIMEOUT", "180"))

SEVERITY_COLOR = {
    "incidental":  "#4CAF50",
    "significant": "#FF9800",
    "urgent":      "#F44336",
    "critical":    "#9C27B0",
}

EXAMPLE_QUESTIONS = [
    "What are the findings in this ultrasound image?",
    "Is this lesion benign or malignant?",
    "What is the BI-RADS category and recommended follow-up?",
    "Describe the size, location, and shape of the lesion.",
    "What are the suspicious features in this scan?",
]

# Maps dropdown display name -> value sent to the server
HINT_OPTIONS = {
    "Auto-detect": None,
    "Breast US":   "breast",
    "Thyroid US":  "thyroid",
}


# Orchestrator API calls

def call_orchestrator(
    image_pil: Image.Image,
    organ_hint: str = None,
) -> dict:
    """Send image + hint -> ReportOutput dict."""
    buf = io.BytesIO()
    image_pil.save(buf, format="PNG")
    buf.seek(0)

    data = {}
    if organ_hint:
        data["organ_hint"] = organ_hint

    with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
        resp = client.post(
            f"{ORCHESTRATOR_URL}/analyze",
            files={"image": ("image.png", buf, "image/png")},
            data=data,
        )

    if resp.status_code >= 400:
        try:
            detail = resp.json().get("detail", resp.text)
        except Exception:
            detail = resp.text
        prefix = "[Pipeline rejected]" if resp.status_code == 422 else f"[Orchestrator error {resp.status_code}]"
        raise gr.Error(f"{prefix} {detail}")
    resp.raise_for_status()
    return resp.json()


def call_chat(image_id: str, message: str, history: list) -> str:
    """
    Call /chat with the existing context - does not resend the image.
    history: list[dict] {"role": "user"|"assistant", "content": str}
    """
    with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
        resp = client.post(
            f"{ORCHESTRATOR_URL}/chat",
            json={
                "image_id": image_id,
                "message":  message,
                "history":  history,
            },
        )
    if resp.status_code in (404, 400):
        raise gr.Error(resp.json().get("detail", "Context not found."))
    resp.raise_for_status()
    return resp.json().get("reply", "")


# HTML and image rendering helpers

def draw_bbox_on_image(
    image_pil: Image.Image,
    bbox: list,
    label: str,
    color: str,
) -> Image.Image:
    """Draw the bounding box + label on the original image."""
    img = image_pil.copy().convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    if bbox and len(bbox) == 4 and any(v > 0 for v in bbox):
        x1, y1, x2, y2 = bbox
        r, g, b = tuple(int(color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))
        draw.rectangle([x1, y1, x2, y2], fill=(r, g, b, 40), outline=(r, g, b, 220), width=3)
        draw.rectangle([x1, y1 - 22, x1 + len(label) * 9 + 8, y1], fill=(r, g, b, 220))
        draw.text((x1 + 4, y1 - 20), label, fill="white")

    return Image.alpha_composite(img, overlay).convert("RGB")


def _build_rag_citations_html(rag_sources: list) -> str:
    """Render the citation list from rag_sources (list of dict {file, page})."""
    if not rag_sources:
        return ""

    items = ""
    for src in rag_sources:
        if isinstance(src, dict):
            fname = html.escape(src.get("file", "unknown"))
            page = src.get("page", 0)
            items += f'<li><code>{fname}</code>, page {page}</li>'
        else:
            items += f"<li>{html.escape(str(src))}</li>"

    return f"""
<div style="margin-top:10px; padding:10px 14px; background:#1a1a2e;
            border-left:4px solid #585b70; border-radius:4px; font-size:11px; color:#a6adc8;">
  <b>References (RAG):</b>
  <ul style="margin:4px 0 0 16px; padding:0;">{items}</ul>
</div>
"""


def build_warning_banners(report: dict, t1: dict) -> str:
    """
    Return the HTML string of warning banners placed at the top of the
    Document Report. The clinician must see these warnings before reading
    the findings content.
    """
    banners = ""

    if report.get("hard_conflict") is True:
        mapper_r = report.get("mapper_result") or {}
        cot_r    = report.get("cot_result") or {}
        t1_d     = report.get("tier_1_structured") or {}
        cnn_lbl  = html.escape(str(t1_d.get("label", "?")))
        cot_lbl  = html.escape(str(cot_r.get("cot_label", "?"))) if isinstance(cot_r, dict) else "?"
        banners += f"""
<div style="background:#3a0a0a; border-left:6px solid #f38ba8; padding:14px 16px;
            border-radius:4px; color:#ff9eae; font-size:13px; margin-bottom:10px;">
  <b>MANDATORY RADIOLOGIST REVIEW</b><br>
  CNN model classified this as <b>{cnn_lbl}</b>, but independent AI reasoning
  classified this as <b>{cot_lbl}</b>, and severity levels differ by more than 1 point.
  This report must not be used for clinical decision-making without radiologist confirmation.
  <br>Rule engine: severity={html.escape(str(mapper_r.get('severity','?')))} (level {html.escape(str(mapper_r.get('severity_level','?')))}),
  ICD-10={html.escape(str(mapper_r.get('icd10_hint','?')))}.
  CoT: severity={html.escape(str(cot_r.get('severity','?') if isinstance(cot_r,dict) else '?'))} (level {html.escape(str(cot_r.get('severity_level','?') if isinstance(cot_r,dict) else '?'))}),
  ICD-10={html.escape(str(cot_r.get('icd10_hint','?') if isinstance(cot_r,dict) else '?'))}.
</div>
"""

    if t1.get("hint_conflict"):
        note = html.escape(t1.get("hint_resolution_note") or "")
        banners += f"""
<div style="background:#1a1a2e; border-left:4px solid #89b4fa; padding:10px 14px;
            border-radius:4px; color:#cdd6f4; font-size:12px; margin-bottom:8px;">
  <b>[Hint Conflict]</b> {note}
</div>
"""

    rag_warning = report.get("rag_disabled_warning")
    if rag_warning:
        banners += f"""
<div style="background:#1e1a0a; border-left:4px solid #f9e2af; padding:10px 14px;
            border-radius:4px; color:#f9e2af; font-size:12px; margin-bottom:8px;">
  <b>[No clinical guideline retrieval]</b> {html.escape(rag_warning)}
</div>
"""

    calib_note = t1.get("confidence_calibration_note")
    if calib_note:
        banners += f"""
<div style="background:#1e1a0a; border-left:4px solid #f9e2af; padding:10px 14px;
            border-radius:4px; color:#f9e2af; font-size:12px; margin-bottom:8px;">
  <b>[Confidence calibration]</b> {html.escape(calib_note)}
</div>
"""

    # Warning banner when mapper and CoT disagree
    consensus = report.get("consensus")
    icd10_agreement = report.get("icd10_agreement")
    if consensus is False:
        mapper_r = report.get("mapper_result") or {}
        cot_r    = report.get("cot_result") or {}
        banners += f"""
<div style="background:#2a1a1a; border-left:4px solid #f38ba8; padding:10px 14px;
            border-radius:4px; color:#f5c2e7; font-size:12px; margin-bottom:8px;">
  <b>[Rule-Engine vs AI Reasoning Disagreement]</b>
  Rule engine: severity={html.escape(str(mapper_r.get('severity','?')))} (level {html.escape(str(mapper_r.get('severity_level','?')))}),
  ICD-10={html.escape(str(mapper_r.get('icd10_hint','?')))}.
  CoT reasoning: severity={html.escape(str(cot_r.get('severity','?')))} (level {html.escape(str(cot_r.get('severity_level','?')))}),
  ICD-10={html.escape(str(cot_r.get('icd10_hint','?')))}.
  <br><b>Radiologist confirmation required due to disagreement between rule-based and AI reasoning.</b>
</div>
"""
    elif icd10_agreement is False:
        # Severity agrees but ICD-10 codes differ - this is a separate
        # disagreement, not folded into the severity banner above.
        mapper_r = report.get("mapper_result") or {}
        cot_r    = report.get("cot_result") or {}
        banners += f"""
<div style="background:#2a1a1a; border-left:4px solid #f38ba8; padding:10px 14px;
            border-radius:4px; color:#f5c2e7; font-size:12px; margin-bottom:8px;">
  <b>[ICD-10 Code Disagreement]</b>
  Rule engine: ICD-10={html.escape(str(mapper_r.get('icd10_hint','?')))}.
  CoT reasoning: ICD-10={html.escape(str(cot_r.get('icd10_hint','?')))}.
  <br><b>Radiologist confirmation required due to ICD-10 code disagreement between rule-based and AI reasoning.</b>
</div>
"""

    return banners


def build_document_report_html(report: dict) -> str:
    """
    Combine the 3 tiers into one cohesive medical report following the
    standard radiology format:
    Banners -> Patient/Study Info -> Findings -> Impression -> Citations -> Disclaimer.
    """
    t1  = report.get("tier_1_structured", {})
    t2  = report.get("tier_2_radiological_description", "")
    t3  = report.get("tier_3_diagnostic_suggestion", "")

    severity = html.escape(t1.get("severity", "unknown"))
    color    = SEVERITY_COLOR.get(t1.get("severity", "unknown"), "#607D8B")
    level    = t1.get("severity_level", 0)
    dots     = "*" * level + "-" * (4 - level)

    banners = build_warning_banners(report, t1)
    citations_html = _build_rag_citations_html(report.get("rag_sources", []))

    return f"""
<div style="font-family: 'Segoe UI', monospace; background:#1e1e2e; color:#cdd6f4;
            padding:20px; border-radius:10px; line-height:1.8; font-size:13px;">

  {banners}

  <div style="background:#1e1e2e; font-size:15px; font-weight:bold; color:#89b4fa; margin-bottom:14px;
              border-bottom:1px solid #313244; padding-bottom:8px;">
    Radiology Report
  </div>

  <div style="background:#1e1e2e; color:#cdd6f4; margin-bottom:14px;">
    <span style="color:#a6e3a1; font-weight:bold;">Study Info</span><br>
    Modality / Organ: <b>{html.escape(t1.get('modality','?').upper())} / {html.escape(t1.get('organ','?').upper())}</b>
    &nbsp; | &nbsp; Image ID: <code style="color:#cba6f7">{html.escape(report.get('image_id','?'))}</code>
  </div>

  <div style="background:#1e1e2e; color:#cdd6f4; margin-bottom:14px;">
    <span style="color:#a6e3a1; font-weight:bold;">Classification</span><br>
    <b>{html.escape(t1.get('label','?').upper())}</b>
    <span style="color:#585b70"> ({t1.get('confidence',0):.0%} confidence)</span>
    &nbsp; | &nbsp; {html.escape(t1.get('risk_category','?'))}
    &nbsp; | &nbsp; ICD-10: <code>{html.escape(t1.get('icd10_hint','?'))}</code>
    {('<span style="color:#f38ba8; font-size:11px;"> (Rule-based / AI reasoning)</span>' if report.get('icd10_agreement') is False else '')}
  </div>

  <div style="background:#1e1e2e; color:#cdd6f4; margin-bottom:14px;">
    <span style="color:{color}; font-weight:bold;">{severity.upper()}</span>
    &nbsp;<span style="color:{color}">{dots}</span>
    <span style="color:#a6adc8"> (level {level}/4)</span>
  </div>

  <div style="background:#1e1e2e; color:#cdd6f4; margin-bottom:14px;">
    <span style="color:#a6e3a1; font-weight:bold;">Spatial Measurements</span><br>
    Location: <b>{html.escape(t1.get('location_quadrant','?'))}</b>
    &nbsp; | &nbsp; Area: <b>{f"{t1['area_cm2']:.3f} cm2" if t1.get('area_cm2') is not None else "unavailable (no DICOM metadata)"}</b>
    &nbsp; | &nbsp; Aspect ratio: {t1.get('aspect_ratio',0):.3f}
    {f'&nbsp;<span style="color:#a6adc8; font-size:11px;">[{html.escape(t1.get("aspect_ratio_interpretation",""))}]</span>' if t1.get('aspect_ratio_interpretation') else ''}
    &nbsp; | &nbsp; Circularity: {t1.get('circularity',0):.3f}
    {'&nbsp;<span style="color:#f38ba8">[irregular margin]</span>' if t1.get('circularity',0) < 0.5 else ''}
  </div>

  <div style="margin-bottom:14px; padding:12px; background:#25253a; border-radius:6px;">
    <span style="color:#a6e3a1; font-weight:bold;">Findings</span><br>
    <span style="color:#cdd6f4">{html.escape(t2)}</span>
  </div>

  <div style="margin-bottom:14px; padding:12px; background:#25253a; border-radius:6px;">
    <span style="color:#89b4fa; font-weight:bold;">Impression &amp; Recommendation</span><br>
    <span style="color:#cdd6f4">{html.escape(t3)}</span>
  </div>

  {citations_html}

  <div style="background:#2a1a1a; border-left:4px solid #f38ba8; padding:10px 14px;
              border-radius:4px; color:#f5c2e7; font-size:12px; margin-top:4px;">
    Disclaimer: This AI-generated report is for screening assistance only and does
    not constitute a medical diagnosis. All findings must be reviewed and confirmed by a
    qualified radiologist or physician.
  </div>

</div>
"""


# Gradio callbacks

def run_analysis(
    image_pil: Image.Image,
    hint_display: str,
    progress=gr.Progress(),
):
    """
    Gradio callback when the user clicks Analyze.

    No longer takes a question from the UI -- always uses EXAMPLE_QUESTIONS[0]
    to seed the first RAG retrieve and Tier 2/3. The user's actual question
    now goes through the Follow-up Questions chatbot after results are in.

    Returns:
        annotated_image, document_report_html, raw_json, image_id_state

    Note on Gradio 5+/6+ compatibility:
        Depending on the Gradio version, the Image component may deliver the
        uploaded file as a PIL.Image, a filepath string, or a dict like
        {'path': '/tmp/gradio/...', 'url': '...'}. We normalise all three
        shapes here so the rest of the function always works with a PIL image.
    """
    # --- Normalise Gradio image input shapes ---
    if image_pil is None:
        raise gr.Error("Please upload an image first.")

    if isinstance(image_pil, dict):
        # Gradio 5+/6+ may pass a dict when the temp file lifecycle races with
        # the preprocess step. Extract the path and open it ourselves.
        path = image_pil.get("path") or image_pil.get("name")
        if not path or not os.path.exists(path):
            raise gr.Error(
                "The uploaded image could not be read (temp file missing). "
                "Please re-upload the image and try again."
            )
        image_pil = Image.open(path).copy()  # .copy() detaches from the file handle

    elif isinstance(image_pil, str):
        # Gradio type="filepath" passes a string path
        if not os.path.exists(image_pil):
            raise gr.Error(
                f"Image file not found: {image_pil}. Please re-upload and try again."
            )
        image_pil = Image.open(image_pil).copy()
    # --- End normalisation ---
    organ_hint = HINT_OPTIONS.get(hint_display or "Auto-detect")

    progress(0.1, desc="Sending to orchestrator...")

    try:
        report = call_orchestrator(image_pil, organ_hint=organ_hint)
    except gr.Error:
        raise
    except Exception as e:
        raise gr.Error(
            f"Connection error: {e}\nCheck that the orchestrator is running at {ORCHESTRATOR_URL}"
        )

    progress(0.8, desc="Rendering report...")

    t1     = report.get("tier_1_structured", {})
    bbox   = t1.get("bbox", [0, 0, 0, 0])
    label  = t1.get("label", "unknown")
    color  = SEVERITY_COLOR.get(t1.get("severity", "incidental"), "#607D8B")

    annotated     = draw_bbox_on_image(image_pil, bbox, label, color)
    document_html = build_document_report_html(report)
    raw_json      = json.dumps(report, indent=2, ensure_ascii=False)
    image_id      = report.get("image_id", "")

    progress(1.0, desc="Done")
    return annotated, document_html, raw_json, image_id


def send_chat(
    message: str,
    chat_history: list,
    image_id: str,
):
    """
    Gradio callback for the chatbot - calls /chat without resending the image.

    chat_history: list[dict] {"role": "user"|"assistant", "content": str}.
    Must be sanitized before sending to the orchestrator (Gradio may add a
    stray key).
    """
    if not image_id:
        raise gr.Error("No image has been analyzed yet. Please run Analyze first.")
    if not message.strip():
        return chat_history, ""

    # Keep only role + content str, skip invalid entries
    history_dicts = []
    for turn in chat_history:
        role = turn.get("role")
        content = turn.get("content")
        if role not in ("user", "assistant") or not isinstance(content, str):
            continue
        history_dicts.append({"role": role, "content": content})

    try:
        reply = call_chat(image_id, message, history_dicts)
    except gr.Error:
        raise
    except Exception as e:
        raise gr.Error(f"Chat error: {e}")

    chat_history = chat_history + [
        {"role": "user",      "content": message},
        {"role": "assistant", "content": reply},
    ]
    return chat_history, ""


# Build UI layout

# Force dark mode, independent of OS setting
_FORCE_DARK_JS = """
() => {
    document.documentElement.setAttribute('data-theme', 'dark');
    document.body.style.colorScheme = 'dark';
}
"""

_CSS = """
body, .gradio-container {
    background-color: #1e1e2e !important;
    color: #cdd6f4 !important;
}
.report-box textarea {
    font-size: 13px !important;
    line-height: 1.7 !important;
    background: #1e1e2e !important;
    color: #cdd6f4 !important;
    border: 1px solid #313244 !important;
}
.title-text { text-align: center; padding: 12px 0; }
label, .label-wrap span, .svelte-1gfkn6j {
    color: #a6e3a1 !important;
}
footer { display: none !important; }

.message-wrap, [data-testid="bot"], [data-testid="user"] {
    color: #cdd6f4 !important;
}
.bubble, .message, .message-bubble-border,
.role-assistant .message-content, .role-user .message-content,
[data-testid="bot"] .prose, [data-testid="user"] .prose {
    background-color: #25253a !important;
    color: #cdd6f4 !important;
    border: 1px solid #313244 !important;
}
.role-user .message-content, [data-testid="user"] .prose {
    background-color: #313244 !important;
}
.bubble strong, .message strong, .prose strong {
    color: #f5f5fa !important;
}
.bubble a, .message a, .prose a {
    color: #89b4fa !important;
}
"""


def build_ui() -> gr.Blocks:
    with gr.Blocks(
        title="Med-Platform -- AI Medical Imaging Assistant",
    ) as demo:

        # Store image_id so the chatbot knows which analysis is being asked about
        image_id_state = gr.State("")

        gr.HTML("""
        <div class="title-text" style="background:#1e1e2e; padding:12px 0;">
          <h1 style="margin:0; color:#89b4fa;">Med-Platform</h1>
          <p style="margin:4px 0 0; color:#a6adc8; font-size:14px;">
            AI-assisted Medical Imaging -- Breast &amp; Thyroid Ultrasound
          </p>
        </div>
        """)

        with gr.Row():

            # Left column: input and annotated image
            with gr.Column(scale=1):

                image_input = gr.Image(
                    type="pil",
                    label="Upload Ultrasound Image",
                    sources=["upload", "clipboard"],
                    height=280,
                )
                modality_hint_dropdown = gr.Dropdown(
                    choices=list(HINT_OPTIONS.keys()),
                    value="Auto-detect",
                    label="Modality / Organ hint",
                )
                analyze_btn = gr.Button("Analyze", variant="primary", size="lg")
                gr.HTML("""
                <div style="font-size:11px; color:#585b70; margin-top:8px; text-align:center;
                            background:#1e1e2e;">
                  Supported: Breast US (BUSI) &middot; Thyroid US
                </div>
                """)

                annotated_output = gr.Image(
                    label="Annotated Image (bbox overlay)",
                    height=280,
                    interactive=False,
                )

            # Right column: report and chatbot
            with gr.Column(scale=2):

                with gr.Tabs():
                    with gr.Tab("Document Report"):
                        document_report_output = gr.HTML(
                            value=(
                                '<div style="color:#585b70; padding:16px; background:#1e1e2e;">'
                                'Run analysis to see results.'
                                '</div>'
                            )
                        )

                    with gr.Tab("Raw JSON"):
                        raw_output = gr.Code(
                            label="Full ReportOutput",
                            language="json",
                            lines=20,
                        )

                gr.HTML("""
                <div style="color:#a6e3a1; font-size:13px; font-weight:bold;
                            margin:12px 0 4px; background:#1e1e2e;">
                  Follow-up Questions
                </div>
                <div style="color:#585b70; font-size:11px; margin-bottom:8px; background:#1e1e2e;">
                  Ask follow-up questions about the analyzed results -- no need to re-upload the image.
                </div>
                """)
                chatbot = gr.Chatbot(
                    label="Chatbot",
                    height=260,
                    show_label=False,
                )
                with gr.Row():
                    chat_input = gr.Textbox(
                        placeholder=f"e.g. \"{EXAMPLE_QUESTIONS[1]}\"",
                        label="",
                        lines=1,
                        scale=5,
                        show_label=False,
                    )
                    chat_btn = gr.Button("Send", scale=1, size="sm")

        # Wire up events to callbacks
        analyze_btn.click(
            fn=run_analysis,
            inputs=[image_input, modality_hint_dropdown],
            outputs=[annotated_output, document_report_output, raw_output, image_id_state],
            api_name="analyze",
        )

        chat_btn.click(
            fn=send_chat,
            inputs=[chat_input, chatbot, image_id_state],
            outputs=[chatbot, chat_input],
        )
        chat_input.submit(
            fn=send_chat,
            inputs=[chat_input, chatbot, image_id_state],
            outputs=[chatbot, chat_input],
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        favicon_path=None,
        theme=gr.themes.Base(
            primary_hue="blue",
            secondary_hue="slate",
            neutral_hue="slate",
        ).set(
            body_background_fill="#1e1e2e",
            body_text_color="#cdd6f4",
            block_background_fill="#25253a",
            block_border_color="#313244",
            block_label_text_color="#a6e3a1",
            input_background_fill="#181825",
            input_border_color="#313244",
            button_primary_background_fill="#89b4fa",
            button_primary_text_color="#1e1e2e",
        ),
        css=_CSS,
        js=_FORCE_DARK_JS,
    )