"""
ui/app.py
==========
Gradio UI -- giao dien demo cho Med-Platform.

Layout 2 cot x 2 tang:
    Trai tren:  upload anh + dropdown hint + cau hoi + nut Analyze
    Trai duoi:  anh annotated (bbox overlay)
    Phai tren:  tabs [Document Report] [Raw JSON]
    Phai duoi:  chatbot hoi-dap nhieu luot

Chay:
    python ui/app.py
    hoac docker compose up ui
"""

import os
import io
import json
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

# Map ten hien thi trong dropdown -> gia tri gui len server
HINT_OPTIONS = {
    "Auto-detect": None,
    "Breast US":   "breast",
    "Thyroid US":  "thyroid",
}


# Goi API orchestrator

def call_orchestrator(
    image_pil: Image.Image,
    question: str,
    organ_hint: str = None,
) -> dict:
    """Gui anh + question + hint -> ReportOutput dict."""
    buf = io.BytesIO()
    image_pil.save(buf, format="PNG")
    buf.seek(0)

    data = {"question": question}
    if organ_hint:
        data["organ_hint"] = organ_hint

    with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
        resp = client.post(
            f"{ORCHESTRATOR_URL}/analyze",
            files={"image": ("image.png", buf, "image/png")},
            data=data,
        )

    if resp.status_code == 422:
        raise gr.Error(f"[Pipeline rejected] {resp.json().get('detail', 'Unknown error')}")
    resp.raise_for_status()
    return resp.json()


def call_chat(image_id: str, message: str, history: list) -> str:
    """
    Goi /chat voi context da co -- khong gui lai anh.
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
        raise gr.Error(resp.json().get("detail", "Context khong tim thay."))
    resp.raise_for_status()
    return resp.json().get("reply", "")


# Helper render HTML va anh

def draw_bbox_on_image(
    image_pil: Image.Image,
    bbox: list,
    label: str,
    color: str,
) -> Image.Image:
    """Ve bounding box + label len anh goc."""
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


def build_warning_banners(report: dict, t1: dict) -> str:
    """
    Tra ve HTML chuoi cac banner canh bao de dat len dau Document Report.
    Clinician phai thay canh bao truoc khi doc noi dung findings.
    """
    banners = ""

    if t1.get("hint_conflict"):
        note = t1.get("hint_resolution_note") or ""
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
  <b>[No clinical guideline retrieval]</b> {rag_warning}
</div>
"""

    calib_note = t1.get("confidence_calibration_note")
    if calib_note:
        banners += f"""
<div style="background:#1e1a0a; border-left:4px solid #f9e2af; padding:10px 14px;
            border-radius:4px; color:#f9e2af; font-size:12px; margin-bottom:8px;">
  <b>[Confidence calibration]</b> {calib_note}
</div>
"""

    return banners


def build_document_report_html(report: dict) -> str:
    """
    Gop 3 tier thanh 1 bao cao y khoa lien mach theo format radiology chuan:
    Banners -> Patient/Study Info -> Findings -> Impression -> Disclaimer.
    """
    t1  = report.get("tier_1_structured", {})
    t2  = report.get("tier_2_radiological_description", "")
    t3  = report.get("tier_3_diagnostic_suggestion", "")

    severity = t1.get("severity", "unknown")
    color    = SEVERITY_COLOR.get(severity, "#607D8B")
    level    = t1.get("severity_level", 0)
    dots     = "*" * level + "-" * (4 - level)

    banners = build_warning_banners(report, t1)

    # Canh bao dat truoc Findings theo yeu cau cua TODO section 5.1
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
    Modality / Organ: <b>{t1.get('modality','?').upper()} / {t1.get('organ','?').upper()}</b>
    &nbsp; | &nbsp; Image ID: <code style="color:#cba6f7">{report.get('image_id','?')}</code>
  </div>

  <div style="background:#1e1e2e; color:#cdd6f4; margin-bottom:14px;">
    <span style="color:#a6e3a1; font-weight:bold;">Classification</span><br>
    <b>{t1.get('label','?').upper()}</b>
    <span style="color:#585b70"> ({t1.get('confidence',0):.0%} confidence)</span>
    &nbsp; | &nbsp; {t1.get('risk_category','?')}
    &nbsp; | &nbsp; ICD-10: <code>{t1.get('icd10_hint','?')}</code>
  </div>

  <div style="background:#1e1e2e; color:#cdd6f4; margin-bottom:14px;">
    <span style="color:{color}; font-weight:bold;">{severity.upper()}</span>
    &nbsp;<span style="color:{color}">{dots}</span>
    <span style="color:#a6adc8"> (level {level}/4)</span>
  </div>

  <div style="background:#1e1e2e; color:#cdd6f4; margin-bottom:14px;">
    <span style="color:#a6e3a1; font-weight:bold;">Spatial Measurements</span><br>
    Location: <b>{t1.get('location_quadrant','?')}</b>
    &nbsp; | &nbsp; Area: <b>{t1.get('area_cm2',0):.3f} cm2</b>
    &nbsp; | &nbsp; Aspect ratio: {t1.get('aspect_ratio',0):.3f}
    {'&nbsp;<span style="color:#f38ba8">[elongated]</span>' if t1.get('aspect_ratio',0) > 1.5 else ''}
    &nbsp; | &nbsp; Circularity: {t1.get('circularity',0):.3f}
    {'&nbsp;<span style="color:#f38ba8">[irregular margin]</span>' if t1.get('circularity',0) < 0.5 else ''}
  </div>

  <div style="margin-bottom:14px; padding:12px; background:#25253a; border-radius:6px;">
    <span style="color:#a6e3a1; font-weight:bold;">Findings</span><br>
    <span style="color:#cdd6f4">{t2}</span>
  </div>

  <div style="margin-bottom:14px; padding:12px; background:#25253a; border-radius:6px;">
    <span style="color:#89b4fa; font-weight:bold;">Impression &amp; Recommendation</span><br>
    <span style="color:#cdd6f4">{t3}</span>
  </div>

  <div style="background:#2a1a1a; border-left:4px solid #f38ba8; padding:10px 14px;
              border-radius:4px; color:#f5c2e7; font-size:12px; margin-top:4px;">
    Disclaimer: This AI-generated report is for screening assistance only and does
    not constitute a medical diagnosis. All findings must be reviewed and confirmed by a
    qualified radiologist or physician.
  </div>

</div>
"""


# Callback cua Gradio

def run_analysis(
    image_pil: Image.Image,
    question: str,
    hint_display: str,
    progress=gr.Progress(),
):
    """
    Gradio callback khi user nhan Analyze.

    Returns:
        annotated_image, document_report_html, raw_json, image_id_state
    """
    if image_pil is None:
        raise gr.Error("Vui long upload anh truoc.")
    if not question.strip():
        question = EXAMPLE_QUESTIONS[0]

    organ_hint = HINT_OPTIONS.get(hint_display or "Auto-detect")

    progress(0.1, desc="Sending to orchestrator...")

    try:
        report = call_orchestrator(image_pil, question, organ_hint=organ_hint)
    except gr.Error:
        raise
    except Exception as e:
        raise gr.Error(
            f"Connection error: {e}\nKiem tra orchestrator dang chay tai {ORCHESTRATOR_URL}"
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
    Gradio callback cho chatbot -- goi /chat khong gui lai anh.

    chat_history: list[dict] {"role": "user"|"assistant", "content": str}.
    Phai sanitize truoc khi gui sang orchestrator (Gradio co the them key la).
    """
    if not image_id:
        raise gr.Error("Chua co anh duoc phan tich. Vui long Analyze truoc.")
    if not message.strip():
        return chat_history, ""

    # Chi giu role + content str, bo qua entry khong hop le
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


# Xay dung layout UI

# Ep dark mode, khong phu thuoc setting OS
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

/* Ghi de mau nen/chu cho chatbot bubble cua Gradio */
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

        # Luu image_id de chatbot biet phan tich nao dang duoc hoi
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

            # Cot trai: input va anh annotated
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
                question_input = gr.Textbox(
                    label="Clinical Question",
                    placeholder=EXAMPLE_QUESTIONS[0],
                    lines=2,
                )
                gr.Examples(
                    examples=[[q] for q in EXAMPLE_QUESTIONS],
                    inputs=[question_input],
                    label="Example questions",
                    examples_per_page=5,
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

            # Cot phai: bao cao va chatbot
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
                  Hoi them ve ket qua da phan tich -- khong can upload lai anh.
                </div>
                """)
                chatbot = gr.Chatbot(
                    label="Chatbot",
                    height=260,
                    show_label=False,
                )
                with gr.Row():
                    chat_input = gr.Textbox(
                        placeholder="Hoi them ve ket qua nay...",
                        label="",
                        lines=1,
                        scale=5,
                        show_label=False,
                    )
                    chat_btn = gr.Button("Send", scale=1, size="sm")

        # Ket noi events voi callback
        analyze_btn.click(
            fn=run_analysis,
            inputs=[image_input, question_input, modality_hint_dropdown],
            outputs=[annotated_output, document_report_output, raw_output, image_id_state],
            api_name="analyze",
        )
        question_input.submit(
            fn=run_analysis,
            inputs=[image_input, question_input, modality_hint_dropdown],
            outputs=[annotated_output, document_report_output, raw_output, image_id_state],
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
