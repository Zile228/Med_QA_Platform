# Scalable Multi-Agent Medical Imaging and Clinical Question Answering Platform

A microservice platform that takes a breast or thyroid ultrasound image and returns a structured, citation-backed, 3-tier radiology-style report. The pipeline combines a deep-learning vision stack (modality routing -> segmentation/classification), a deterministic rule-based clinical knowledge layer, an independent LLM Chain-of-Thought reasoning agent, and a RAG layer over real clinical guideline PDFs (BI-RADS, TI-RADS, ATA, ICD-10), all coordinated by a LangGraph orchestrator with fan-out/fan-in parallelism, a consistency guard, and full observability (OpenTelemetry + Jaeger + Prometheus + Grafana).

> **For research / demo purposes only.** Every report carries an explicit disclaimer and is designed to *never* present a definitive diagnosis — findings must be confirmed by a qualified radiologist.

---

## Table of Contents

- [What it does](#what-it-does)
- [Pipeline / Architecture](#pipeline--architecture)
- [Layer-by-layer detail](#layer-by-layer-detail)
- [Repository structure](#repository-structure)
- [Quickstart](#quickstart)
- [Configuration (.env)](#configuration-env)
- [API Reference](#api-reference)
- [UI](#ui)
- [Observability](#observability)
- [Testing](#testing)
- [Known limitations & roadmap](#known-limitations--roadmap)
- [Disclaimer](#disclaimer)

---

## What it does

1. A user uploads an ultrasound image (optionally hinting `breast` or `thyroid`).
2. The platform automatically routes the image to the right modality, segments and classifies the lesion, derives spatial features (size, shape, location) from the segmentation mask, maps everything to clinical risk categories and ICD‑10 codes via an auditable rule engine, cross-checks that rule-based judgment against an independent LLM reasoning pass, retrieves and re-ranks supporting passages from real clinical guideline PDFs, and finally asks an LLM to write a structured report.
3. The response is a 3-tier report: structured fields (Tier 1, no LLM needed), a natural-language radiological description (Tier 2), and a diagnostic suggestion with follow-up recommendation (Tier 3) - plus PDF/page citations, a disagreement banner if the rule engine and the LLM reasoning disagree, and a multi-turn chat endpoint to ask follow-up questions about the same image without re-uploading it.


## Pipeline / Architecture

```
                 ┌─────────┐
   image ───────▶│  route  │  Layer 1 - modality router (EfficientNet-B0)
                 └────┬────┘
                      │
        ┌─────────────┴──────────────┐
        ▼                            ▼
   ┌──────────┐                ┌─────────────┐
   │  vision  │                │ rag_retrieve│  ← runs IN PARALLEL with vision
   │ (seg+cls)│                │ (pass 1: Q  │     (query = user question only,
   └────┬─────┘                │   only)     │      vision result not known yet)
        ▼                      └──────┬──────┘
   ┌──────────┐                       │
   │ spatial  │  bbox / area / shape  │
   │ (mask →  │  from segmentation    │
   │ features)│  mask, BEFORE fan-out │
   └────┬─────┘                       │
        │                             │
   ┌────┴─────────┐                   │
   ▼              ▼                   │
┌──────────┐ ┌──────────────┐         │
│knowledge │ │cot_reasoning │         │
│(rule-based│ │(LLM, blind to│        │
│ mapper)  │ │ mapper result)│        │
└────┬─────┘ └──────┬───────┘         │
     └───────────────┴────────────────┘
                      ▼
              ┌──────────────┐
              │     merge     │  ← barrier: waits for ALL 3 branches
              └──────┬────────┘
                      ▼
           ┌────────────────────┐
           │  consistency_guard  │  pass 2 RAG (query enriched with
           │  - 2nd RAG retrieve │  label+organ+icd10) + rerank (cross-
           │  - rerank top-3     │  encoder); compares mapper vs CoT:
           │  - compare mapper   │   consensus        (severity_level Δ≤1)
           │    vs CoT           │   icd10_agreement  (icd10_hint match)
           └──────────┬──────────┘
                      ▼
              ┌──────────────┐
              │   qa_agent    │  ← LLM writes the 3-tier report,
              └──────┬────────┘     surfaces disagreement if any
                      ▼
                    END
```


### Sequence per request

| Step | Service | What happens |
|---|---|---|
| 1 | **Router** (`:8001`) | EfficientNet-B0 classifies the image into `us_breast` / `us_thyroid`, or rejects it as **out-of-distribution** if confidence < threshold. A user-supplied organ hint is blended in via a weighted formula; conflicts are recorded, not hidden. |
| 2 | **Vision** (`:8002`) | A multi-task U‑Net (EfficientNet‑B4 encoder) produces a **segmentation mask** + **classification** + a summary of the encoder's bottleneck activations (used later as a coarse "what is the model looking at" signal for the LLM). |
| 3 | **Orchestrator: spatial** | Decodes the mask, computes bounding box, area (cm2), centroid, aspect ratio, circularity, and breast quadrant / thyroid lobe location via OpenCV contour analysis. |
| 4a | **Knowledge** (`:8003`) | Pure rule-based lookup: label -> BI-RADS/TI-RADS category, severity (`incidental`/`significant`/`urgent`/`critical`), ICD-10 hint, plus an automatic "confidence may be miscalibrated" note for suspiciously high softmax scores. |
| 4b | **CoT reasoning** | An LLM independently reasons step-by-step (classification -> spatial -> bottleneck -> RAG) and outputs its own severity/ICD-10/risk_category - **without seeing the rule engine's answer.** |
| 4c | **RAG retrieve (pass 1)** | FAISS + sentence-transformer embeddings retrieve candidate guideline passages using only the user's question, in parallel with vision/knowledge so latency isn't wasted waiting. |
| 5 | **Consistency guard** | Re-retrieves with an enriched query (question + label + organ + ICD-10), merges + dedups with pass 1, reranks with a cross-encoder, keeps top 3; compares mapper vs. CoT severity (`consensus`) and ICD-10 code (`icd10_agreement`) as two **independent** flags. |
| 6 | **QA agent (LLM)** | Builds one prompt from all of the above and asks the LLM for Tier 2 (radiological description) and Tier 3 (diagnostic suggestion). If mapper and CoT disagree, the prompt explicitly instructs the LLM to present both views and recommend radiologist confirmation. |
| 7 | **Response** | `ReportOutput`: Tier 1 (structured, parsed without any LLM call) + Tier 2 + Tier 3 + RAG citations (`file`, `page`) + disclaimer + `consensus`/`icd10_agreement` flags. Context is cached in-process by `image_id` so a follow-up `/chat` call can ask questions without resending the image. |

## Layer-by-layer detail

### Layer 1 - Router (`services/router`)
- **Model:** `EfficientNet-B0` (via `timm`) + a small custom FC head -> 2 classes (`us_breast`, `us_thyroid`). Designed so adding an X-ray class later is a config change, not a rewrite.
- **OOD rejection:** if `max(softmax) < OOD_THRESHOLD` (default `0.6` in `.env.example`, tune per deployment), the image is flagged `is_ood=True` and the orchestrator blocks the pipeline before vision is ever called.
- **Hint resolution:** `model.py::resolve_with_hint()` blends router probabilities with a user-chosen hint using a weighted formula (`ROUTER_HINT_ROUTER_WEIGHT`, default `0.7` toward the router), and records whether the hint conflicted with the router's own top‑1 guess.
- **Degraded mode:** if no checkpoint file exists, the service still starts (useful for local dev) but every response is tagged `router_degraded=True`; the orchestrator refuses to act on a degraded routing decision unless `ALLOW_DEGRADED_ROUTER=true`.

### Layer 2 - Vision (`services/vision`)
- **Architecture:** `UNet_MTL`, an EfficientNet-B4 encoder shared between two heads: a U-Net-style decoder for **segmentation** (sigmoid mask) and a small FC head off the bottleneck for **classification**.
<p align="center">
  <img src="images/MTL architecture.png" width="600">
</p>

- **Two near-identical modules**, one per organ, differing only in `NUM_CLASSES` (3 for breast/BUSI, 2 for thyroid/TN3K), normalization stats, and class mapping:
  - `services/vision/us_breast` - trained on the **BUSI** dataset.
  - `services/vision/us_thyroid` - trained on the **TN3K** dataset.
- **Bottleneck features:** `extract_bottleneck_summary()` reduces the (448, 7, 7) encoder bottleneck into `activation_energy`, `top_channel_activations`, and an `attention_hotspot_grid` coordinate, a lightweight, text-readable proxy for "where is the model focusing," consumed later by the LLM prompts as a soft uncertainty signal (if attention diverges far from the segmentation bbox, that's a hint of model uncertainty).
- **Mask transport:** the mask is returned as a base64-encoded PNG over the HTTP response body, never written to a shared disk path, vision and knowledge are separate containers.
- **X-ray endpoint (`/analyze/xray`)** exists as a stub returning HTTP 501, explicitly scoped as Phase 2, not implemented.

### Layer 3 - Knowledge (`services/knowledge`)
- **No ML.** `mapper.py` is hardcoded lookup tables mapping `(organ, label)` -> BI-RADS/TI-RADS category, severity level (1–4), ICD‑10 code, and a clinical description string for the LLM prompt — deliberately auditable by a clinician without reading ML code.
- **Severity escalation rule:** `malignant` + confidence >= 0.9 escalates to `critical`; `malignant` + confidence < 0.5 is downgraded back to `significant`.
- **Calibration warning:** confidence >= `CONFIDENCE_CALIBRATION_THRESHOLD` (default `0.999`) attaches a note warning that such a high score on a small, non-calibrated training set should be read as a *ranking signal*, not a calibrated probability.
- **Spatial derivation:** wraps the organ-specific `postprocess_mask()` (OpenCV contour -> bbox, area in cm2 via `pixel_spacing_mm`, centroid, aspect ratio, circularity) and breast-quadrant / thyroid-lobe localization with a location-confidence estimate based on distance to the image edge.

### Layer 4 - Orchestrator (`services/orchestrator`)
- **Graph engine:** LangGraph `StateGraph` if installed, otherwise `AsyncSequentialFallback` (a hand-written async class that replicates the exact same partial-state-merge semantics, so behavior is identical either way), used in CI/tests when `langgraph` isn't available.
- **RAG store:** `rag/faiss_store.py`, FAISS similarity search over `sentence-transformers/all-MiniLM-L6-v2` embeddings, with `organ`-aware filtering (a chunk tagged `breast` is hidden from thyroid queries unless tagged `general`), de-duplication, and `cross-encoder/ms-marco-MiniLM-L-6-v2` reranking.
- **LLM client:** one `BaseLLMClient` interface, three backends: `OllamaClient` (local, default, no key, uses Ollama's native `/api/chat` for multi-turn), `GoogleGeminiClient` (`gemini-2.5-flash` by default), `MockLLMClient` (offline/dev placeholder).
- **Module registry:** `module_registry.yaml` is the single source of truth for service URLs, endpoint paths, and which vision modalities are `enabled` by adding a disabled-by-default modality (like the X-ray stub) is a YAML edit, and the orchestrator refuses to call a disabled module even if asked.
- **Chat endpoint:** `/chat` reuses the cached analysis context (TTL-based in-process dict, configurable via `CHAT_CONTEXT_TTL`) to answer follow-up questions without re-running vision/router/knowledge, only the LLM is called again. Documented limitation: this cache is per-process, so it does not work if the orchestrator is horizontally scaled beyond 1 replica without swapping in Redis/a real session store.

### UI (`ui/app.py`)
A Gradio app (2x2 layout: upload + hint dropdown + Analyze button -> annotated image with the bbox overlay; report tabs (rendered HTML + raw JSON) -> multi-turn chatbot reusing the same `image_id`). Talks to the orchestrator exclusively over HTTP. no direct access to any other service.

### Observability
- **Tracing:** every service calls `setup_tracing()` (OpenTelemetry SDK) and exports to **Jaeger** (`:16686` UI, OTLP gRPC on `:4317`). Spans cover routing, vision inference, spatial derivation, knowledge mapping, CoT reasoning, RAG retrieval, the consistency guard, and final report generation. So a slow or failed request can be traced end-to-end across containers.
- **Metrics:** every service exposes `/metrics` in Prometheus format (latency histograms + request counters labeled by organ/label/status), scraped by **Prometheus** (`:9090`) and visualized in a provisioned **Grafana** dashboard (`:3000`), including dedicated counters for OOD rejections and mapper-vs-CoT disagreement rate.

## Repository structure

```
Med_QA_Platform/
├── docker-compose.yml          # 9 services: ollama, router, vision, knowledge,
│                                #   orchestrator, ui, jaeger, prometheus, grafana
├── module_registry.yaml        # single source of truth: service URLs + enabled modalities
├── .env.example                # all configurable env vars (copy to .env)
│
├── services/
│   ├── router/                 # Layer 1 — modality classifier (EfficientNet-B0)
│   │   ├── main.py             #   FastAPI app: POST /route, /health, /metrics
│   │   └── model.py            #   ModalityRouter, OOD logic, hint resolution
│   │
│   ├── vision/                 # Layer 2 — segmentation + classification
│   │   ├── main.py             #   FastAPI app: /analyze/us_breast, /analyze/us_thyroid, /analyze/xray (stub)
│   │   ├── us_breast/          #   UNet_MTL trained on BUSI (3 classes)
│   │   │   ├── arch.py         #     Config, ConvBlock, UNet_MTL, UNet_Segmentation
│   │   │   ├── model.py        #     load_model(), run_inference(), bottleneck summary
│   │   │   └── postprocess.py  #     mask → bbox/area/centroid/quadrant
│   │   └── us_thyroid/         #   Same architecture, trained on TN3K (2 classes)
│   │
│   ├── knowledge/               # Layer 3 — rule-based clinical mapping (no ML)
│   │   ├── main.py             #   FastAPI app: POST /map
│   │   └── mapper.py           #   BI-RADS/TI-RADS/ICD-10/severity lookup tables
│   │
│   └── orchestrator/            # Layer 4 — LangGraph gateway
│       ├── main.py             #   FastAPI app: POST /analyze, POST /chat
│       ├── graph.py             #   LangGraph StateGraph + AsyncSequentialFallback,
│       │                       #     all node factories, prompt builders
│       ├── llm_client.py        #   BaseLLMClient + Ollama/Gemini/Mock implementations
│       ├── module_registry.py   #   loads & validates module_registry.yaml
│       ├── entrypoint.sh        #   auto-builds the FAISS index on container start
│       └── rag/
│           ├── faiss_store.py   #   FAISSStore: retrieve, retrieve_with_meta, rerank
│           ├── docs/            #   source clinical PDFs (BI-RADS, TI-RADS, ATA, ICD-10...)
│           └── vectordb/        #   built FAISS index + chunks.pkl + metadata.pkl
│
├── shared/                      # imported by every service
│   ├── schemas.py               #   Pydantic models — the contract between all layers
│   ├── image_validation.py      #   shared upload size/dimension checks
│   └── telemetry.py             #   OpenTelemetry setup helper
│
├── ui/
│   └── app.py                   # Gradio demo UI (upload, report tabs, chatbot)
│
├── scripts/
│   ├── build_vectordb.py        # offline: PDFs → chunks → FAISS index + metadata
│   ├── check_drift.py           # manual data-drift check (Deepchecks/Evidently)
│   └── eval_bus_cot.py          # offline pipeline evaluation vs. the BUS-CoT dataset
│
├── monitoring/
│   ├── prometheus.yml           # scrape configs for all 4 services
│   └── grafana/provisioning/    # pre-built datasource + dashboard JSON
│
├── models/checkpoints/          # router_effnet_b0.pth, mtl_effnet_fc_conv*.pt
├── data/{busi,tn3k}/            # local dataset mount points (gitkeep only)
├── requirements/                # one requirements.txt per service (slim images)
└── tests/                       # pytest suite (see Testing below)
```

## Quickstart

### Prerequisites
- Docker + Docker Compose
- (Optional) NVIDIA GPU + `nvidia-container-toolkit` for faster inference — CPU works fine for the demo, just slower
- Pretrained checkpoints already placed in `models/checkpoints/` (`router_effnet_b0.pth`, `mtl_effnet_fc_conv.pt`, `mtl_effnet_fc_conv_thyroid.pt`)

### 1. Configure environment

```bash
cp .env.example .env
```

At minimum, set `GRAFANA_PASSWORD` (Grafana refuses to start without it) and pick an LLM backend (see [Configuration](#configuration-env) below).

### 2. Start everything

```bash
docker compose up -d
```

This brings up: `ollama`, `router` (`:8001`), `vision` (`:8002`), `knowledge` (`:8003`), `orchestrator` (`:8000`), `ui` (`:7860`), `jaeger` (`:16686`), `prometheus` (`:9090`), `grafana` (`:3000`).

The orchestrator's entrypoint **automatically builds the FAISS RAG index** from the PDFs in `services/orchestrator/rag/docs/` on first start (and rebuilds it if a PDF is newer than the existing index). If no PDFs are present, the platform still works — every report just carries a `rag_disabled_warning`.

### 3. Pull a local LLM (if using Ollama)

```bash
docker compose exec ollama ollama pull phi4-mini
```

`phi4-mini` is the default (MIT-licensed, no clinical-use restriction, CPU-friendly). If throughput is insufficient on your hardware:

```bash
# in .env: OLLAMA_MODEL=qwen2.5:7b
docker compose exec ollama ollama pull qwen2.5:7b
```

### 4. Open the UI

Visit **http://localhost:7860**, upload a breast or thyroid ultrasound image, optionally pick a modality hint, ask a question, and click Analyze.

### 5. Or call the API directly

```bash
curl -X POST http://localhost:8000/analyze \
  -F "image=@/path/to/scan.png" \
  -F "question=Is this lesion benign or malignant?" \
  -F "organ_hint=breast"
```

## Configuration (.env)

| Variable | Default | Purpose |
|---|---|---|
| `ROUTER_PORT` / `VISION_PORT` / `KNOWLEDGE_PORT` / `ORCHESTRATOR_PORT` | `8001`/`8002`/`8003`/`8000` | Service ports |
| `BUSI_CHECKPOINT`, `THYROID_CHECKPOINT`, `ROUTER_CHECKPOINT` | `models/checkpoints/...` | Checkpoint paths mounted read-only into containers |
| `OOD_THRESHOLD` | `0.06` | Router confidence floor below which an image is rejected as out-of-distribution |
| `ALLOW_DEGRADED_ROUTER` | `false` | If `true`, allows the pipeline to proceed even when the router has no trained checkpoint (random weights) — **dev/demo only** |
| `CONFIDENCE_CALIBRATION_THRESHOLD` | `0.999` | Confidence level above which a calibration warning is attached to the report |
| `MODULE_REGISTRY_PATH` | `module_registry.yaml` | Path to the service/modality registry |
| `LLM_BACKEND` | `ollama` (compose default is `google` in `.env.example` — check before deploying) | `ollama` \| `google` \| `mock` |
| `OLLAMA_BASE_URL`, `OLLAMA_MODEL` | `http://ollama:11434`, `phi4-mini` | Self-hosted LLM backend |
| `GOOGLE_API_KEY`, `GOOGLE_MODEL` | _(empty)_ | Cloud LLM backend (Gemini) |
| `FAISS_INDEX_PATH`, `RAG_DOCS_DIR`, `RAG_VECTORDB_DIR` | `services/orchestrator/rag/...` | RAG index location + source PDFs |
| `CHAT_CONTEXT_TTL` | `3600` | Seconds before a cached `/analyze` context expires for `/chat` follow-ups |
| `GRAFANA_PASSWORD` | _(required, no default)_ | Grafana admin password — compose fails to start Grafana without it |

> Double-check `LLM_BACKEND` before deploying: the code default is `ollama`, but the checked-in `.env.example` sets it to `google` — pick whichever backend you've actually configured a key/model for.

## API Reference

### `POST /analyze` (orchestrator, `:8000`)

`multipart/form-data`:

| Field | Type | Notes |
|---|---|---|
| `image` | file | PNG/JPG, ≤ 15 MB, ≤ 8000px per side |
| `question` | string | Default: *"What are the findings in this ultrasound image?"* |
| `image_id` | string (optional) | Custom ID; auto-generated UUID if omitted |
| `modality_hint` / `organ_hint` | `"breast"` \| `"thyroid"` (optional) | Blended with the router's own prediction, not a hard override |

Returns `ReportOutput` (see `shared/schemas.py`): `tier_1_structured` (modality, organ, label, confidence, risk_category, severity, icd10_hint, bbox, area_cm2, aspect_ratio, circularity, hint_conflict, icd10_agreement, …), `tier_2_radiological_description`, `tier_3_diagnostic_suggestion`, `disclaimer`, `rag_sources` (`[{file, page}]`), `rag_disabled_warning`, `mapper_result`, `cot_result`, `consensus`, `icd10_agreement`.

Error responses:
- `422` — pipeline blocked (out-of-distribution image, or a degraded/untrained router with `ALLOW_DEGRADED_ROUTER=false`)
- `413` — image too large or dimensions exceed the limit
- `500` — internal pipeline error (check Jaeger traces / service logs)

### `POST /chat` (orchestrator, `:8000`)

```json
{ "image_id": "abc123", "message": "What follow-up interval do you recommend?", "history": [{"role": "user", "content": "..."}] }
```

Requires a prior `/analyze` call for the same `image_id` (context cached in-process, expires after `CHAT_CONTEXT_TTL` seconds). Does **not** re-run vision/router/knowledge — only the LLM is invoked, with the cached report + RAG context + conversation history.

### Internal service endpoints (not normally called directly)

| Service | Endpoint | Purpose |
|---|---|---|
| Router `:8001` | `POST /route` | image (+ optional hint) → `RoutingResult` |
| Vision `:8002` | `POST /analyze/us_breast`, `/analyze/us_thyroid` | image → `ModelOutput` (mask, label, bottleneck features) |
| Knowledge `:8003` | `POST /map` | `ModelOutput` fields → `KnowledgeMapped` + `SpatialDerived` |

All four services also expose `GET /health` and `GET /metrics`.

## UI

`ui/app.py` (Gradio, port `7860`) provides:
- Image upload + modality hint dropdown + Analyze button
- Annotated image with the lesion bounding box overlaid, color-coded by severity
- Tabbed report view: rendered HTML document vs. raw JSON
- A multi-turn chatbot panel wired to `/chat`, reusing the same analyzed image without re-upload

## Observability

| Tool | URL | What you'll see |
|---|---|---|
| Jaeger | http://localhost:16686 | End-to-end traces per request: route → vision → spatial → knowledge/cot/rag → consistency_guard → qa_agent, with latency per span |
| Prometheus | http://localhost:9090 | Raw metrics: `*_duration_seconds` histograms, `*_requests_total` counters by organ/label/status, `orchestrator_ood_rejections_total`, `orchestrator_consensus_false_total`, `orchestrator_icd10_disagreement_total` |
| Grafana | http://localhost:3000 | Pre-provisioned dashboard (`monitoring/grafana/provisioning/`) combining the above into latency/throughput/disagreement-rate panels |

## Testing

```bash
pip install -r requirements/orchestrator.txt -r requirements/vision.txt -r requirements/router.txt -r requirements/knowledge.txt
pytest tests/
```

Test coverage includes:
- `test_hint_resolution.py` — router/user-hint weighted resolution and conflict edge cases
- `test_knowledge_mapper.py` — organ-specific spatial postprocessing dispatch, invalid-organ errors
- `test_parallel_pipeline.py` — the LangGraph fan-out/fan-in contract: RAG node uses the question (not the label), CoT never sees the mapper's result before reasoning, merge short-circuits on branch errors, consensus/icd10_agreement edge cases, and that the `AsyncSequentialFallback` reproduces the exact same dependency ordering as the real graph
- `test_chat_endpoint.py` — context cache round-trip, TTL expiry, chat prompt construction
- `test_ui_report.py` — HTML report rendering, banner ordering, and HTML-escaping of LLM/model output (XSS hardening)


## Known limitations & roadmap

- **X-ray modality** (`/analyze/xray`) is a stub returning HTTP 501 — explicitly scoped as Phase 2 (target: NIH ChestX-ray14).
- **Chat context cache is in-process**, not Redis/DB-backed — `/chat` will not work correctly if the orchestrator is scaled to more than 1 replica without first swapping in a shared session store.
- **Pixel spacing is a placeholder** (`pixel_spacing_mm=0.1`) for both breast and thyroid — real-world area measurements (`area_cm2`) need calibration against the actual probe/DICOM metadata before being clinically meaningful.
- **Training data is small and narrow** (BUSI: 780 images; TN3K), so confidence scores are explicitly flagged as potentially uncalibrated rather than treated as ground truth probabilities — see the `confidence_calibration_note` mechanism.
- **Breast quadrant logic assumes a right breast** (a documented simplification in `postprocess.py`) — needs a laterality input to be correct for left-breast images.

## Disclaimer

This platform's LLM-facing layers are intentionally **text-only and never directly view the uploaded image** — the vision and routing models do all image analysis; the LLM (and the CoT reasoning agent) only reason over structured labels, measurements, and retrieved clinical guideline text. This means the report-writing step cannot be influenced by anything embedded directly in the image pixels.

Every generated report carries an explicit disclaimer: **this is AI-generated screening assistance, not a medical diagnosis, and all findings must be reviewed and confirmed by a qualified radiologist or physician.**