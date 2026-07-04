```
                          ┌─────────────────────────────────────────────┐
                          │                   UI (Gradio)                │
                          │              ui/app.py — port 7860           │
                          └───────────────────┬───────────────────────────┘
                                              │ HTTP (httpx)
                                   POST /analyze, POST /chat
                                              ▼
   ┌───────────────────────────────────────────────────────────────────────────┐
   │                  ORCHESTRATOR (Layer 4) — port 8000                       │
   │                  services/orchestrator/main.py + graph.py                 │
   │   - Đọc module_registry.yaml để biết URL/endpoint từng service           │
   │   - Build LangGraph 1 lần ở lifespan(), giữ singleton _graph              │
   │   - /analyze: chạy toàn bộ pipeline (xem mục 2)                          │
   │   - /chat: dùng lại context đã cache theo image_id (KHÔNG gọi lại        │
   │     router/vision/knowledge, chỉ gọi LLM)                                │
   └──────────┬───────────────────┬───────────────────┬────────────────────────┘
              │ POST /route       │ POST /analyze/...  │ POST /map/spatial
              │                   │                     │ POST /map/knowledge
              ▼                   ▼                     ▼
   ┌─────────────────┐  ┌──────────────────────┐  ┌──────────────────────┐
   │ ROUTER (L1)      │  │ VISION (L2)           │  │ KNOWLEDGE (L3)        │
   │ port 8001        │  │ port 8002             │  │ port 8003             │
   │ EfficientNet-B0   │  │ UNet_MTL              │  │ Rule-based:           │
   │ modality classify │  │ EfficientNet-B4       │  │ - mapper.py           │
   │ + OOD detection    │  │ encoder + UNet decoder│  │ - postprocess (gọi    │
   │                   │  │ seg + classify (3 lớp)│  │   lại code của Vision)│
   └─────────────────┘  └──────────────────────┘  └──────────────────────┘
              (Không có ML — chỉ thuần rule + hardcoded clinical knowledge)

   Orchestrator còn gọi trực tiếp (in-process, không qua HTTP):
     - llm_client.py        -> LLM (Ollama / Gemini / OpenAI / Remote vLLM / Local HF / Mock)
     - rag/faiss_store.py    -> FAISS vector store (RAG)
     - birads_describer.py   -> LLM vision (BI-RADS/TI-RADS observation)
     - visual_interpreter.py -> dịch numeric feature -> clinical flag (text)
```
```
                         route
                           │
                           ▼
                         vision
                           │
                           ▼
                         spatial
              ┌────────────┼────────────┐
              ▼            ▼            ▼
          knowledge   rag_retrieve  birads_description
              │            └─────┬──────┘
              │                  ▼
              │            cot_reasoning
              └────────┬─────────┘
                       ▼
                     merge
                       │
                       ▼
              consistency_guard      (chạy RAG lần 2 — "second retrieve")
                       │
                       ▼
                    qa_agent          (gọi LLM sinh Tier 2 + Tier 3)
                       │
                       ▼
                      END
```
```
        CNN classify ──► Knowledge mapper ──► severity_level (mapper)
                                                      │
                                                      ▼  so sánh
        Spatial + RAG + (BI-RADS) ──► CoT LLM ──► severity_level (CoT)
                                                      │
                                          consensus / icd10_agreement /
                                          label_agreement / hard_conflict
```
