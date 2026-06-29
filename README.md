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
```
Giai đoạn 1 (offline, không cần Docker/LLM)
  1a. eval_router.py   — Router CNN (modality + OOD)
  1b. eval_vision.py   — Vision CNN (segmentation + classification)
        │
        ▼
Giai đoạn 2 (offline, cần LLM API key)
  eval_cot.py với LLM_BACKEND=google/openai  — CoT baseline
        │
        ├──► Giai đoạn 2.5 (tuỳ chọn — fine-tune Qwen làm LLM rẻ hơn)
        │      2.5a. generate_finetune_data.py  (Gemini/OpenAI làm "teacher")
        │      2.5b. eval_cot.py với Qwen BASE (LLM_BACKEND=remote, trước train) — baseline so sánh
        │      2.5c. finetune_cot_colab.ipynb   (LoRA fine-tune trên Colab/Kaggle)
        │      2.5d. eval_cot.py với Qwen FINE-TUNED (sau train, trên cùng test set)
        │      2.5e. So sánh 3 cột: Gemini / Qwen-base / Qwen-finetuned
        │
        ▼
Giai đoạn 3 (độc lập với 2/2.5 — có thể chạy song song)
  3a. build_vectordb.py            — build FAISS index từ PDF
  3b. generate_ragas_testset.py    — sinh testset Q&A tự động từ tài liệu
  3c. eval_rag.py + eval_ragas.py (--mode retrieval)  — đánh giá retrieval thuần
  3d. run_pipeline_batch.py + eval_ragas.py (--mode pipeline)  — đánh giá faithfulness trên full pipeline (cần Docker)
        │
        ▼
Giai đoạn 4 (cần Docker, ngay sau 3d trong cùng phiên — TTL cache)
  run_pipeline_batch.py + eval_qa.py   — đánh giá /chat (chatbot) bằng G-Eval
```