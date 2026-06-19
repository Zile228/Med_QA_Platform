# Med_QA_Platform
Note: Add bottleneck feature làm context. Knowledge đang rule-based 1 chiều. Thêm bước query reformation trước khi embed (ghép organ + top_label vào query sau khi vision xong). Metadata filtering, re-ranking, map ngược lại citation. Chat / context cache 

```
                ┌──────────┐
   image ──────▶│  route  │ (Layer 1 — Router)
                └────┬─────┘
                     │
         ┌───────────┴────────────┐
         ▼                        ▼
   ┌──────────┐             ┌─────────────┐
   │  vision  │             │ rag_retrieve│   ← chạy SONG SONG với vision
   └────┬─────┘             └──────┬──────┘
        ▼                         │
   ┌──────────┐                   │
   │knowledge │                   │
   └────┬─────┘                   │
        └───────────┬─────────────┘
                     ▼
               ┌──────────┐
               │  merge   │  ← barrier: đợi CẢ HAI nhánh xong
               └────┬─────┘
                     ▼
               ┌──────────┐
               │ qa_agent │  ← gọi LLM sinh report 3 tầng
               └────┬─────┘
                     ▼
                   END
```

**Giải thích từng bước:**

1. **`route`** — gọi `POST router:8001/route`. Router dùng EfficientNet-B0 phân loại ảnh là `us_breast` hay `us_thyroid`. Nếu confidence dưới ngưỡng (`OOD_THRESHOLD`, default 0.6) → ảnh bị reject (out-of-distribution), pipeline dừng với lỗi 422.
2. **Fan-out** — sau khi route xong, hai nhánh chạy đồng thời:
   - **`vision`**: gọi `POST vision:8002/analyze/{modality}` để segment + classify tổn thương.
   - **`rag_retrieve`**: tra cứu FAISS vector DB (local, không qua HTTP) bằng câu hỏi của user, lấy top-3 đoạn văn bản liên quan từ các PDF lâm sàng (BI-RADS, TI-RADS, ICD-10...).
   - Việc chạy song song là **cố ý** — `rag_retrieve` không cần chờ kết quả vision vì nó query trực tiếp câu hỏi gốc của user, không phụ thuộc nhãn phân loại.
3. **`knowledge`** (tiếp nối `vision`) — gọi `POST knowledge:8003/map`, nhận classification result + mask → trả về BI-RADS/TI-RADS, severity, ICD-10 hint, và các đặc trưng không gian (vị trí, diện tích, aspect ratio, circularity...) tính từ segmentation mask.
4. **`merge`** — là một **barrier thực sự**: đợi cả nhánh `knowledge` và nhánh `rag_retrieve` hoàn thành trước khi đi tiếp. Đây là điểm kỹ thuật quan trọng: nếu khai báo 2 edge riêng lẻ (`knowledge→merge` và `rag_retrieve→merge`) trong LangGraph, node `merge` sẽ bị kích hoạt ngay khi nhánh nhanh nhất xong (thường là RAG, vì chỉ là lookup local) — gây lỗi vì knowledge chưa có. Code dùng `add_edge(["knowledge", "rag_retrieve"], "merge")` (1 edge với nhiều source) để LangGraph chờ đúng cả hai.
5. **`qa_agent`** — build prompt từ toàn bộ dữ liệu đã thu thập (classification, spatial features, RAG context, câu hỏi user) → gọi LLM (Ollama hoặc Google Gemini) → parse response thành **Tier 2** (mô tả radiological) và **Tier 3** (gợi ý chẩn đoán + follow-up).

Nếu `langgraph` chưa được cài, hệ thống tự fallback sang `AsyncSequentialFallback` — mô phỏng lại đúng hành vi fan-out/fan-in bằng `asyncio.gather`, đảm bảo behavior nhất quán dù môi trường có/không có LangGraph.



- **Không phải thiết bị y tế** : mọi report đều có disclaimer bắt buộc, không dùng để chẩn đoán cuối cùng.
- **Coverage hạn chế**: model breast chỉ biết 3 lớp từ BUSI (780 ảnh); confidence cao bất thường (≥99.9%) có thể là dấu hiệu overfitting, không phải độ tin cậy thật.
- **Chat cache không scale**: `/chat` dùng in-memory dict, chỉ chạy đúng với 1 replica orchestrator. Cần thay bằng Redis/DB cho production multi-instance.
- **X-Ray chưa implement**: endpoint `/analyze/xray` luôn trả `501`, đã có placeholder trong `module_registry.yaml` (`enabled: false`) để thêm modality mới trong tương lai mà không phải sửa schema.
- **Thêm modality mới**: về nguyên tắc chỉ cần: (1) thêm entry vào `module_registry.yaml`, (2) thêm 1 cặp model+postprocess trong `services/vision/`, (3) thêm bảng tra cứu vào `services/knowledge/mapper.py` không cần sửa `shared/schemas.py` hay orchestrator graph.