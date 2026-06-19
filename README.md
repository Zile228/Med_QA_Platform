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

