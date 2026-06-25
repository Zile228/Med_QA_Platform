Ảnh Siêu Âm
                                       │
                                       ▼
    ┌──────────────────┐
    │ [Router Service] │ ◄─── Chạy EfficientNet-B0
    │ "đây là ảnh gì?" │ ────► Phân loại: us_breast / us_thyroid / ood (hủy bỏ)
    └──────────────────┘
                                       │
                                       ▼
  ┌──────────────────┐
  │ [Vision Service] │ ◄─── UNet_MTL (EfficientNet-B4 encoder)
  │                  │   ├─► Segmentation mask ◄── decoder branch
  │                  │   ├─► Classification label ◄── cls_head từ bottleneck
  │                  │   ├─► Spatial features ◄── tính aspect ratio, circularity từ mask
  │                  │   │
  │                  │   │   [Đã tạm tắt để hiệu chuẩn thêm]:
  │                  │   ├─► Bottleneck features (raw 448x7x7)
  │                  │   ├─► Grad-CAM heatmap (XAI)
  │                  │   ├─► Texture features (so sánh trong/ngoài mask)
  │                  │   └─► Uncertainty (MC-Dropout)
  └──────────────────┘
                                       │
                     ┌─────────────────┴─────────────────┐
                     ▼                                   ▼
    ┌─────────────────────────────────┐ ┌─────────────────────────────────┐
    │        [Knowledge Node]         │ │        [RAG Giai Đoạn 1]        │
    │   Lấy thông tin đo đạc ảnh học  │ │   Tìm kiếm sơ bộ trên vector DB │
    │   từ nhánh [Vision Service]     │ │   theo từ khóa "Modality + Organ"  │
    └────────────────┬────────────────┘ └────────────────┬────────────────┘
                     │                                   │
                     │                                   ▼
                     │                  ┌─────────────────────────────────┐
                     │                  │      [CoT Reasoning Node]       │
                     │                  │   LLM độc lập suy luận chỉ từ   │
                     │                  │   thông số hình học & RAG GĐ 1  │
                     │                  │   (Tuyệt đối giấu nhãn của CNN) │
                     │                  └────────────────┬────────────────┘
                     │                                   │
                     └─────────────────┬─────────────────┘
                                       ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │                         [Consistency Guard]                         │
    │   - Thực hiện RAG Giai Đoạn 2 (Rerank lấy top 3 chunks tối ưu nhất) │
    │   - So khớp kết quả của [CNN + Rule] vs [CoT LLM]                   │
    │   - Kiểm tra: consensus, label_agreement, hard_conflict            │
    └──────────────────────────────────┬──────────────────────────────────┘
                                       │
                                       ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │                           [QA Agent (LLM)]                          │
    │   Tổng hợp dữ liệu thành báo cáo 3 tầng (Tier 1, 2, 3)              │
    │   * Đưa ra cảnh báo đỏ nghiêm trọng nếu phát hiện "Hard Conflict"   │
    └─────────────────────────────────────────────────────────────────────┘

[BẮT ĐẦU TRUY VẤN RAG]
                                  │
                                  ▼
┌───────────────────────────────────────────────────────────────────┐
│ GIAI ĐOẠN 1 (Sau khi xác định được bộ phận siêu âm)     │
│                                                                   │
│ - Query: "{modality} {organ}" (VD: "ultrasound breast")            │
│ - Thực hiện: Quét toàn bộ vector DB (FAISS Store) lọc theo Organ. │
│ - Kết quả: Lấy ra TOP 100 CHUNKS thô đưa vào State.               │
└─────────────────────────────────┬─────────────────────────────────┘
                                  │
                                  ▼
┌───────────────────────────────────────────────────────────────────┐
│ GIAI ĐOẠN 2 (Sau khi có nhãn CNN và mã ICD-10 của Rule) │
│                                                                   │
│ - Enriched Query: "{top_label} {organ} ultrasound findings {icd10}"│
│   (VD: "malignant breast ultrasound findings C50.9")              │
│                                                                   │
│ - Thực hiện:                                                      │
│   1. Tìm thêm TOP 5 CHUNKS mới nhất từ DB bằng Enriched Query.    │
│   2. Gộp 5 chunks mới này với 100 chunks thô ở Giai đoạn 1.       │
│   3. Loại bỏ các chunk trùng lặp nội dung.                        │
│   4. Chạy mô hình Reranker để tái xếp hạng dựa trên Enriched Query│
│                                                                   │
│ - Kết quả: Chọn ra TOP 3 CHUNKS xuất sắc nhất nạp vào QA Agent.  │
└───────────────────────────────────────────────────────────────────┘