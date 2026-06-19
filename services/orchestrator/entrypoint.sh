#!/bin/sh
# services/orchestrator/entrypoint.sh
# =====================================
# Tự động build FAISS index lúc container start nếu:
#   - Có ít nhất 1 PDF trong RAG_DOCS_DIR, VÀ
#   - Index chưa tồn tại HOẶC PDF mới hơn index hiện có
#
# Nếu không có PDF nào, KHÔNG block container start — orchestrator vẫn
# chạy được, chỉ là rag_disabled_warning sẽ xuất hiện trong mọi report
# (set ở graph.py::make_report_node).

set -e

DOCS_DIR="${RAG_DOCS_DIR:-services/orchestrator/rag/docs}"
OUT_DIR="${RAG_VECTORDB_DIR:-services/orchestrator/rag/vectordb}"
INDEX_FILE="${OUT_DIR}/index.faiss"

echo "[entrypoint] Kiểm tra RAG index..."

PDF_COUNT=$(find "$DOCS_DIR" -name "*.pdf" 2>/dev/null | wc -l | tr -d ' ')

if [ "$PDF_COUNT" -eq 0 ]; then
    echo "[entrypoint] Không có PDF trong $DOCS_DIR — bỏ qua build RAG index."
    echo "[entrypoint] Report sẽ có rag_disabled_warning cho tới khi PDF được thêm vào và container restart."
else
    NEED_BUILD=0
    if [ ! -f "$INDEX_FILE" ]; then
        NEED_BUILD=1
        echo "[entrypoint] Index chưa tồn tại — sẽ build."
    else
        # Build lại nếu có PDF mới hơn index hiện có
        NEWEST_PDF=$(find "$DOCS_DIR" -name "*.pdf" -newer "$INDEX_FILE" 2>/dev/null | head -n 1)
        if [ -n "$NEWEST_PDF" ]; then
            NEED_BUILD=1
            echo "[entrypoint] Phát hiện PDF mới hơn index ($NEWEST_PDF) — sẽ build lại."
        fi
    fi

    if [ "$NEED_BUILD" -eq 1 ]; then
        echo "[entrypoint] Building RAG index từ $PDF_COUNT PDF(s) trong $DOCS_DIR..."
        python scripts/build_vectordb.py --docs_dir "$DOCS_DIR" --out_dir "$OUT_DIR"
    else
        echo "[entrypoint] Index đã up-to-date — bỏ qua build."
    fi
fi

echo "[entrypoint] Starting orchestrator..."
exec uvicorn services.orchestrator.main:app --host 0.0.0.0 --port 8000
