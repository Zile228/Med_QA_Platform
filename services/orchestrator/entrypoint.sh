#!/bin/sh
# services/orchestrator/entrypoint.sh
#
# Does not build the FAISS index. Building involves PDF-to-OCR work that
# competes for CPU with other services starting at the same time, so it
# must be run separately before `docker compose up`, e.g.:
#   docker compose run --rm build_vectordb

DOCS_DIR="${RAG_DOCS_DIR:-services/orchestrator/rag/docs}"
OUT_DIR="${RAG_VECTORDB_DIR:-services/orchestrator/rag/vectordb}"
INDEX_FILE="${OUT_DIR}/index.faiss"

echo "[entrypoint] Checking RAG index..."

if [ ! -f "$INDEX_FILE" ]; then
    echo "[entrypoint] WARNING: no index at $INDEX_FILE."
    echo "[entrypoint] Run 'docker compose run --rm build_vectordb' first, then restart."
    echo "[entrypoint] Reports will carry rag_disabled_warning until then."
else
    PDF_COUNT=$(find "$DOCS_DIR" -name "*.pdf" 2>/dev/null | wc -l | tr -d ' ')
    NEWEST_PDF=$(find "$DOCS_DIR" -name "*.pdf" -newer "$INDEX_FILE" 2>/dev/null | head -n 1)
    if [ -n "$NEWEST_PDF" ]; then
        echo "[entrypoint] WARNING: $NEWEST_PDF is newer than the index."
        echo "[entrypoint] Run 'docker compose run --rm build_vectordb' to pick up the change."
    else
        echo "[entrypoint] Index found ($PDF_COUNT PDF(s) tracked) -- using it as is."
    fi
fi

echo "[entrypoint] Starting orchestrator..."
exec uvicorn services.orchestrator.main:app --host 0.0.0.0 --port 8000
