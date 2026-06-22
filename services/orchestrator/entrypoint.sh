#!/bin/sh
# services/orchestrator/entrypoint.sh
# =====================================
# Automatically builds the FAISS index on container start if:
#   - There is at least 1 PDF in RAG_DOCS_DIR, AND
#   - The index does not exist OR a PDF is newer than the current index
#
# If there is no PDF, or the build itself fails (e.g. a PDF has no
# extractable text, a dependency is missing), this does NOT block container
# start -- the orchestrator still runs, it's just that rag_disabled_warning
# will appear in every report (set in graph.py::make_report_node). A failed
# build should never be fatal: a broken/empty PDF must not take down the
# whole service on every restart.

DOCS_DIR="${RAG_DOCS_DIR:-services/orchestrator/rag/docs}"
OUT_DIR="${RAG_VECTORDB_DIR:-services/orchestrator/rag/vectordb}"
INDEX_FILE="${OUT_DIR}/index.faiss"

echo "[entrypoint] Checking RAG index..."

PDF_COUNT=$(find "$DOCS_DIR" -name "*.pdf" 2>/dev/null | wc -l | tr -d ' ')

if [ "$PDF_COUNT" -eq 0 ]; then
    echo "[entrypoint] No PDF found in $DOCS_DIR -- skipping RAG index build."
    echo "[entrypoint] Reports will carry rag_disabled_warning until a PDF is added and the container restarts."
else
    NEED_BUILD=0
    if [ ! -f "$INDEX_FILE" ]; then
        NEED_BUILD=1
        echo "[entrypoint] Index does not exist -- will build."
    else
        # Rebuild if a PDF is newer than the current index
        NEWEST_PDF=$(find "$DOCS_DIR" -name "*.pdf" -newer "$INDEX_FILE" 2>/dev/null | head -n 1)
        if [ -n "$NEWEST_PDF" ]; then
            NEED_BUILD=1
            echo "[entrypoint] Found a PDF newer than the index ($NEWEST_PDF) -- will rebuild."
        fi
    fi

    if [ "$NEED_BUILD" -eq 1 ]; then
        echo "[entrypoint] Building RAG index from $PDF_COUNT PDF(s) in $DOCS_DIR..."
        if ! python scripts/build_vectordb.py --docs_dir "$DOCS_DIR" --out_dir "$OUT_DIR"; then
            echo "[entrypoint] WARNING: RAG index build failed -- continuing startup with RAG disabled."
            echo "[entrypoint] Check the PDF files in $DOCS_DIR and the build_vectordb.py logs above."
        fi
    else
        echo "[entrypoint] Index is already up-to-date -- skipping build."
    fi
fi

echo "[entrypoint] Starting orchestrator..."
exec uvicorn services.orchestrator.main:app --host 0.0.0.0 --port 8000
