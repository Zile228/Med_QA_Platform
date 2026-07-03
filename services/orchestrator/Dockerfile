FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

COPY requirements/orchestrator.txt requirements.txt
# sentence-transformers pulls in torch; install the CPU-only build first so pip
# does not automatically download the default CUDA build (very heavy, prone to
# broken/interrupted downloads)
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchvision
RUN pip install --no-cache-dir -r requirements.txt

COPY shared/ ./shared/
COPY services/orchestrator/ ./services/orchestrator/
COPY scripts/ ./scripts/
COPY module_registry.yaml ./module_registry.yaml

RUN chmod +x services/orchestrator/entrypoint.sh

RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

ENTRYPOINT ["services/orchestrator/entrypoint.sh"]
