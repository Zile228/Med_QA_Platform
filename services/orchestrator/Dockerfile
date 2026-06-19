FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements/orchestrator.txt requirements.txt
# sentence-transformers kéo theo torch; cài bản CPU-only trước để pip không
# tự động tải bản CUDA mặc định (rất nặng, dễ lỗi tải hỏng/đứt giữa đường)
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchvision
RUN pip install --no-cache-dir -r requirements.txt

COPY shared/ ./shared/
COPY services/orchestrator/ ./services/orchestrator/
COPY scripts/ ./scripts/
COPY module_registry.yaml ./module_registry.yaml

RUN chmod +x services/orchestrator/entrypoint.sh

EXPOSE 8000

ENTRYPOINT ["services/orchestrator/entrypoint.sh"]
