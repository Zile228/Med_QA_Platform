FROM python:3.11-slim

WORKDIR /app

# System deps for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to take advantage of Docker layer caching
COPY requirements/vision.txt requirements.txt
# Install the CPU-only torch build (much lighter than the default CUDA
# build from PyPI, avoids broken/interrupted downloads from oversized files)
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchvision
RUN pip install --no-cache-dir -r requirements.txt

# Copy shared schemas and service code
COPY shared/ ./shared/
COPY services/vision/ ./services/vision/

RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8002

CMD ["uvicorn", "services.vision.main:app", "--host", "0.0.0.0", "--port", "8002"]
