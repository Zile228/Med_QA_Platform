FROM python:3.11-slim

WORKDIR /app

# System deps cho OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements trước để tận dụng Docker layer cache
COPY requirements/vision.txt requirements.txt
# Cài torch bản CPU-only (nhẹ hơn nhiều so với bản CUDA mặc định từ PyPI,
# tránh lỗi tải hỏng/đứt giữa đường do file quá lớn)
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchvision
RUN pip install --no-cache-dir -r requirements.txt

# Copy shared schemas và service code
COPY shared/ ./shared/
COPY services/vision/ ./services/vision/

EXPOSE 8002

CMD ["uvicorn", "services.vision.main:app", "--host", "0.0.0.0", "--port", "8002"]
