FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements/router.txt requirements.txt
# Cài torch bản CPU-only (nhẹ hơn nhiều so với bản CUDA mặc định từ PyPI,
# tránh lỗi tải hỏng/đứt giữa đường do file quá lớn)
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchvision
RUN pip install --no-cache-dir -r requirements.txt

COPY shared/ ./shared/
COPY services/router/ ./services/router/

RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8001

CMD ["uvicorn", "services.router.main:app", "--host", "0.0.0.0", "--port", "8001"]
