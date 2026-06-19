FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements/knowledge.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY shared/ ./shared/
COPY services/__init__.py ./services/__init__.py
COPY services/knowledge/ ./services/knowledge/
# Copy postprocess.py từ vision cho cả 2 organ — knowledge chỉ cần spatial features, KHÔNG cần model
COPY services/vision/__init__.py ./services/vision/__init__.py
COPY services/vision/us_breast/postprocess.py ./services/vision/us_breast/postprocess.py
COPY services/vision/us_breast/__init__knowledge.py ./services/vision/us_breast/__init__.py
COPY services/vision/us_thyroid/postprocess.py ./services/vision/us_thyroid/postprocess.py
COPY services/vision/us_thyroid/__init__knowledge.py ./services/vision/us_thyroid/__init__.py

EXPOSE 8003

CMD ["uvicorn", "services.knowledge.main:app", "--host", "0.0.0.0", "--port", "8003"]
