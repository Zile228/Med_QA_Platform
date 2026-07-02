FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements/knowledge.txt ./requirements/knowledge.txt
RUN pip install --no-cache-dir -r requirements/knowledge.txt

COPY shared/ ./shared/
COPY services/__init__.py ./services/__init__.py
COPY services/knowledge/ ./services/knowledge/
# Copy postprocess.py from vision for both organs -- knowledge only needs spatial features, NOT the model
COPY services/vision/__init__.py ./services/vision/__init__.py
COPY services/vision/us_breast/postprocess.py ./services/vision/us_breast/postprocess.py
COPY services/vision/us_breast/__init__knowledge.py ./services/vision/us_breast/__init__.py
COPY services/vision/us_thyroid/postprocess.py ./services/vision/us_thyroid/postprocess.py
COPY services/vision/us_thyroid/__init__knowledge.py ./services/vision/us_thyroid/__init__.py

RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8003

CMD ["uvicorn", "services.knowledge.main:app", "--host", "0.0.0.0", "--port", "8003"]
