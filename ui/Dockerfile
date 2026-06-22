FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir gradio>=4.26.0 httpx>=0.27.0 Pillow>=10.2.0 numpy>=1.26.0

COPY ui/app.py ./ui/app.py

RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 7860

CMD ["python", "ui/app.py"]
