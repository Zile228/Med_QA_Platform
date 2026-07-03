FROM python:3.11-slim

WORKDIR /app

COPY requirements/ui.txt ./requirements/ui.txt
RUN pip install --no-cache-dir -r requirements/ui.txt

COPY ui/app.py ./ui/app.py

RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 7860

CMD ["python", "ui/app.py"]
