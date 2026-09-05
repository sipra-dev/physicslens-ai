FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        libpq5 \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir \
    torch==2.14.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir --retries 10 --timeout 120 -r requirements.txt

COPY . .

EXPOSE 8000
EXPOSE 8501

CMD ["uvicorn", "apps.api.main:app", "--host", "0.0.0.0", "--port", "8000"]