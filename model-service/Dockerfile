FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    REVIEWS_DATA_DIR=/data \
    BANKI_RU_DATASET=/data/banki_ru_full.csv \
    SRAVNI_RU_DATASET=/data/sravni_ru_full.csv

WORKDIR /app/hack-LCT

COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /data/processed/reports

EXPOSE 8002

CMD ["uvicorn", "service.api:app", "--host", "0.0.0.0", "--port", "8002"]
