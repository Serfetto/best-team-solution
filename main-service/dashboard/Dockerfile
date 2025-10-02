FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app/main_service

COPY requirements.txt ./dashboard/requirements.txt
RUN pip install --no-cache-dir -r ./dashboard/requirements.txt

COPY . ./

RUN mkdir -p /app/main_service/hack-LCT/data/processed

EXPOSE 8000

CMD ["python", "dashboard/app.py"]
