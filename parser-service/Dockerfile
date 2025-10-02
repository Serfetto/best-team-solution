FROM mcr.microsoft.com/playwright/python:v1.49.0-jammy

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app/parser_service

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY parsing_service ./parsing_service
COPY data ./data

EXPOSE 8001

CMD ["uvicorn", "parsing_service.app:app", "--host", "0.0.0.0", "--port", "8001"]
