FROM python:3.13-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_SOURCE=local \
    MODEL_DIR=/app/models/api/p1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY proyecto_mibici ./proyecto_mibici
COPY models/api/p1 ./models/api/p1

EXPOSE 8000

CMD ["uvicorn", "proyecto_mibici.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
