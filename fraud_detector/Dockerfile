FROM python:3.12-slim

WORKDIR /app

RUN mkdir -p /app/logs && chmod -R 777 /app/logs

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY src ./src
COPY models ./models

ENV PYTHONUNBUFFERED=1 MODEL_DIR=/app/models LOG_DIR=/app/logs

CMD ["python", "./app/app.py"]
