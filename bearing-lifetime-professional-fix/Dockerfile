FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY bearing_app/requirements.txt /app/bearing_app/requirements.txt
RUN pip install --no-cache-dir -r /app/bearing_app/requirements.txt

COPY bearing_app /app/bearing_app

EXPOSE 8000

CMD ["sh", "-c", "python -m uvicorn bearing_app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
