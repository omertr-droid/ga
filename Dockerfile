FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
COPY . /app
RUN pip install --no-cache-dir -r requirements.txt
ENV PYTHONPATH=/app
EXPOSE 10000
CMD ["sh", "-c", "uvicorn clinic.api.app:app --host 0.0.0.0 --port ${PORT:-10000}"]
