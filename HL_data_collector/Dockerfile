# Lightweight Python base
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System deps (kept minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
  && rm -rf /var/lib/apt/lists/*

# Python deps – hyperliquid relies on websockets, etc.
RUN pip install --no-cache-dir \
    hyperliquid-python-sdk \
    websockets

# Copy your collector and wrapper
COPY hyperliquid_data_collector.py /app/hyperliquid_data_collector.py
COPY run_collector.py            /app/run_collector.py

# Default command (also set in compose)
CMD ["python", "run_collector.py"]

