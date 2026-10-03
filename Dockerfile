FROM python:3.13-slim

# No application defaults baked into the image: all configuration
# (including BRIDGE_PORT) is injected at runtime via .env / env_file.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY audio-bridge.py .

# Fails (unhealthy) when BRIDGE_PORT is unset or the server is not accepting connections
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import os, socket; s = socket.create_connection(('127.0.0.1', int(os.environ['BRIDGE_PORT'])), timeout=4); s.close()" || exit 1

CMD ["sh", "-c", "exec python audio-bridge.py"]
