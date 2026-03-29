
# Builds wheels in a throw-away layer so the final image stays lean.
FROM python:3.11-slim AS builder

WORKDIR /build

# System deps needed to compile some Python packages (e.g. bcrypt, cryptography)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libffi-dev \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt

 \
    runtime image
FROM python:3.11-slim AS runtime

# ffmpeg is required by Whisper for audio decoding
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install pre-built wheels from stage 1 (no compilation in runtime stage)
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir --no-index --find-links=/wheels /wheels/*

# Copy application source
COPY backend/main/ /app/

# Copy ChromaDB vector store directories (pre-built from index_pdf runs)
# If these are large, consider mounting them as a volume instead.
COPY Mental_Health_Remedies/         /app/Mental_Health_Remedies/
COPY Mental_Health_Taboos_in_India/  /app/Mental_Health_Taboos_in_India/

# Per-user conversation memory — always mount as a named volume so data
# persists across container restarts (see docker-compose.yml)
RUN mkdir -p /app/chroma

# Non-root user for security
RUN adduser --disabled-password --gecos "" appuser \
 && chown -R appuser:appuser /app
USER appuser

EXPOSE 8001

# Use multiple uvicorn workers for concurrency.
# Adjust --workers based on available CPU cores (2×cores + 1 is a common rule).
CMD ["uvicorn", "apis:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "2"]