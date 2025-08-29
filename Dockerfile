FROM python:3.12-slim

WORKDIR /app

# Install system dependencies (for Whisper, TTS, etc.)
RUN apt-get update && apt-get install -y \
    git-lfs \
    ffmpeg \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# Copy backend code
COPY backend/ .

# Upgrade pip + install dependencies
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# Expose port for Render
EXPOSE 10000

# Start FastAPI (change apis:app to your actual entry point)
CMD ["uvicorn", "main.apis:app", "--host", "0.0.0.0", "--port", "10000"]
