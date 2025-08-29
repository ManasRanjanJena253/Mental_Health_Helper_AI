# Use Python 3.12 for compatibility with Torch, Whisper, etc.
FROM python:3.12.11

# Set working directory inside the container
WORKDIR /app

# Install system dependencies (for Whisper, audio, etc.)
RUN apt-get update && apt-get install -y \
    git-lfs \
    ffmpeg \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# Enable Git LFS if needed
RUN git lfs install

# Copy entire project into container (keeps backend folder)
COPY . .

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel
RUN pip install langchain-chroma
RUN pip install --no-cache-dir -r backend/requirements.txt

# Expose port for Render
EXPOSE 10000

# Start FastAPI app
CMD ["uvicorn", "backend.main.apis:app", "--host", "0.0.0.0", "--port", "10000"]
