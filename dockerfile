# Use Python 3.12.5 as base image
FROM python:3.12.5-slim

# Set working directory
WORKDIR /app

# Install system dependencies including curl for healthchecks
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libmagic1 \
    tesseract-ocr \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Expose ports
EXPOSE 8000 443 80 20

# Run the server
CMD ["uvicorn", "core.api:app", "--host", "0.0.0.0", "--port", "443"]
