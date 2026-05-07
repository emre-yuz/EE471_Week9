# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Install system dependencies required for OpenCV and MediaPipe
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    libglvnd0 \
    libegl1 \
    libgomp1 \
    libgthread-2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements file and install Python dependencies
COPY container/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and model files
COPY *.py ./
COPY *.task ./
COPY *.tflite ./

# Copy sample images (optional, for testing)
COPY *.jpg *.png *.jpeg ./

# Create a non-root user for security
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Default command - can be overridden
CMD ["python", "POSE_ESTIMATION.py"]