# Use Python 3.11 (IMPORTANT for TensorFlow compatibility)
FROM python:3.11-slim

# Prevent Python from buffering stdout/stderr
ENV PYTHONUNBUFFERED=1

# Install system dependencies needed for OpenCV & MediaPipe
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first (better caching)
COPY requirements.txt .

# Upgrade pip
RUN pip install --upgrade pip

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire project
COPY . .

# Expose port (Hugging Face uses 7860 by default)
EXPOSE 7860

# Start FastAPI
CMD ["uvicorn", "predict_api_v2_fixed_almost_done:app", "--host", "0.0.0.0", "--port", "7860"]
