FROM python:3.11-slim

WORKDIR /app

# Install minimal system dependencies for numpy & sklearn
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . /app

# Default port for HuggingFace
ENV PORT=7860

# Run backend
CMD ["python", "app.py"]
