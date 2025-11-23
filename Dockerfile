FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for numpy & sklearn
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Expose the Hugging Face default port
ENV PORT=7860

# Run the app with gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:7860", "app:app"]
