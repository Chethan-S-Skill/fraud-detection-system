# Use a lightweight python runtime
FROM python:3.9-slim

# Set environment configurations
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install build dependencies for compiling package extensions (e.g. SHAP C++ parts)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application files and folders
COPY app.py .
COPY src/ ./src/
COPY web/ ./web/

# Expose port 8000 for standard access
EXPOSE 8000

# Execute uvicorn server in production mode
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
