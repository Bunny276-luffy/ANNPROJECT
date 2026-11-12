FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code and artifacts
COPY src/ ./src/
COPY configs/ ./configs/
COPY deployment/ ./deployment/
COPY reports/ ./reports/
COPY plots/ ./plots/
COPY frontend/ ./frontend/
COPY predict.py .

EXPOSE 8000

CMD ["uvicorn", "src.prediction:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]

