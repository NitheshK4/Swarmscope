FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

# Expose ports: 8000 for FastAPI API, 8501 for Streamlit Dashboard
EXPOSE 8000
EXPOSE 8501

# Entrypoint script to start both services
RUN echo '#!/bin/bash\n\
uvicorn sandbox.api:app --host 0.0.0.0 --port 8000 &\n\
streamlit run sandbox/dashboard/app.py --server.port 8501 --server.address 0.0.0.0\n\
' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

CMD ["/app/entrypoint.sh"]
