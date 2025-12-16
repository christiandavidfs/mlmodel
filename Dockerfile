FROM python:3.11-slim

# Keep buffering off for real-time logs
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system deps needed to build some Python packages
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential git curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . /app

# Ensure data directory exists and is writable
RUN mkdir -p /app/data && chmod -R a+rw /app/data

# Create a non-root user and switch ownership of /app
RUN useradd -m -s /bin/bash appuser && chown -R appuser:appuser /app /app/data

# Switch to non-root user
USER appuser

# Default entrypoint runs the CLI
ENTRYPOINT ["python", "main_interface.py"]
# When running container without args, show help
CMD ["--help"]
