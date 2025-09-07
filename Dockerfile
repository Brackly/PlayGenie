# Use official Python runtime as base image
FROM python:3.10-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Set working directory in container
WORKDIR /app

# Install system dependencies if needed
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy uv.lock and pyproject.toml for dependency resolution
COPY pyproject.toml uv.lock ./

# Install dependencies using uv
RUN uv sync --frozen --no-install-project

# Copy the entire project
COPY . .

# Install the project itself
RUN uv sync --frozen

# Make sure the src directory is in Python path
ENV PYTHONPATH=/app/src:$PYTHONPATH

# Use uv run to ensure proper environment activation
CMD ["uv", "run", "python", "main.py", "--mode", "model_training"]