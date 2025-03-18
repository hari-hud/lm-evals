FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml poetry.lock README.md ./
COPY lm_evals ./lm_evals
COPY examples ./examples

# Install Python dependencies
RUN pip install poetry && \
    poetry config virtualenvs.create false && \
    poetry install --no-dev

# Set environment variables
ENV PYTHONPATH=/app

# Default command to run evaluations
ENTRYPOINT ["python", "examples/run_judge_eval.py"] 
