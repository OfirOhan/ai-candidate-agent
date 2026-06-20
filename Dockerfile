# ──────────────────────────────────────────────────────────────
# Stage 1 — Install dependencies (cached layer)
# ──────────────────────────────────────────────────────────────
FROM python:3.10-slim AS builder

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy only dependency files first (cache-friendly)
COPY pyproject.toml uv.lock ./

# Install production dependencies only (no dev tools like ruff/pytest)
RUN uv sync --frozen --no-dev --no-install-project

# ──────────────────────────────────────────────────────────────
# Stage 2 — Final runtime image
# ──────────────────────────────────────────────────────────────
FROM python:3.10-slim

# System dependencies for document parsing (Tesseract OCR, Poppler)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        tesseract-ocr \
        libtesseract-dev \
        poppler-utils && \
    rm -rf /var/lib/apt/lists/*

# Install uv (needed for `uv run`)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy the venv from the builder stage
COPY --from=builder /app/.venv /app/.venv

# Copy project files
COPY pyproject.toml uv.lock ./
COPY main.py ./
COPY .streamlit ./.streamlit
COPY agent ./agent
COPY rag ./rag
COPY store ./store
COPY pages ./pages
COPY evaluation ./evaluation
COPY images ./images

# Create uploads directory
RUN mkdir -p uploads

# Expose Streamlit port (matches .streamlit/config.toml)
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:3000/_stcore/health')" || exit 1

# Run the app
CMD ["uv", "run", "streamlit", "run", "main.py", "--server.address", "0.0.0.0"]
