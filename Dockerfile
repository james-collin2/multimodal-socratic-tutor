# ─────────────────────────────────────────────────────────────
#  socratOT — Dockerfile
#  Multi-stage build: builder → runtime
#  Supports: linux/amd64, linux/arm64 (Apple Silicon via Rosetta)
# ─────────────────────────────────────────────────────────────

# ── Stage 1: Builder ──────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install system deps needed to compile some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Install Python dependencies into /build/wheels
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Install CPU-only torch (needed by sentence-transformers / RAGAS deps)
RUN pip install --no-cache-dir --prefix=/install \
    torch==2.4.1 torchvision==0.19.1 \
    --index-url https://download.pytorch.org/whl/cpu


# ── Stage 2: Runtime ──────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Non-root user for security
RUN useradd --create-home --shell /bin/bash socratot

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY --chown=socratot:socratot . .

# Create required directories
RUN mkdir -p \
    data/processed/chroma_db \
    data/processed/faiss_index \
    data/images \
    data/models \
    logs \
    evaluation/results \
    && chown -R socratot:socratot data logs evaluation

# Streamlit config
RUN mkdir -p /home/socratot/.streamlit
COPY --chown=socratot:socratot .streamlit/ /home/socratot/.streamlit/ 2>/dev/null || true

USER socratot

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

EXPOSE 8501

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

CMD ["streamlit", "run", "app/main.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
