# ─────────────────────────────────────────────────────────────
#  socratOT — Dockerfile
#  Multi-stage build: builder → runtime
#  Supports: linux/amd64, linux/arm64 (Apple Silicon via Rosetta)
# ─────────────────────────────────────────────────────────────

# ── Stage 1: Builder ──────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# System deps needed to compile some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps into an isolated prefix (copied into runtime later).
# CPU-only torch is installed first from PyTorch's index so the heavy
# CUDA build never gets pulled in transitively by sentence-transformers.
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --no-cache-dir --prefix=/install \
        torch==2.4.1 torchvision==0.19.1 \
        --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: Runtime ──────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# curl is needed for the container healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --shell /bin/bash socratot

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

# Installed packages from the builder stage
COPY --from=builder /install /usr/local

# Application code (data/logs/evaluation are bind-mounted at runtime — see
# docker-compose.yml — and are excluded from the build via .dockerignore)
COPY --chown=socratot:socratot . .

# Pre-create writable mount points + Streamlit config dir
RUN mkdir -p data/processed logs evaluation/results /home/socratot/.streamlit \
    && cp -r .streamlit/. /home/socratot/.streamlit/ \
    && chown -R socratot:socratot data logs evaluation /home/socratot/.streamlit

USER socratot

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "src/app/main.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
