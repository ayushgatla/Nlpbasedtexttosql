# ------------------------------------------------------------------------------
# NeuroSQL Text-to-SQL Studio Container
# ------------------------------------------------------------------------------
FROM python:3.10-slim

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency specifications first to leverage Docker layer caching
COPY requirements.txt .

# Install PyTorch CPU-optimized wheel first (avoids 2.5GB CUDA bloat for CPU inference),
# then install remaining project dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download required NLTK tokenizers so container starts offline without runtime delay
RUN python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)"

# Create non-root application user for container security
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/data/custom_dbs && \
    chown -R appuser:appuser /app

# Copy application source code and schema metadata
COPY --chown=appuser:appuser app.py .
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser spider/ ./spider/
COPY --chown=appuser:appuser data/ ./data/

# Copy fine-tuned relational Seq2Seq model checkpoint
COPY --chown=appuser:appuser best_model_v2_relational/ ./best_model_v2_relational/

USER appuser

# Expose Streamlit default port
EXPOSE 8501

# Healthcheck to verify Streamlit server responsiveness
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Launch Streamlit Studio
CMD ["streamlit", "run", "app.py"]
