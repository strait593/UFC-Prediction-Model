FROM python:3.13-slim

WORKDIR /app

# Install runtime dependencies
RUN pip install --no-cache-dir \
    gender-guesser>=0.4.0 \
    fastapi>=0.115.0 \
    numpy>=2.5.3 \
    pandas>=3.0.5 \
    scikit-learn>=1.9.1 \
    uvicorn[standard]>=0.30.0 \
    requests>=2.31.0

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

# Copy application code
COPY src src
COPY scripts scripts
COPY data data

# Create non-root user
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

USER appuser

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/api/health')" || exit 1

# Default command: start API server
CMD ["uvicorn", "ufc_pred_model.api:app", "--host", "0.0.0.0", "--port", "8000"]
