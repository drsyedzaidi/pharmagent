# PharmAgent backend — multi-stage, non-root, with a healthcheck.
# Build:  docker build -t pharmagent-backend ./backend
# Run:    docker run -p 8000:8000 -v pharmagent-data:/data pharmagent-backend

# ── builder: install pinned deps into a venv ─────────────────────────────────
FROM python:3.13-slim AS builder
WORKDIR /build
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── runtime: copy the venv + app, drop privileges ───────────────────────────
FROM python:3.13-slim AS runtime
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PHARMAGENT_DATA_DIR=/data \
    PHARMAGENT_DB_PATH=/data/pharmagent.db
COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY app ./app
COPY sample_data ./sample_data

# Non-root user owning the writable data volume.
RUN useradd --create-home --uid 10001 pharma && \
    mkdir -p /data && chown -R pharma:pharma /data /app
USER pharma
VOLUME ["/data"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=2).status==200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
