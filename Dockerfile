# syntax=docker/dockerfile:1

# =====================================================================
# STAGE 1 — "builder": install Python dependencies into a clean venv.
# Doing this in its own stage means the final image doesn't carry build
# tools (gcc, headers, pip's cache) that we only needed to COMPILE
# dependencies, not to run them. Smaller image = faster deploys, smaller
# attack surface.
# =====================================================================
FROM python:3.12-slim AS builder

WORKDIR /build

# Build tools some ML packages (catboost, scikit-learn) need to compile
# native extensions.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir -r requirements.txt

# =====================================================================
# STAGE 2 — "runtime": copy only the venv + app code, no build tools.
# =====================================================================
FROM python:3.12-slim AS runtime

# Run as a non-root user. If the app is ever compromised, the attacker
# doesn't get root inside the container. This is a baseline expectation
# in any real deployment (and required outright on many platforms).
RUN groupadd --system app && useradd --system --gid app app

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy only what's needed to RUN the app -- training scripts, notebooks,
# and raw data don't belong in the production image.
COPY app/ ./app/
COPY common/ ./common/
COPY models/ ./models/

# The model artifacts must already exist (run train/train_pipeline.py
# before building, or in a CI step — see DEPLOYMENT.md).
RUN test -f models/catboost_model.pkl || \
    (echo "ERROR: models/catboost_model.pkl missing. Run 'python train/train_pipeline.py' before building the image." && exit 1)

RUN chown -R app:app /app
USER app

EXPOSE 8000

# A container-level health check lets `docker ps`, Docker Compose, and
# orchestrators (Kubernetes, ECS, etc.) know if the app is actually
# serving traffic, not just that the process hasn't crashed.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=3)" || exit 1

# No --reload here (that's dev-only), and we bind 0.0.0.0 so the
# container's port mapping actually reaches the process inside.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
