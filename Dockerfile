# Production image for the FinBERT sentiment API.
#
# Build (from the project root):
#   docker build -t finbert-api:local .
#
# BUILD THIS FOR THE ARCHITECTURE IT WILL RUN ON. An image built on an
# Apple-silicon Mac is linux/arm64; an ECS Fargate task definition defaults to
# X86_64 and the task dies on start with "exec format error", which surfaces in
# the ECS console as a container that exits immediately with no application log
# at all. The CI pipeline builds on ubuntu-latest (amd64), so the deployed path
# is correct; a local build pushed by hand is not. For that case:
#   docker buildx build --platform linux/amd64 -t finbert-api:local .
#
# WHY requirements-prod.txt AND NOT requirements.txt. requirements.txt is the
# development set: it carries dvc, deepchecks, locust, pytest and the full
# datasets stack, none of which the running service imports. It also resolves
# torch to the default PyPI build, which bundles CUDA — measured at 2.7 GB of
# nvidia-* libraries plus 690 MB of triton, on top of torch's own 1.2 GB. That
# is roughly 3.4 GB of GPU runtime shipped to a Fargate task that has no GPU.
# requirements-prod.txt installs the CPU-only wheel instead.

FROM python:3.12-slim

# PYTHONDONTWRITEBYTECODE: no .pyc files in the image layer.
# PYTHONUNBUFFERED: the structured JSON log lines reach the container's stdout
#   as they are emitted rather than sitting in a buffer, which is what makes
#   `docker logs -f` and the ECS log driver useful during an incident.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/home/appuser/.cache/huggingface

WORKDIR /app

# Dependencies are copied and installed before the application code so that
# editing app/main.py does not invalidate the (slow, large) dependency layer.
COPY requirements-prod.txt ./
RUN pip install --no-cache-dir -r requirements-prod.txt

# The application. app/ carries main.py and utils.py; nothing else in the
# repository is imported at runtime.
COPY app/ ./app/

# Run as a non-root user. The model cache is written at runtime under HF_HOME,
# so that directory has to belong to this user or the first model download
# fails with a permission error several seconds into startup.
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p "$HF_HOME" \
    && chown -R appuser:appuser /app /home/appuser
USER appuser

EXPOSE 8000

# uvicorn is invoked directly rather than through `python app/main.py` so that
# the process is PID 1 and receives SIGTERM from Docker and from ECS. Launched
# via the module's __main__ block, uvicorn would be a child process and a stop
# request would be delivered to the wrong process, leaving ECS to kill the task
# after the 30-second grace period on every single deployment.
#
# app.main:app (not main:app) because the import root is /app, where app/ is a
# package directory. app/main.py accepts both `from app.utils import` and
# `from utils import` so this entry point and `python app/main.py` both resolve.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
