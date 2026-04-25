FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PR_REVIEW_TOOL_BACKEND=heuristic \
    PORT=7860

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md openenv.yaml /app/
COPY envs/ /app/envs/
COPY tasks/ /app/tasks/
COPY fixtures/ /app/fixtures/
COPY docs/ /app/docs/
COPY assets/ /app/assets/
COPY benchmarks/ /app/benchmarks/
COPY train/ /app/train/
COPY ui/ /app/ui/

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -e ".[server]"

EXPOSE 7860

CMD ["sh", "-c", "uvicorn envs.pr_review_env.server.app:app --host 0.0.0.0 --port ${PORT:-7860}"]
