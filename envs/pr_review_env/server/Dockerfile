ARG BASE_IMAGE=openenv-base:latest
FROM ${BASE_IMAGE}

WORKDIR /app

COPY envs/pr_review_env/server/requirements.txt /tmp/requirements.txt
# Install core deps; semgrep requires a native binary so install best-effort
RUN grep -v semgrep /tmp/requirements.txt > /tmp/req_core.txt \
    && pip install --no-cache-dir -r /tmp/req_core.txt \
    && (pip install --no-cache-dir semgrep || echo "semgrep unavailable — heuristic fallback active") \
    && rm /tmp/requirements.txt /tmp/req_core.txt

COPY envs/pr_review_env/ /app/envs/pr_review_env/
COPY tasks/ /app/tasks/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "envs.pr_review_env.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
