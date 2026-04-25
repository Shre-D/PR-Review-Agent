ARG BASE_IMAGE=openenv-base:latest
FROM ${BASE_IMAGE}

# ── System toolchains ──────────────────────────────────────────────────────
# These are needed so check_build_and_types runs real compilers/type-checkers
# instead of falling back to heuristics for Go, Rust, Java, and TypeScript.
# All installs are best-effort: a missing toolchain silently degrades to the
# heuristic layer rather than failing the build.

RUN apt-get update && apt-get install -y --no-install-recommends \
        default-jdk \
        golang-go \
        nodejs \
        npm \
        curl \
        ca-certificates \
    && npm install -g typescript \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Rust via rustup (apt rust is often too old for cargo check to work reliably)
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --default-toolchain stable --profile minimal \
    && echo 'source $HOME/.cargo/env' >> /etc/profile.d/cargo.sh
ENV PATH="/root/.cargo/bin:${PATH}"

# ── Python deps ────────────────────────────────────────────────────────────

WORKDIR /app

COPY envs/pr_review_env/server/requirements.txt /tmp/requirements.txt
# semgrep needs a native binary — install best-effort
RUN grep -v semgrep /tmp/requirements.txt > /tmp/req_core.txt \
    && pip install --no-cache-dir -r /tmp/req_core.txt \
    && (pip install --no-cache-dir semgrep || echo "semgrep unavailable — heuristic fallback active") \
    && rm /tmp/requirements.txt /tmp/req_core.txt

COPY envs/pr_review_env/ /app/envs/pr_review_env/
COPY tasks/ /app/tasks/

EXPOSE 8000

# HF Spaces injects $PORT (usually 7860). Fall back to 8000 for local/Docker use.
ENV PORT=8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD sh -c 'python -c "import urllib.request; urllib.request.urlopen(\"http://localhost:${PORT:-8000}/health\")"' || exit 1

CMD ["sh", "-c", "uvicorn envs.pr_review_env.server.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
