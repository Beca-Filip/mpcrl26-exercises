# Python environment for .devcontainer/devcontainer.json.
FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    SDL_VIDEODRIVER=dummy

RUN apt-get update && apt-get install -y --no-install-recommends \
    git libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.12 /uv /usr/local/bin/uv

WORKDIR /workspace
COPY pyproject.toml uv.lock README.md ./

# Install the versions pinned in uv.lock (CPU-only PyTorch) into the image's Python.
RUN UV_PROJECT_ENVIRONMENT=/usr/local uv sync --locked --no-install-project --no-cache

RUN useradd --create-home --shell /bin/bash vscode \
    && chown vscode:vscode /workspace
COPY --chown=vscode:vscode . /workspace
USER vscode
