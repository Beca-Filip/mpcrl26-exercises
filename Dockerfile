# Python environment for .devcontainer/devcontainer.json.
FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    SDL_VIDEODRIVER=dummy

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git libgl1 libglib2.0-0 openssh-client \
    && rm -rf /var/lib/apt/lists/*

# acados at the commit pinned as a submodule in leap-c.
ARG ACADOS_COMMIT=6111f01f2220a241057b4d56d8fe3a2cddd0309f
ENV ACADOS_SOURCE_DIR=/opt/acados \
    LD_LIBRARY_PATH=/opt/acados/lib
RUN git init -q /opt/acados && cd /opt/acados \
    && git remote add origin https://github.com/acados/acados.git \
    && git fetch -q --depth 1 origin "$ACADOS_COMMIT" && git checkout -q FETCH_HEAD \
    && git submodule update -q --init --recursive --depth 1 \
    && cmake -S . -B build -DACADOS_WITH_OPENMP=ON -DACADOS_NUM_THREADS=1 \
    && cmake --build build --target install -j"$(nproc)" \
    && rm -rf build .git

COPY --from=ghcr.io/astral-sh/uv:0.12 /uv /usr/local/bin/uv

WORKDIR /workspace
COPY pyproject.toml uv.lock README.md ./

# Install the versions pinned in uv.lock (CPU-only PyTorch) into the image's Python.
RUN UV_PROJECT_ENVIRONMENT=/usr/local uv sync --locked --no-install-project --no-cache

# acados Python interface (not on PyPI). The tera renderer is fetched now, as root:
# at runtime acados would ask y/N and could not write to /opt/acados.
RUN uv pip install --system --no-cache /opt/acados/interfaces/acados_template \
    && python -c "from acados_template.utils import get_tera; get_tera(force_download=True)"

RUN useradd --create-home --shell /bin/bash vscode \
    && chown vscode:vscode /workspace
COPY --chown=vscode:vscode . /workspace
USER vscode

# Let `uv run` in the container use the image's Python, never a host `.venv`
# that the bind mount brings into /workspace. Agg avoids matplotlib hanging on
# the X11 display that VS Code forwards into the container.
ENV UV_PROJECT_ENVIRONMENT=/usr/local \
    UV_NO_SYNC=1 \
    MPLBACKEND=Agg
