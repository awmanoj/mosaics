# Mosaic web UI. Build: docker build -t mosaic .   Run: docker run -p 8765:8765 mosaic
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_NO_DEV=1
WORKDIR /app

# dependencies first so they cache separately from the source
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

COPY README.md ./
COPY src ./src
RUN uv sync --frozen

ENV MOSAIC_HOST=0.0.0.0 MOSAIC_PORT=8765
EXPOSE 8765
CMD ["uv", "run", "--no-sync", "mosaic-web"]
