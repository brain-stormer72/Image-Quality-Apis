# ---- Build stage ----
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# Install build essentials for Nuitka, but keep minimal
RUN apt-get update && apt-get install -y \
    build-essential \
    libffi-dev libssl-dev libgl1 libglib2.0-0 libgl1-mesa-glx \
    patchelf \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files and install dependencies using uv
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -r requirements.txt

# Install Nuitka using uv
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system nuitka

# Copy the application code
COPY . .

# Compile Python application to standalone executable using Nuitka
# --standalone: Create a standalone executable with all dependencies
# --onefile: Create a single executable file
# --assume-yes-for-downloads: Auto-download missing dependencies
# --enable-plugin=anti-bloat: Reduce executable size
# --nofollow-imports: Don't follow unnecessary imports
# --follow-import-to=fastapi,uvicorn,pydantic: Include essential FastAPI modules
# --include-package=cv2,numpy,PIL,aiohttp: Include computer vision dependencies
# --output-filename=blur-detection-api: Set executable name
RUN uv run python -m nuitka \
    --standalone \
    --onefile \
    --enable-plugin=anti-bloat \
    --assume-yes-for-downloads \
    --no-deployment-flag=self-execution \
    --include-module=main \
    --output-filename=blur-detection-api \
    --output-dir=/app/dist \
    main.py

# ---- Runtime stage ----
FROM debian:bookworm-slim AS runtime

# Only install libraries your app executable needs at runtime
RUN apt-get update && apt-get install -y \
    libgomp1 libglib2.0-0 libsm6 libxext6 libxrender1 libgl1-mesa-glx \
    curl \
    && rm -rf /var/lib/apt/lists/* && apt-get clean

WORKDIR /app

COPY --from=builder /app/dist/image-quality-apis /app/image-quality-apis

# Security: run as a non-root user
RUN adduser --system --group appuser && chown appuser:appuser /app/image-quality-apis
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

ENTRYPOINT ["/app/image-quality-apis"]
CMD ["--host", "0.0.0.0", "--port", "8000"]
