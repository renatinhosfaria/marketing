# syntax=docker/dockerfile:1.4
# Marketing - Microservico de Machine Learning
# Build otimizado com uv e BuildKit cache

# ==================== BUILD STAGE ====================
FROM python:3.11-slim AS builder

WORKDIR /app

# Instalar uv (instalador de pacotes ultra-rapido) e dependencias de compilacao
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    && curl -LsSf https://astral.sh/uv/install.sh | sh

# Adicionar uv ao PATH
ENV PATH="/root/.local/bin:$PATH"

# Copiar requirements e instalar dependencias com cache
COPY requirements.txt .

# Usar uv para instalar (10-100x mais rapido que pip)
# Cache de downloads e compilacoes
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    uv pip install --system --compile-bytecode -r requirements.txt

# ==================== PRODUCTION STAGE ====================
FROM python:3.11-slim AS production

# Metadata
LABEL maintainer="Marketing Team"
LABEL description="Microservico de ML para otimizacao de Facebook Ads"
LABEL version="1.0.0"

# Variaveis de ambiente
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    TZ=America/Sao_Paulo \
    HOME=/home/marketing \
    MPLCONFIGDIR=/home/marketing/.config/matplotlib

# Criar usuario nao-root com home
RUN groupadd -r marketing && useradd -r -g marketing -m -d /home/marketing marketing

WORKDIR /app

# Instalar dependencias de runtime
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copiar pacotes Python instalados do builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copiar codigo da aplicacao
COPY --chown=marketing:marketing shared/ ./shared/
COPY --chown=marketing:marketing projects/ ./projects/
COPY --chown=marketing:marketing app/ ./app/
COPY --chown=marketing:marketing scripts/ ./scripts/
COPY --chown=marketing:marketing alembic/ ./alembic/
COPY --chown=marketing:marketing alembic.ini .

# Criar diretorios necessarios
RUN mkdir -p /app/models_storage /app/logs /home/marketing/.config/matplotlib && \
    chown -R marketing:marketing /app /home/marketing

# Trocar para usuario nao-root
USER marketing

# Expor porta
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Comando padrao - iniciar API
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
