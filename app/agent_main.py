"""
FamaChat Agent API — Entry point do microservico de AI Agent.

Processo FastAPI isolado para o ecossistema multi-agente LangGraph.
Porta interna: 8001 (mapeada para 8008 no host).
"""

import os
import structlog
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from shared.config import settings
from shared.core.logging import setup_logging
from shared.core.tracing.middleware import TraceMiddleware
from shared.infrastructure.middleware.rate_limit import RateLimitMiddleware, RateLimitConfig
from shared.observability import (
    setup_metrics,
    setup_tracing,
    instrument_fastapi,
    instrument_sqlalchemy,
)
from shared.db.session import engine
from projects.agent.api.router import router as agent_router
from projects.agent.memory.store import create_store_cm
from projects.agent.memory.checkpointer import create_checkpointer_cm
from projects.agent.tools.http_client import init_ml_http_client, close_ml_http_client
from projects.agent.graph.builder import compile_graph
from projects.agent.config import agent_settings


# Configurar logging estruturado
setup_logging(settings.log_level)
logger = structlog.get_logger(__name__)


def _assert_single_worker():
    """Guardrail: impede startup se detectar multi-worker.

    Semaphores in-memory (_ml_api_semaphores, _stream_semaphores) so funcionam
    com 1 worker. Com 2+, os limites dobram sem controle.
    """
    workers = int(os.environ.get("WEB_CONCURRENCY", "1"))
    if workers > 1:
        raise RuntimeError(
            f"WEB_CONCURRENCY={workers} detectado. Agent API requer --workers 1. "
            "Para escalar, migre semaphores para Redis."
        )


def _validate_secrets():
    """Fail-fast se secrets estao com valores default em producao.

    Verifica:
      - AGENT_APPROVAL_TOKEN_SECRET nao pode ser o default
      - AGENT_API_KEY_HASH deve estar configurado se auth habilitada
    """
    env = settings.environment
    if env != "production":
        logger.info(
            "secrets.skip_validation",
            environment=env,
            reason="Validacao de secrets so aplica em producao",
        )
        return

    errors = []

    if agent_settings.approval_token_secret in {"change-me", "change-me-in-env"}:
        if agent_settings.require_auth:
            errors.append(
                "AGENT_APPROVAL_TOKEN_SECRET esta com valor default. "
                "Configure um secret seguro."
            )
        else:
            logger.warning(
                "secrets.insecure_default_accepted",
                detail=(
                    "AGENT_APPROVAL_TOKEN_SECRET em default no modo publico "
                    "sem auth (risco operacional aceito)."
                ),
            )

    if agent_settings.require_auth and not agent_settings.api_key_hash:
        errors.append(
            "AGENT_API_KEY_HASH nao configurado. "
            "Gere com: python -c \"import hashlib; "
            "print(hashlib.sha256(b'SUA_KEY').hexdigest())\""
        )

    if errors:
        for err in errors:
            logger.error("secrets.validation_failed", detail=err)
        raise RuntimeError(
            "Secrets inseguros detectados em producao:\n- " + "\n- ".join(errors)
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gerencia o ciclo de vida do Agent API.

    Startup:
      - Valida single-worker
      - Inicializa PostgresStore (memoria de longo prazo)
      - Inicializa AsyncPostgresSaver (checkpointer)
      - Cria client HTTP persistente para ML API
    """
    _assert_single_worker()
    _validate_secrets()

    logger.info(
        "Iniciando FamaChat Agent API",
        version=agent_settings.agent_version,
        environment=settings.environment,
        require_auth=agent_settings.require_auth,
        has_api_key_hash=bool(agent_settings.api_key_hash),
    )

    # Inicializar componentes via context managers
    async with create_store_cm() as store, create_checkpointer_cm() as checkpointer:
        ml_client = init_ml_http_client()

        # Compilar grafo UMA VEZ (caro — nao recompilar por request)
        graph = compile_graph(checkpointer=checkpointer, store=store)

        # Armazenar no state da app
        app.state.store = store
        app.state.checkpointer = checkpointer
        app.state.ml_client = ml_client
        app.state.graph = graph

        logger.info("FamaChat Agent API inicializado com sucesso")

        yield

        # Shutdown: cleanup de conexoes HTTP
        logger.info("Encerrando FamaChat Agent API")
        await close_ml_http_client()


# Criar aplicacao FastAPI
app = FastAPI(
    title="FamaChat Agent API",
    description="""
    Microservico de AI Agent para otimizacao de Facebook Ads.

    ## Funcionalidades

    * **Chat**: Interface de chat com agentes especializados via SSE
    * **6 Agentes**: Monitor de Saude, Analista de Performance, Especialista em Criativos,
      Especialista em Audiencias, Cientista de Previsao, Gerente de Operacoes
    * **Memoria**: Memoria de longo prazo com busca semantica (pgvector)
    * **Aprovacao**: Operacoes de escrita com interrupt para aprovacao humana

    ## Autenticacao

    Opera em modo configuravel:
    - `AGENT_REQUIRE_AUTH=true`: exige header `X-API-Key`
    - `AGENT_REQUIRE_AUTH=false`: endpoint publico single-user
    """,
    version=agent_settings.agent_version,
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
    openapi_url="/openapi.json" if settings.debug else None,
    lifespan=lifespan,
)

# Observabilidade: tracing + instrumentacao
setup_tracing(service_name="agent-api", service_version=agent_settings.agent_version)
instrument_fastapi(app)
instrument_sqlalchemy(engine)
setup_metrics(app, service_name="agent-api")

# Adicionar middleware de tracing
app.add_middleware(TraceMiddleware)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://www.famachat.com.br",
        "https://famachat.com.br",
        "https://marketing.famachat.com.br",
        "http://localhost:3000",
        "http://localhost:3001",
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rate_limit_config = RateLimitConfig(
    requests_per_minute=agent_settings.rate_limit_requests_per_minute,
    requests_per_hour=agent_settings.rate_limit_requests_per_hour,
    burst_limit=agent_settings.rate_limit_burst,
    enabled=agent_settings.rate_limit_enabled,
)
app.add_middleware(RateLimitMiddleware, config=rate_limit_config)

# Incluir rotas da API v1
app.include_router(agent_router, prefix="/api/v1/agent")


@app.get("/", tags=["Root"])
async def root():
    """Endpoint raiz - informacoes basicas do servico."""
    return {
        "service": "FamaChat Agent API",
        "version": agent_settings.agent_version,
        "status": "running",
        "docs": "/docs" if settings.debug else "disabled",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.agent_main:app",
        host="0.0.0.0",
        port=8001,
        reload=settings.debug,
        workers=1,  # OBRIGATORIO: semaphores in-memory
    )
