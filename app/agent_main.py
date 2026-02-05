"""
Marketing Agent - Entry point do microservico de Agente IA.
Processo FastAPI isolado para o Agente de Trafego Pago.
"""

import structlog
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from shared.config import settings
from shared.core.logging import setup_logging
from shared.core.tracing.middleware import TraceMiddleware
from app.agent_router import agent_api_router
from shared.db.session import engine, check_database_connection


# Configurar logging estruturado
setup_logging(settings.log_level)
logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gerencia o ciclo de vida do microservico Agent.
    """
    # Startup
    logger.info(
        "Iniciando Marketing Agent",
        version=settings.app_version,
        environment=settings.environment,
    )

    db_ok = await check_database_connection()
    if not db_ok:
        logger.error("Falha na conexao com o banco de dados")
    else:
        logger.info("Conexao com banco de dados estabelecida")

    yield

    # Shutdown
    logger.info("Encerrando Marketing Agent")
    await engine.dispose()


# Criar aplicacao FastAPI
app = FastAPI(
    title="Marketing Agent",
    description="""
    Microservico do Agente IA para gestao de trafego pago.

    ## Funcionalidades

    * **Chat**: Conversa com o agente sobre campanhas
    * **Streaming**: Respostas em tempo real via SSE
    * **Multi-Agent**: Sistema orquestrado com subagentes especializados
    * **Analise**: Analises rapidas sem persistir conversa
    * **Feedback**: Sistema de avaliacao de respostas

    ## Autenticacao

    Rotas protegidas requerem header `X-API-Key` ou Bearer token JWT.
    """,
    version=settings.app_version,
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
    openapi_url="/openapi.json" if settings.debug else None,
    lifespan=lifespan,
)

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

# Incluir rotas da API v1
app.include_router(agent_api_router, prefix="/api/v1")


@app.get("/", tags=["Root"])
async def root():
    """Endpoint raiz - informacoes basicas do servico."""
    return {
        "service": "Marketing Agent",
        "version": settings.app_version,
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
    )
