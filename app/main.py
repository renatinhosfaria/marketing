"""
FamaChat ML - Entry point da aplicação FastAPI.
Microserviço de Machine Learning para otimização de Facebook Ads.
"""

import structlog
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.core.logging import setup_logging
from app.core.tracing.middleware import TraceMiddleware
from app.api.v1.router import api_router
from app.db.session import engine, check_database_connection


# Configurar logging estruturado
setup_logging(settings.log_level)
logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gerencia o ciclo de vida da aplicação.
    Executa na inicialização e finalização.
    """
    # Startup
    logger.info(
        "Iniciando FamaChat ML",
        version=settings.app_version,
        environment=settings.environment
    )

    # Verificar conexão com banco de dados
    db_ok = await check_database_connection()
    if not db_ok:
        logger.error("Falha na conexão com o banco de dados")
    else:
        logger.info("Conexão com banco de dados estabelecida")

    yield

    # Shutdown
    logger.info("Encerrando FamaChat ML")
    await engine.dispose()


# Criar aplicação FastAPI
app = FastAPI(
    title=settings.app_name,
    description="""
    Microserviço de Machine Learning para otimização de campanhas do Facebook Ads.

    ## Funcionalidades

    * **Recomendações**: Sugestões de otimização baseadas em regras e ML
    * **Classificações**: Categorização de campanhas por performance
    * **Previsões**: Forecast de CPL e leads
    * **Anomalias**: Detecção de comportamentos atípicos

    ## Autenticação

    Todas as rotas requerem header `X-API-Key` com a chave de API válida.
    """,
    version=settings.app_version,
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
    openapi_url="/openapi.json" if settings.debug else None,
    lifespan=lifespan
)

# Adicionar middleware de tracing (PRIMEIRO para capturar trace_id o mais cedo possível)
app.add_middleware(TraceMiddleware)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Em produção, especificar origens permitidas
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Incluir rotas da API v1
app.include_router(api_router, prefix="/api/v1")


@app.get("/", tags=["Root"])
async def root():
    """Endpoint raiz - informações básicas do serviço."""
    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "status": "running",
        "docs": "/docs" if settings.debug else "disabled"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug
    )
