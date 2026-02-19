"""
Marketing Facebook Ads - Entry point do microservico de Facebook Ads.
Processo FastAPI isolado para integracao com a API do Facebook.
"""

import structlog
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from shared.config import settings
from shared.core.logging import setup_logging
from shared.core.tracing.middleware import TraceMiddleware
from shared.observability import setup_metrics, setup_tracing, instrument_fastapi, instrument_sqlalchemy
from app.fb_ads_router import fb_ads_api_router
from shared.db.session import engine, check_database_connection


# Configurar logging estruturado
setup_logging(settings.log_level)
logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gerencia o ciclo de vida do microservico Facebook Ads.
    """
    # Startup
    logger.info(
        "Iniciando Marketing Facebook Ads",
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
    logger.info("Encerrando Marketing Facebook Ads")
    await engine.dispose()


# Criar aplicacao FastAPI
app = FastAPI(
    title="Marketing Facebook Ads",
    description="""
    Microservico de integracao com Facebook Ads.

    ## Funcionalidades

    * **OAuth**: Autenticacao com contas Facebook
    * **Sync**: Sincronizacao de campanhas, ad sets e anuncios
    * **Insights**: Metricas e dados de performance
    * **Config**: Gerenciamento de contas conectadas

    ## Autenticacao

    Rotas protegidas requerem header `X-API-Key` ou Bearer token JWT.
    """,
    version=settings.app_version,
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
    openapi_url="/openapi.json" if settings.debug else None,
    lifespan=lifespan,
)

# Observabilidade: tracing + instrumentação (antes de adicionar middleware manual)
setup_tracing(service_name="fb-ads-api", service_version=settings.app_version)
instrument_fastapi(app)
instrument_sqlalchemy(engine)

# Métricas Prometheus (/metrics)
setup_metrics(app, service_name="fb-ads-api")

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
app.include_router(fb_ads_api_router, prefix="/api/v1")


@app.get("/", tags=["Root"])
async def root():
    """Endpoint raiz - informacoes basicas do servico."""
    return {
        "service": "Marketing Facebook Ads",
        "version": settings.app_version,
        "status": "running",
        "docs": "/docs" if settings.debug else "disabled",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.fb_ads_main:app",
        host="0.0.0.0",
        port=8002,
        reload=settings.debug,
    )
