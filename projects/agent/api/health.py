"""
Endpoints de health check do Agente IA.
Nao requerem autenticacao.
"""

from datetime import datetime

from fastapi import APIRouter, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from shared.config import settings
from shared.db.session import check_database_connection
from shared.core.logging import get_logger
from projects.agent.config import get_agent_settings

logger = get_logger(__name__)
router = APIRouter()


class AgentHealthResponse(BaseModel):
    """Resposta do health check do Agent."""
    status: str
    timestamp: datetime
    version: str
    db: str
    redis: str
    llm_provider: str
    llm_model: str


@router.get("", response_model=AgentHealthResponse)
async def health_check():
    """
    Health check do Agente IA.
    Verifica banco de dados, Redis e configuracao do LLM.
    """
    db_ok = await check_database_connection()
    redis_ok = await _check_redis()
    agent_settings = get_agent_settings()
    status_value = "healthy" if db_ok and redis_ok else "unhealthy"

    response = AgentHealthResponse(
        status=status_value,
        timestamp=datetime.utcnow(),
        version=settings.app_version,
        db="ok" if db_ok else "fail",
        redis="ok" if redis_ok else "fail",
        llm_provider=agent_settings.llm_provider,
        llm_model=agent_settings.llm_model,
    )

    if not db_ok or not redis_ok:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=response.model_dump(mode="json"),
        )
    return response


@router.get("/live")
async def liveness_check():
    """Liveness check - verifica se o processo esta vivo."""
    return {"status": "alive", "timestamp": datetime.utcnow().isoformat()}


@router.get("/ready")
async def readiness_check():
    """Readiness check - verifica se o servico esta pronto para trafego."""
    db_ok = await check_database_connection()
    if not db_ok:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "not_ready", "reason": "database_unavailable"},
        )
    return {"status": "ready"}


async def _check_redis() -> bool:
    """Verifica conexao com Redis."""
    try:
        import redis.asyncio as aioredis
        client = aioredis.from_url(settings.redis_url)
        await client.ping()
        await client.close()
        return True
    except Exception as e:
        logger.error("Erro ao conectar ao Redis", error=str(e))
        return False
