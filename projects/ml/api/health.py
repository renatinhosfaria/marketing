"""
Endpoints de health check.
Não requerem autenticação.
"""

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from shared.config import settings
from shared.db.session import get_db, check_database_connection
from shared.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


class HealthResponse(BaseModel):
    """Resposta do health check simples."""
    status: str
    timestamp: datetime
    version: str
    db: str
    redis: str


class DetailedHealthResponse(BaseModel):
    """Resposta do health check detalhado."""
    status: str
    timestamp: datetime
    version: str
    environment: str
    checks: dict[str, dict]


@router.get("", response_model=HealthResponse)
async def health_check():
    """
    Health check simples.
    Retorna status básico do serviço.
    """
    db_ok = await check_database_connection()
    redis_ok = await check_redis_connection()
    status_value = "healthy" if db_ok and redis_ok else "unhealthy"
    response = HealthResponse(
        status=status_value,
        timestamp=datetime.utcnow(),
        version=settings.app_version,
        db="ok" if db_ok else "fail",
        redis="ok" if redis_ok else "fail",
    )
    if not db_ok or not redis_ok:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=response.model_dump(mode='json'),
        )
    return response


@router.get("/detailed", response_model=DetailedHealthResponse)
async def detailed_health_check():
    """
    Health check detalhado.
    Verifica todas as dependências (banco, Redis, etc).
    """
    checks = {}

    # Verificar banco de dados
    db_healthy = await check_database_connection()
    checks["database"] = {
        "status": "healthy" if db_healthy else "unhealthy",
        "type": "postgresql"
    }

    # Verificar Redis (para Celery)
    redis_healthy = await check_redis_connection()
    checks["redis"] = {
        "status": "healthy" if redis_healthy else "unhealthy",
        "url": settings.redis_url.split("@")[-1] if "@" in settings.redis_url else settings.redis_url
    }

    # Status geral
    all_healthy = all(c["status"] == "healthy" for c in checks.values())

    return DetailedHealthResponse(
        status="healthy" if all_healthy else "degraded",
        timestamp=datetime.utcnow(),
        version=settings.app_version,
        environment=settings.environment,
        checks=checks
    )


async def check_redis_connection() -> bool:
    """Verifica conexão com Redis."""
    try:
        import redis.asyncio as redis
        client = redis.from_url(settings.redis_url)
        await client.ping()
        await client.close()
        return True
    except Exception as e:
        logger.error("Erro ao conectar ao Redis", error=str(e))
        return False


@router.get("/ready")
async def readiness_check():
    """
    Readiness check para Kubernetes/Docker.
    Verifica se o serviço está pronto para receber tráfego.
    """
    db_ok = await check_database_connection()

    if not db_ok:
        return {"status": "not_ready", "reason": "database_unavailable"}

    return {"status": "ready"}


@router.get("/live")
async def liveness_check():
    """
    Liveness check para Kubernetes/Docker.
    Verifica se o processo está vivo.
    """
    return {"status": "alive", "timestamp": datetime.utcnow().isoformat()}
