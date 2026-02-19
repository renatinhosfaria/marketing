"""
Dependencias FastAPI do Agent.

Auth: API Key estatica por ambiente via header X-API-Key.
Hash SHA-256 da key e comparado com AGENT_API_KEY_HASH (timing-safe).
"""

import hashlib
import hmac

import structlog
from fastapi import Request, HTTPException
from pydantic import BaseModel
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres import AsyncPostgresStore

from projects.agent.config import agent_settings

logger = structlog.get_logger()


class AuthUser(BaseModel):
    """Usuario identificado na request."""
    user_id: str
    name: str


async def verify_api_key(request: Request) -> AuthUser:
    """Valida API Key via header X-API-Key.

    Compara SHA-256(key) com AGENT_API_KEY_HASH usando hmac.compare_digest
    (timing-safe) para prevenir timing attacks.

    Se AGENT_REQUIRE_AUTH=false, usa identidade single-user fixa do runtime.
    """
    if not agent_settings.require_auth:
        return AuthUser(
            user_id=agent_settings.runtime_user_id,
            name=agent_settings.runtime_user_name,
        )

    api_key = request.headers.get("X-API-Key")
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="Header X-API-Key obrigatorio.",
        )

    if not agent_settings.api_key_hash:
        logger.error(
            "auth.api_key_hash_not_configured",
            hint="Configure AGENT_API_KEY_HASH no .env",
        )
        raise HTTPException(
            status_code=500,
            detail="Autenticacao nao configurada no servidor.",
        )

    key_hash = hashlib.sha256(api_key.encode()).hexdigest()
    if not hmac.compare_digest(key_hash, agent_settings.api_key_hash):
        raise HTTPException(
            status_code=403,
            detail="API Key invalida.",
        )

    return AuthUser(
        user_id=agent_settings.runtime_user_id,
        name=agent_settings.runtime_user_name,
    )


async def get_graph(request: Request):
    """Retorna grafo compilado do app.state."""
    return request.app.state.graph


async def get_store(request: Request) -> AsyncPostgresStore:
    """Retorna o PostgresStore do app.state."""
    return request.app.state.store


async def get_checkpointer(request: Request) -> AsyncPostgresSaver:
    """Retorna o checkpointer do app.state."""
    return request.app.state.checkpointer
