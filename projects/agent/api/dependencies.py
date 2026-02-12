"""
Dependencias FastAPI do Agent.
"""

from fastapi import Request
from pydantic import BaseModel
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres import AsyncPostgresStore


class AuthUser(BaseModel):
    """Usuario identificado na request."""
    user_id: str
    name: str


async def verify_api_key() -> AuthUser:
    """Retorna usuario default enquanto auth nao esta implementada."""
    return AuthUser(user_id="default", name="Usuario")


async def get_graph(request: Request):
    """Retorna grafo compilado do app.state."""
    return request.app.state.graph


async def get_store(request: Request) -> AsyncPostgresStore:
    """Retorna o PostgresStore do app.state."""
    return request.app.state.store


async def get_checkpointer(request: Request) -> AsyncPostgresSaver:
    """Retorna o checkpointer do app.state."""
    return request.app.state.checkpointer
