"""
Gerenciador do ciclo de vida do SSESessionStore.
Inicializado no lifespan da app.
"""

import redis.asyncio as aioredis
import structlog
from projects.agent.api.session_store import SSESessionStore

logger = structlog.get_logger(__name__)


class SSESessionManager:
    def __init__(self, redis_url: str):
        self._redis_url = redis_url
        self._redis: aioredis.Redis | None = None
        self._store: SSESessionStore | None = None

    async def connect(self):
        self._redis = aioredis.from_url(
            self._redis_url,
            encoding="utf-8",
            decode_responses=True,
            max_connections=10,
        )
        await self._redis.ping()
        self._store = SSESessionStore(self._redis)
        logger.info("sse_session_manager.connected")

    async def close(self):
        if self._redis:
            await self._redis.aclose()
            self._redis = None
            self._store = None

    @property
    def store(self) -> SSESessionStore | None:
        return self._store
