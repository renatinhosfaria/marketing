"""
Semaphore distribuído para controle de concorrência entre múltiplos workers.

Substitui asyncio.Semaphore (in-memory) por coordenação via Redis.
Garante que no máximo N slots sejam usados simultaneamente mesmo com M workers.

Usa pipeline WATCH/MULTI/EXEC (optimistic locking) — sem Lua scripts,
compatível com fakeredis em testes.

TTL automático previne travamento permanente em caso de crash de worker.
"""

import structlog
import redis.asyncio as aioredis
import redis.exceptions

logger = structlog.get_logger(__name__)


class RedisDistributedSemaphore:
    """
    Semaphore distribuído usando Redis com TTL para auto-recover de crashes.

    Uso:
        sem = RedisDistributedSemaphore(redis_client, "user:abc", max_concurrent=3)
        if not await sem.acquire():
            raise HTTPException(429, "Limite atingido")
        try:
            ...
        finally:
            await sem.release()

        # Ou via context manager:
        async with sem:
            ...
    """

    def __init__(
        self,
        redis: aioredis.Redis,
        key: str,
        max_concurrent: int,
        ttl: int = 300,  # 5 minutos — auto-release se worker crashar
    ):
        self._redis = redis
        self._key = f"agent:sem:{key}"
        self._max = max_concurrent
        self._ttl = ttl
        self._acquired = False

    async def acquire(self) -> bool:
        """
        Tenta adquirir o semaphore atomicamente via WATCH/MULTI/EXEC.
        Retorna True se adquirido, False se limite atingido.
        """
        for _ in range(10):  # Máximo de retries em caso de contenção
            async with self._redis.pipeline(transaction=True) as pipe:
                try:
                    await pipe.watch(self._key)
                    count = int(await pipe.get(self._key) or 0)
                    if count >= self._max:
                        await pipe.unwatch()
                        return False
                    pipe.multi()
                    pipe.incr(self._key)
                    pipe.expire(self._key, self._ttl)
                    await pipe.execute()
                    self._acquired = True
                    return True
                except redis.exceptions.WatchError:
                    # Outro worker modificou a chave — tentar de novo
                    continue
        return False

    async def release(self):
        """Libera o semaphore decrementando o contador."""
        if not self._acquired:
            return
        async with self._redis.pipeline(transaction=True) as pipe:
            for _ in range(10):
                try:
                    await pipe.watch(self._key)
                    count = int(await pipe.get(self._key) or 0)
                    pipe.multi()
                    if count > 0:
                        pipe.decr(self._key)
                        pipe.expire(self._key, self._ttl)
                    await pipe.execute()
                    self._acquired = False
                    return
                except redis.exceptions.WatchError:
                    continue

    async def is_full(self) -> bool:
        """Verifica se está no limite (sem slots disponíveis)."""
        count = await self._redis.get(self._key)
        return int(count or 0) >= self._max

    async def current_count(self) -> int:
        """Retorna o número atual de slots usados."""
        count = await self._redis.get(self._key)
        return int(count or 0)

    async def __aenter__(self):
        await self.acquire()
        return self

    async def __aexit__(self, *args):
        await self.release()


class RedisSemaphoreFactory:
    """
    Factory para criar semaphores distribuídos com pool Redis compartilhado.

    Inicializada no lifespan da app, disponibilizada via app.state.sem_factory.
    """

    def __init__(self, redis_url: str):
        self._redis_url = redis_url
        self._redis: aioredis.Redis | None = None

    async def connect(self):
        """Cria o pool de conexões Redis."""
        self._redis = aioredis.from_url(
            self._redis_url,
            encoding="utf-8",
            decode_responses=True,
            max_connections=20,
        )
        # Testa conectividade
        await self._redis.ping()
        logger.info("redis_semaphore.connected", url=self._redis_url)

    async def close(self):
        """Fecha o pool de conexões."""
        if self._redis:
            await self._redis.aclose()
            self._redis = None
            logger.info("redis_semaphore.closed")

    def semaphore(
        self,
        key: str,
        max_concurrent: int,
        ttl: int = 300,
    ) -> RedisDistributedSemaphore:
        """Cria um semaphore para a chave dada."""
        if self._redis is None:
            raise RuntimeError(
                "RedisSemaphoreFactory não conectado. Chame connect() primeiro."
            )
        return RedisDistributedSemaphore(self._redis, key, max_concurrent, ttl)

    async def is_full(self, key: str, max_concurrent: int) -> bool:
        """Verifica rapidamente se o semaphore está cheio."""
        return await self.semaphore(key, max_concurrent).is_full()
