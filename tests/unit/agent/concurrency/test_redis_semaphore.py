"""Testes unitários do RedisDistributedSemaphore."""
import pytest
import pytest_asyncio

try:
    import fakeredis.aioredis as fakeredis_async
    HAS_FAKEREDIS = True
except ImportError:
    HAS_FAKEREDIS = False

from projects.agent.concurrency.redis_semaphore import (
    RedisDistributedSemaphore,
    RedisSemaphoreFactory,
)

skip_no_fakeredis = pytest.mark.skipif(
    not HAS_FAKEREDIS,
    reason="fakeredis nao instalado — pular testes Redis",
)


@skip_no_fakeredis
class TestRedisDistributedSemaphore:
    @pytest_asyncio.fixture
    async def redis_client(self):
        client = fakeredis_async.FakeRedis(decode_responses=True)
        yield client
        await client.aclose()

    @pytest.mark.asyncio
    async def test_acquire_and_release(self, redis_client):
        sem = RedisDistributedSemaphore(redis_client, "test:basic", max_concurrent=2)
        assert await sem.acquire() is True
        assert await sem.current_count() == 1
        await sem.release()
        assert await sem.current_count() == 0

    @pytest.mark.asyncio
    async def test_max_concurrent_blocks(self, redis_client):
        sem1 = RedisDistributedSemaphore(redis_client, "test:max", max_concurrent=2)
        sem2 = RedisDistributedSemaphore(redis_client, "test:max", max_concurrent=2)
        sem3 = RedisDistributedSemaphore(redis_client, "test:max", max_concurrent=2)

        assert await sem1.acquire() is True
        assert await sem2.acquire() is True
        assert await sem3.acquire() is False   # Limite atingido
        assert await sem1.is_full() is True

        await sem1.release()
        assert await sem3.acquire() is True    # Slot liberado

    @pytest.mark.asyncio
    async def test_context_manager(self, redis_client):
        sem = RedisDistributedSemaphore(redis_client, "test:ctx", max_concurrent=1)
        async with sem:
            assert await sem.current_count() == 1
        assert await sem.current_count() == 0

    @pytest.mark.asyncio
    async def test_release_without_acquire_is_noop(self, redis_client):
        sem = RedisDistributedSemaphore(redis_client, "test:noop", max_concurrent=1)
        # Nao deve lancar excecao
        await sem.release()
        assert await sem.current_count() == 0

    @pytest.mark.asyncio
    async def test_is_full(self, redis_client):
        sem = RedisDistributedSemaphore(redis_client, "test:full", max_concurrent=1)
        assert await sem.is_full() is False
        await sem.acquire()
        assert await sem.is_full() is True
        await sem.release()
        assert await sem.is_full() is False
