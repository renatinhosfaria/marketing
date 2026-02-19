"""Testes do SSESessionStore."""
import pytest
import pytest_asyncio

try:
    import fakeredis.aioredis as fakeredis_async
    HAS_FAKEREDIS = True
except ImportError:
    HAS_FAKEREDIS = False

from projects.agent.api.session_store import SSESessionStore

skip_no_fakeredis = pytest.mark.skipif(
    not HAS_FAKEREDIS,
    reason="fakeredis nao instalado",
)


@skip_no_fakeredis
class TestSSESessionStore:
    @pytest_asyncio.fixture
    async def redis_client(self):
        client = fakeredis_async.FakeRedis(decode_responses=True)
        yield client
        await client.aclose()

    @pytest_asyncio.fixture
    async def store(self, redis_client):
        return SSESessionStore(redis_client)

    @pytest.mark.asyncio
    async def test_create_session(self, store):
        sid = await store.create_session("thread:123")
        assert sid is not None
        assert len(sid) == 36  # UUID4

    @pytest.mark.asyncio
    async def test_session_exists(self, store):
        sid = await store.create_session("thread:123")
        assert await store.session_exists(sid) is True
        assert await store.session_exists("non-existent") is False

    @pytest.mark.asyncio
    async def test_reuse_existing_session(self, store):
        sid = await store.create_session("thread:123")
        sid2 = await store.create_session("thread:123", existing_session_id=sid)
        assert sid == sid2

    @pytest.mark.asyncio
    async def test_publish_and_replay(self, store):
        sid = await store.create_session("thread:123")
        eid1 = await store.publish(sid, "message", {"content": "oi"})
        eid2 = await store.publish(sid, "done", {"thread_id": "thread:123"})

        # Replay a partir do eid1 (exclusivo) deve retornar apenas eid2
        events = await store.replay(sid, eid1)
        assert len(events) == 1
        assert events[0][1] == "done"

    @pytest.mark.asyncio
    async def test_replay_empty_if_no_events_after_cursor(self, store):
        sid = await store.create_session("thread:123")
        eid1 = await store.publish(sid, "done", {"thread_id": "t"})
        events = await store.replay(sid, eid1)
        assert events == []

    @pytest.mark.asyncio
    async def test_close_session(self, store):
        sid = await store.create_session("thread:123")
        await store.close_session(sid)
        # Sessao ainda existe apos close (so muda status e TTL reduzido)
        assert await store.session_exists(sid) is True
