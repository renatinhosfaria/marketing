"""
Gerenciador de sessoes SSE distribuidas via Redis Stream.

Cada sessao possui:
  - stream_session_id: UUID4 unico por conexao
  - Redis Stream key: "agent:sse:{stream_session_id}"
  - TTL: 5 minutos (renovado a cada evento)
  - Suporte a replay via XRANGE com cursor Last-Event-ID
"""

import uuid
import json
import time
import redis.asyncio as aioredis
import structlog

logger = structlog.get_logger(__name__)

SESSION_TTL = 300          # 5 minutos de vida da sessao
STREAM_MAX_LEN = 500       # Maximo de eventos no stream (trim automatico)
SESSION_KEY_PREFIX = "agent:sse:"
SESSION_META_PREFIX = "agent:sse:meta:"


class SSESessionStore:
    """
    Gerencia sessoes SSE no Redis Stream.

    Uso:
        store = SSESessionStore(redis_client)
        session_id = await store.create_session(thread_id)
        event_id = await store.publish(session_id, "message", {"content": "..."})
        # Replay:
        events = await store.replay(session_id, last_event_id="1234567890-0")
    """

    def __init__(self, redis: aioredis.Redis):
        self._redis = redis

    def _stream_key(self, session_id: str) -> str:
        return f"{SESSION_KEY_PREFIX}{session_id}"

    def _meta_key(self, session_id: str) -> str:
        return f"{SESSION_META_PREFIX}{session_id}"

    async def create_session(self, thread_id: str, existing_session_id: str | None = None) -> str:
        """
        Cria ou reutiliza uma sessao SSE.

        Se existing_session_id for fornecido e existir no Redis, retorna ele.
        Caso contrario, cria uma nova sessao.
        """
        if existing_session_id:
            exists = await self._redis.exists(self._meta_key(existing_session_id))
            if exists:
                # Renova TTL da sessao existente
                await self._redis.expire(self._meta_key(existing_session_id), SESSION_TTL)
                await self._redis.expire(self._stream_key(existing_session_id), SESSION_TTL)
                return existing_session_id

        session_id = str(uuid.uuid4())
        await self._redis.hset(self._meta_key(session_id), mapping={
            "thread_id": thread_id,
            "created_at": str(time.time()),
            "status": "active",
        })
        await self._redis.expire(self._meta_key(session_id), SESSION_TTL)
        logger.info("sse_session.created", session_id=session_id, thread_id=thread_id)
        return session_id

    async def publish(self, session_id: str, event_type: str, data: dict) -> str:
        """
        Publica um evento no Redis Stream da sessao.
        Retorna o event_id (Redis Stream ID: "timestamp-seq").
        """
        stream_key = self._stream_key(session_id)
        event_id = await self._redis.xadd(
            stream_key,
            {"type": event_type, "data": json.dumps(data, ensure_ascii=False)},
            maxlen=STREAM_MAX_LEN,
            approximate=True,
        )
        # Renova TTL a cada evento publicado
        await self._redis.expire(stream_key, SESSION_TTL)
        await self._redis.expire(self._meta_key(session_id), SESSION_TTL)
        return event_id

    async def read_new(self, session_id: str, last_id: str = "$", timeout_ms: int = 15000):
        """
        Le novos eventos do stream (blocking XREAD).

        Args:
            session_id: ID da sessao
            last_id: Cursor — "$" para apenas novos, ou last event_id para replay+novos
            timeout_ms: Timeout de espera em ms (0 = nao bloquear)

        Retorna lista de (event_id, event_type, data_dict) ou [] se timeout.
        """
        stream_key = self._stream_key(session_id)
        result = await self._redis.xread(
            {stream_key: last_id},
            block=timeout_ms,
            count=100,
        )
        if not result:
            return []
        events = []
        for _key, entries in result:
            for eid, fields in entries:
                events.append((eid, fields.get("type", ""), json.loads(fields.get("data", "{}"))))
        return events

    async def replay(self, session_id: str, last_event_id: str) -> list[tuple[str, str, dict]]:
        """
        Retorna todos os eventos apos last_event_id (para reconexao).
        Usa XRANGE para buscar eventos historicos.
        """
        stream_key = self._stream_key(session_id)
        # XRANGE com start exclusivo via prefixo "("
        start = f"({last_event_id}" if not last_event_id.startswith("(") else last_event_id
        try:
            entries = await self._redis.xrange(stream_key, min=start, max="+")
        except Exception:
            # Stream nao existe mais
            return []
        return [
            (eid, fields.get("type", ""), json.loads(fields.get("data", "{}")))
            for eid, fields in entries
        ]

    async def close_session(self, session_id: str):
        """Marca sessao como encerrada (mantem stream para replay por TTL)."""
        meta_key = self._meta_key(session_id)
        await self._redis.hset(meta_key, "status", "closed")
        # TTL reduzido apos fechamento (nao precisa de 5min)
        await self._redis.expire(meta_key, 60)
        await self._redis.expire(self._stream_key(session_id), 60)

    async def session_exists(self, session_id: str) -> bool:
        """Verifica se a sessao existe no Redis."""
        return bool(await self._redis.exists(self._meta_key(session_id)))
