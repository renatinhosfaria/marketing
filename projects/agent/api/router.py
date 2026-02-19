"""
Router da API do Agent.

POST /chat: endpoint principal com SSE stream.
GET /health: healthcheck do servico.
"""

import asyncio
import re
import time

from fastapi import APIRouter, Request, Depends, HTTPException
from psycopg import OperationalError
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import Command

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres import AsyncPostgresStore

from projects.agent.api.dependencies import (
    get_graph,
    verify_api_key,
    AuthUser,
    get_store,
    get_checkpointer,
)
from projects.agent.api.schemas import (
    ACCOUNT_ID_PATTERN,
    ChatRequest,
    ConversationPreview,
    ConversationMessages,
)
from projects.agent.memory.namespaces import StoreNamespace
from projects.agent.api.stream import sse_event, sse_event_with_id, _safe_json
from projects.agent.api.session_store import SSESessionStore
from projects.agent.config import agent_settings
from projects.agent.observability.metrics import (
    agent_requests_total,
    agent_response_duration,
    agent_active_streams,
    stream_errors_total,
    stream_first_event_latency,
    stream_time_to_done,
    stream_reconnect_total,
)

import structlog

logger = structlog.get_logger()

router = APIRouter()

MAX_STREAMS_PER_USER = 3
_ACCOUNT_ID_RE = re.compile(ACCOUNT_ID_PATTERN)

# Fallback in-memory para dev/testes (sem Redis)
_stream_semaphores_fallback: dict[str, asyncio.Semaphore] = {}


def _validate_account_id(account_id: str) -> str:
    """Valida account_id aceitando act_<digitos> ou <digitos>."""
    if not _ACCOUNT_ID_RE.fullmatch(account_id):
        raise HTTPException(
            status_code=400,
            detail="account_id invalido. Use act_<digitos> ou <digitos>",
        )
    return account_id


def _build_thread_id(thread_id: str, user_id: str, account_id: str) -> str:
    """Garante que thread_id e unico por (user_id, account_id).

    Formato canonico: "{user_id}:{account_id}:{frontend_thread_id}".
    Se vier um UUID puro do frontend, prefixamos.
    Se vier prefixado, validamos que user/account conferem.
    """
    # Validar formato: apenas alfanumericos, hifens e underscores
    raw_id = thread_id.split(":")[-1] if ":" in thread_id else thread_id
    if not re.match(r'^[\w\-]+$', raw_id):
        raise HTTPException(400, "thread_id contem caracteres invalidos.")

    prefix = f"{user_id}:{account_id}:"
    if thread_id.startswith(prefix):
        return thread_id  # Ja prefixado corretamente

    # Se vier com prefixo de outro tenant, bloqueia
    parts = thread_id.split(":", 2)
    if len(parts) == 3 and (parts[0] != str(user_id) or parts[1] != account_id):
        raise HTTPException(
            status_code=400,
            detail="thread_id pertence a outro tenant.",
        )

    return f"{prefix}{thread_id}"


def _resolve_stream_status(task: asyncio.Task) -> str:
    """Classifica status final da task de stream sem propagar CancelledError."""
    if not task.done():
        return "ok"
    if task.cancelled():
        return "cancelled"
    try:
        exc = task.exception()
    except asyncio.CancelledError:
        return "cancelled"
    return "error" if exc else "ok"


@router.post("/chat")
async def chat_stream(
    request: Request,
    body: ChatRequest,
    user: AuthUser = Depends(verify_api_key),
    graph=Depends(get_graph),
):
    """Endpoint principal de chat. Retorna SSE stream com autenticacao.

    Suporta:
    - Envio de mensagem nova (body.message)
    - Retomada apos interrupt (body.resume_payload)
    """
    account_id = _validate_account_id(body.account_id)

    # Suporte a reconexao SSE
    last_event_id = request.headers.get("Last-Event-ID")
    existing_session_id = request.headers.get("X-Stream-Session-ID")
    is_reconnect = bool(last_event_id and existing_session_id)

    # SSE Session Manager (Redis) — pode ser None em dev/testes
    sse_manager = getattr(request.app.state, "sse_session_manager", None)
    sse_store: SSESessionStore | None = sse_manager.store if sse_manager else None

    logger.info(
        "chat.request_received",
        user_id=str(user.user_id),
        account_id=account_id,
        thread_id=body.thread_id,
        has_message=bool(body.message),
        is_resume=bool(body.resume_payload),
        is_reconnect=is_reconnect,
    )

    # Semaphore distribuído via Redis (ou fallback in-memory em dev)
    sem_factory = getattr(request.app.state, "sem_factory", None)
    if sem_factory is not None:
        redis_stream_sem = sem_factory.semaphore(
            f"stream:user:{user.user_id}",
            max_concurrent=MAX_STREAMS_PER_USER,
            ttl=600,
        )
        if await redis_stream_sem.is_full():
            raise HTTPException(429, "Limite de streams concorrentes atingido.")
    else:
        # Fallback: asyncio.Semaphore in-memory (single-worker)
        redis_stream_sem = None
        uid = str(user.user_id)
        if uid not in _stream_semaphores_fallback:
            _stream_semaphores_fallback[uid] = asyncio.Semaphore(MAX_STREAMS_PER_USER)
        if _stream_semaphores_fallback[uid].locked():
            raise HTTPException(429, "Limite de streams concorrentes atingido.")

    # thread_id prefixado para isolamento multi-tenant
    safe_thread_id = _build_thread_id(
        body.thread_id, str(user.user_id), account_id,
    )
    config = {
        "configurable": {
            "thread_id": safe_thread_id,
            "user_id": str(user.user_id),
            "account_id": account_id,
        }
    }

    # Determina input: nova mensagem ou resume de interrupt
    if body.resume_payload:
        input_data = Command(resume=body.resume_payload.model_dump())
    else:
        if not body.message:
            raise HTTPException(400, "message e obrigatorio quando nao e resume.")
        input_data = {
            "messages": [HumanMessage(content=body.message)],
            "user_context": {
                "user_id": str(user.user_id),
                "account_id": account_id,
                "account_name": "",
                "timezone": "America/Sao_Paulo",
            },
        }

    async def event_generator():
        # Adquire semaphore (Redis ou fallback in-memory)
        if redis_stream_sem is not None:
            acquired = await redis_stream_sem.acquire()
            if not acquired:
                yield sse_event("error", {"message": "Limite de streams atingido."})
                return
        else:
            await _stream_semaphores_fallback[str(user.user_id)].acquire()

        agent_active_streams.inc()
        stream_start = time.monotonic()
        stream_status = "ok"
        keepalive_interval = agent_settings.sse_keepalive_interval
        queue: asyncio.Queue[str | None] = asyncio.Queue(maxsize=100)
        _first_event_recorded = False

        # Criar ou reutilizar sessao SSE no Redis
        stream_session_id: str | None = None
        if sse_store:
            if is_reconnect:
                stream_reconnect_total.labels(status="attempted").inc()
                session_exists = await sse_store.session_exists(existing_session_id)
                if session_exists:
                    stream_session_id = await sse_store.create_session(
                        safe_thread_id, existing_session_id
                    )
                    # Replay eventos perdidos
                    replay_events = await sse_store.replay(stream_session_id, last_event_id)
                    for eid, etype, edata in replay_events:
                        yield sse_event_with_id(eid, etype, edata)
                    stream_reconnect_total.labels(status="success").inc()
                else:
                    stream_reconnect_total.labels(status="session_not_found").inc()
                    stream_session_id = await sse_store.create_session(safe_thread_id)
            else:
                stream_session_id = await sse_store.create_session(safe_thread_id)

            # Envia session_id para o cliente (para usar em reconexao)
            yield sse_event("session", {"stream_session_id": stream_session_id})

        async def _publish_event(event_type: str, data: dict):
            """Publica no Redis Stream e envia via queue local."""
            if sse_store and stream_session_id:
                eid = await sse_store.publish(stream_session_id, event_type, data)
                await queue.put(sse_event_with_id(eid, event_type, data))
            else:
                await queue.put(sse_event(event_type, data))

        async def stream_graph():
            """Roda o grafo e coloca eventos na queue."""
            nonlocal _first_event_recorded
            try:
                async for namespace, mode, chunk in graph.astream(
                    input_data,
                    config=config,
                    stream_mode=["messages", "updates", "custom"],
                    subgraphs=True,
                ):
                    agent_source = namespace[-1] if namespace else "supervisor"

                    if mode == "messages":
                        msg_chunk, metadata = chunk
                        node = metadata.get("langgraph_node", "")

                        # Registra TTFB (primeiro evento nao-keepalive)
                        if not _first_event_recorded:
                            stream_first_event_latency.observe(
                                time.monotonic() - stream_start
                            )
                            _first_event_recorded = True

                        # Tool results de qualquer agente (Generative UI)
                        if isinstance(msg_chunk, ToolMessage) and msg_chunk.name:
                            await _publish_event("tool_result", {
                                "tool": msg_chunk.name,
                                "data": _safe_json(msg_chunk.content),
                                "agent": metadata.get("langgraph_node", "unknown"),
                            })

                        # Texto: apenas o synthesizer streama ao usuario.
                        # Supervisor usa structured output (JSON interno).
                        # Agentes especialistas contribuem via agent_reports
                        # que o synthesizer consolida em resposta unica.
                        if node == "synthesizer" and msg_chunk.content:
                            await _publish_event("message", {
                                "content": msg_chunk.content,
                                "agent": "synthesizer",
                            })

                    elif mode == "updates":
                        if "__interrupt__" in chunk:
                            interrupt_data = chunk["__interrupt__"][0].value
                            await _publish_event("interrupt", {
                                "type": interrupt_data["type"],
                                "approval_token": interrupt_data["approval_token"],
                                "details": interrupt_data["details"],
                                "thread_id": body.thread_id,
                            })
                        else:
                            for node_name, node_output in chunk.items():
                                # Supervisor pode retornar AIMessage direta
                                # (fallback sem agentes / limite de steps).
                                if (
                                    node_name == "supervisor"
                                    and isinstance(node_output, dict)
                                ):
                                    for msg in node_output.get("messages", []):
                                        if isinstance(msg, AIMessage) and msg.content:
                                            await _publish_event(
                                                "message",
                                                {"content": msg.content, "agent": "supervisor"},
                                            )
                                await _publish_event("agent_status", {
                                    "agent": node_name,
                                    "source": agent_source,
                                    "status": "completed",
                                })

                    elif mode == "custom":
                        await _publish_event("agent_progress", {
                            "agent": agent_source,
                            **chunk,
                        })

                stream_time_to_done.observe(time.monotonic() - stream_start)
                await _publish_event("done", {"thread_id": body.thread_id})
            except asyncio.CancelledError:
                stream_errors_total.labels(error_type="cancelled").inc()
                await _publish_event("done", {
                    "thread_id": body.thread_id,
                    "reason": "client_disconnected",
                })
            except Exception as exc:
                stream_errors_total.labels(error_type="graph_error").inc()
                logger.error(
                    "stream.graph_error",
                    error=str(exc),
                    error_type=type(exc).__name__,
                    thread_id=safe_thread_id,
                )
                await _publish_event("error", {
                    "message": "Erro interno ao processar sua mensagem. Tente novamente.",
                    "thread_id": body.thread_id,
                })
            finally:
                await queue.put(None)  # Sentinel: fim do stream

        # Inicia o grafo como task
        graph_task = asyncio.create_task(stream_graph())

        try:
            while True:
                # Verifica disconnect do cliente
                if await request.is_disconnected():
                    stream_status = "cancelled"
                    graph_task.cancel()
                    break

                try:
                    event = await asyncio.wait_for(
                        queue.get(), timeout=keepalive_interval,
                    )
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
                    continue

                if event is None:  # Sentinel
                    break

                yield event
        finally:
            agent_active_streams.dec()
            # Libera semaphore (Redis ou fallback in-memory)
            if redis_stream_sem is not None:
                await redis_stream_sem.release()
            else:
                _stream_semaphores_fallback[str(user.user_id)].release()

            elapsed = time.monotonic() - stream_start
            if not graph_task.done():
                graph_task.cancel()
                try:
                    await graph_task
                except asyncio.CancelledError:
                    stream_status = "cancelled"

            if stream_status == "ok":
                stream_status = _resolve_stream_status(graph_task)

            # Fechar sessao SSE no Redis
            if sse_store and stream_session_id:
                try:
                    await sse_store.close_session(stream_session_id)
                except Exception as e:
                    logger.warning("sse_session.close_failed", error=str(e))

            agent_response_duration.labels(routing_urgency="default").observe(elapsed)
            agent_requests_total.labels(endpoint="/chat", status=stream_status).inc()
            logger.info(
                "chat.stream_completed",
                thread_id=safe_thread_id,
                duration_s=round(elapsed, 2),
                status=stream_status,
            )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/conversations")
async def list_conversations(
    account_id: str,
    user: AuthUser = Depends(verify_api_key),
    checkpointer: AsyncPostgresSaver = Depends(get_checkpointer),
    store: AsyncPostgresStore = Depends(get_store),
):
    """Lista conversas do usuario para uma conta."""
    account_id = _validate_account_id(account_id)
    prefix = f"{user.user_id}:{account_id}:"

    # Query the checkpoints table directly via the pool
    async with checkpointer.conn.connection() as conn:
        rows = await conn.execute(
            """
            SELECT thread_id,
                   MIN(created_at) AS created_at,
                   MAX(created_at) AS last_message_at
            FROM checkpoints
            WHERE thread_id LIKE %s
              AND checkpoint_ns = ''
            GROUP BY thread_id
            ORDER BY MAX(created_at) DESC
            LIMIT 50
            """,
            (f"{prefix}%",),
        )
        threads = await rows.fetchall()

    if not threads:
        return {"conversations": []}

    # Fetch titles from store (graceful degradation se store indisponivel)
    titles: dict[str, str] = {}
    try:
        ns = StoreNamespace.conversation_titles(str(user.user_id), account_id)
        try:
            title_items = await store.asearch(ns)
        except OperationalError:
            # Conexao morta — pool descarta e segunda tentativa pega conexao nova
            logger.warning("store.retry_after_connection_error", operation="asearch")
            title_items = await store.asearch(ns)
        if title_items:
            for item in title_items:
                titles[item.key] = item.value.get("title", "Nova conversa")
    except Exception as e:
        logger.warning("conversations.store_search_failed", error=str(e))

    conversations = []
    for row in threads:
        full_thread_id = row["thread_id"]
        frontend_id = full_thread_id.removeprefix(prefix)
        conversations.append(ConversationPreview(
            thread_id=frontend_id,
            title=titles.get(frontend_id, "Nova conversa"),
            created_at=row["created_at"],
            last_message_at=row["last_message_at"],
        ))

    return {"conversations": conversations}


@router.get("/conversations/{thread_id}/messages")
async def get_conversation_messages(
    thread_id: str,
    account_id: str,
    user: AuthUser = Depends(verify_api_key),
    checkpointer: AsyncPostgresSaver = Depends(get_checkpointer),
):
    """Retorna mensagens de uma conversa."""
    account_id = _validate_account_id(account_id)
    safe_thread_id = _build_thread_id(thread_id, str(user.user_id), account_id)
    config = {"configurable": {"thread_id": safe_thread_id}}

    checkpoint_tuple = await checkpointer.aget_tuple(config)
    if not checkpoint_tuple:
        raise HTTPException(404, "Conversa nao encontrada.")

    state = checkpoint_tuple.checkpoint
    channel_values = state.get("channel_values", {})
    raw_messages = channel_values.get("messages", [])

    messages = []
    for msg in raw_messages:
        if isinstance(msg, HumanMessage):
            messages.append({
                "role": "user",
                "content": msg.content,
            })
        elif isinstance(msg, AIMessage) and msg.content:
            messages.append({
                "role": "assistant",
                "content": msg.content,
            })

    return ConversationMessages(thread_id=thread_id, messages=messages)


@router.delete("/conversations/{thread_id}")
async def delete_conversation(
    thread_id: str,
    account_id: str,
    user: AuthUser = Depends(verify_api_key),
    checkpointer: AsyncPostgresSaver = Depends(get_checkpointer),
    store: AsyncPostgresStore = Depends(get_store),
):
    """Deleta uma conversa (checkpoints + titulo)."""
    account_id = _validate_account_id(account_id)
    safe_thread_id = _build_thread_id(thread_id, str(user.user_id), account_id)

    async with checkpointer.conn.connection() as conn:
        async with conn.transaction():
            await conn.execute(
                "DELETE FROM checkpoint_writes WHERE thread_id = %s",
                (safe_thread_id,),
            )
            await conn.execute(
                "DELETE FROM checkpoint_blobs WHERE thread_id = %s",
                (safe_thread_id,),
            )
            await conn.execute(
                "DELETE FROM checkpoints WHERE thread_id = %s",
                (safe_thread_id,),
            )

    # Delete title from store
    ns = StoreNamespace.conversation_titles(str(user.user_id), account_id)
    try:
        try:
            await store.adelete(ns, thread_id)
        except OperationalError:
            logger.warning("store.retry_after_connection_error", operation="adelete")
            await store.adelete(ns, thread_id)
    except Exception as e:
        logger.warning("title.delete_failed", thread_id=thread_id, error=str(e))

    return {"status": "deleted", "thread_id": thread_id}


@router.get("/health")
async def health():
    """Healthcheck do servico Agent API."""
    return {
        "status": "healthy",
        "service": "famachat-agent-api",
        "version": agent_settings.agent_version,
    }
