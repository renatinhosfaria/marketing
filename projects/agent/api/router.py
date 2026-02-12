"""
Router da API do Agent.

POST /chat: endpoint principal com SSE stream.
GET /health: healthcheck do servico.
"""

import asyncio
import time

from fastapi import APIRouter, Request, Depends, HTTPException
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
from projects.agent.api.schemas import ChatRequest, ConversationPreview, ConversationMessages
from projects.agent.memory.namespaces import StoreNamespace
from projects.agent.api.stream import sse_event, _safe_json
from projects.agent.config import agent_settings
from projects.agent.observability.metrics import (
    agent_requests_total,
    agent_response_duration,
    agent_active_streams,
)

router = APIRouter()

# Controle de streams concorrentes por usuario via Semaphore
_stream_semaphores: dict[str, asyncio.Semaphore] = {}
MAX_STREAMS_PER_USER = 3


def _get_semaphore(user_id: str) -> asyncio.Semaphore:
    """Retorna ou cria o Semaphore para o usuario.

    Thread-safe em single-worker async.
    """
    if user_id not in _stream_semaphores:
        _stream_semaphores[user_id] = asyncio.Semaphore(MAX_STREAMS_PER_USER)
    return _stream_semaphores[user_id]


def _build_thread_id(thread_id: str, user_id: str, account_id: str) -> str:
    """Garante que thread_id e unico por (user_id, account_id).

    Formato canonico: "{user_id}:{account_id}:{frontend_thread_id}".
    Se vier um UUID puro do frontend, prefixamos.
    Se vier prefixado, validamos que user/account conferem.
    """
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
    semaphore = _get_semaphore(str(user.user_id))
    if semaphore.locked():
        raise HTTPException(429, "Limite de streams concorrentes atingido.")

    # thread_id prefixado para isolamento multi-tenant
    safe_thread_id = _build_thread_id(
        body.thread_id, str(user.user_id), body.account_id,
    )
    config = {
        "configurable": {
            "thread_id": safe_thread_id,
            "user_id": str(user.user_id),
            "account_id": body.account_id,
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
                "account_id": body.account_id,
                "account_name": "",
                "timezone": "America/Sao_Paulo",
            },
        }

    async def event_generator():
        async with semaphore:
            agent_active_streams.inc()
            stream_start = time.monotonic()
            keepalive_interval = agent_settings.sse_keepalive_interval
            queue: asyncio.Queue[str | None] = asyncio.Queue(maxsize=100)

            async def stream_graph():
                """Roda o grafo e coloca eventos na queue."""
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
                            node = metadata.get(
                                "langgraph_node", "",
                            )

                            # Tool results de qualquer agente (Generative UI)
                            if (
                                isinstance(msg_chunk, ToolMessage)
                                and msg_chunk.name
                            ):
                                await queue.put(sse_event("tool_result", {
                                    "tool": msg_chunk.name,
                                    "data": _safe_json(msg_chunk.content),
                                    "agent": metadata.get(
                                        "langgraph_node", "unknown",
                                    ),
                                }))

                            # Texto: apenas o synthesizer streama ao usuario.
                            # Supervisor usa structured output (JSON interno).
                            # Agentes especialistas contribuem via agent_reports
                            # que o synthesizer consolida em resposta unica.
                            if node == "synthesizer" and msg_chunk.content:
                                await queue.put(sse_event("message", {
                                    "content": msg_chunk.content,
                                    "agent": "synthesizer",
                                }))

                        elif mode == "updates":
                            if "__interrupt__" in chunk:
                                interrupt_data = chunk["__interrupt__"][0].value
                                await queue.put(sse_event("interrupt", {
                                    "type": interrupt_data["type"],
                                    "approval_token": interrupt_data[
                                        "approval_token"
                                    ],
                                    "details": interrupt_data["details"],
                                    "thread_id": body.thread_id,
                                }))
                            else:
                                for node_name, node_output in chunk.items():
                                    # Supervisor pode retornar AIMessage direta
                                    # (fallback sem agentes / limite de steps).
                                    # Como filtramos "messages" mode do supervisor,
                                    # precisamos capturar respostas textuais aqui.
                                    if (
                                        node_name == "supervisor"
                                        and isinstance(node_output, dict)
                                    ):
                                        for msg in node_output.get("messages", []):
                                            if (
                                                isinstance(msg, AIMessage)
                                                and msg.content
                                            ):
                                                await queue.put(sse_event(
                                                    "message",
                                                    {
                                                        "content": msg.content,
                                                        "agent": "supervisor",
                                                    },
                                                ))
                                    await queue.put(sse_event("agent_status", {
                                        "agent": node_name,
                                        "source": agent_source,
                                        "status": "completed",
                                    }))

                        elif mode == "custom":
                            await queue.put(sse_event("agent_progress", {
                                "agent": agent_source,
                                **chunk,
                            }))

                    await queue.put(sse_event("done", {
                        "thread_id": body.thread_id,
                    }))
                except asyncio.CancelledError:
                    await queue.put(sse_event("done", {
                        "thread_id": body.thread_id,
                        "reason": "client_disconnected",
                    }))
                finally:
                    await queue.put(None)  # Sentinel: fim do stream

            # Inicia o grafo como task
            graph_task = asyncio.create_task(stream_graph())

            try:
                while True:
                    # Verifica disconnect do cliente
                    if await request.is_disconnected():
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
                elapsed = time.monotonic() - stream_start
                agent_response_duration.labels(routing_urgency="default").observe(elapsed)
                agent_requests_total.labels(endpoint="/chat", status="ok").inc()
                if not graph_task.done():
                    graph_task.cancel()
                    try:
                        await graph_task
                    except asyncio.CancelledError:
                        pass

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

    # Fetch titles from store
    ns = StoreNamespace.conversation_titles(str(user.user_id), account_id)
    title_items = await store.asearch(ns)

    titles = {}
    if title_items:
        for item in title_items:
            titles[item.key] = item.value.get("title", "Nova conversa")

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
    safe_thread_id = _build_thread_id(thread_id, str(user.user_id), account_id)

    async with checkpointer.conn.connection() as conn:
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
        await store.adelete(ns, thread_id)
    except Exception:
        pass  # Title may not exist yet

    return {"status": "deleted", "thread_id": thread_id}


@router.get("/health")
async def health():
    """Healthcheck do servico Agent API."""
    return {
        "status": "healthy",
        "service": "famachat-agent-api",
        "version": agent_settings.agent_version,
    }
