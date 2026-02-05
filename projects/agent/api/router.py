"""
Router da API do Agente de Tráfego Pago.
"""

from typing import AsyncGenerator
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
import json
import uuid

from projects.agent.api.schemas import (
    ChatRequest,
    ChatResponse,
    AnalyzeRequest,
    AnalyzeResponse,
    SuggestionsResponse,
    SuggestionItem,
    FeedbackRequest,
    FeedbackResponse,
    ClearConversationRequest,
    ClearConversationResponse,
    ConversationHistoryResponse,
    ConversationListResponse,
    ConversationListItem,
    MessageResponse,
    AgentStatusResponse,
    ErrorResponse,
    # Multi-Agent schemas
    MultiAgentChatRequest,
    MultiAgentChatResponse,
    MultiAgentStatusResponse,
    AgentInfo,
    ListAgentsResponse,
    SubagentInfo,
    SubagentStatusResponse,
    SubagentsListResponse,
    AgentResultDetail,
    ChatDetailedResponse,
)
from projects.agent import get_agent_service, get_agent_settings
from projects.agent.service import get_multi_agent_service
from projects.agent.subagents import SUBAGENT_REGISTRY
from shared.db.session import get_db, async_session_maker
from projects.agent.db.models import AgentConversation, AgentMessage, AgentFeedback, MessageRole
from shared.core.logging import get_logger
from sqlalchemy import select, desc, func


logger = get_logger(__name__)
router = APIRouter(prefix="/agent", tags=["Agent"])


# ==========================================
# Chat Endpoints
# ==========================================

@router.post(
    "/chat",
    response_model=ChatResponse,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
    summary="Enviar mensagem ao agente",
    description="Envia uma mensagem ao agente e recebe a resposta completa."
)
async def chat(
    request: ChatRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Processa uma mensagem do usuário e retorna a resposta do agente.
    """
    try:
        agent_service = await get_agent_service()

        result = await agent_service.chat(
            message=request.message,
            config_id=request.config_id,
            user_id=1,
            thread_id=request.thread_id,
        )

        # Salvar conversa e mensagens no banco
        await _save_conversation(
            db=db,
            thread_id=result["thread_id"],
            config_id=request.config_id,
            user_id=1,
            user_message=request.message,
            assistant_message=result.get("response", ""),
        )

        return ChatResponse(**result)

    except Exception as e:
        logger.error("Erro no chat", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post(
    "/chat/stream",
    summary="Chat com streaming",
    description="Envia mensagem e recebe resposta em streaming (SSE)."
)
async def chat_stream(
    request: ChatRequest,
):
    """
    Processa uma mensagem com streaming de resposta via Server-Sent Events.
    Nota: Não usa db dependency para evitar timeout durante streaming longo.
    """
    async def generate() -> AsyncGenerator[str, None]:
        try:
            agent_service = await get_agent_service()
            full_response = ""
            final_thread_id = ""

            async for chunk in agent_service.stream_chat(
                message=request.message,
                config_id=request.config_id,
                user_id=1,
                thread_id=request.thread_id,
            ):
                # Acumular resposta para salvar depois
                if chunk.get("type") == "text":
                    full_response += chunk.get("content", "")

                # Capturar thread_id
                if chunk.get("thread_id"):
                    final_thread_id = chunk.get("thread_id")

                # Enviar como SSE
                yield f"data: {json.dumps(chunk)}\n\n"

            # Salvar conversa após streaming completo usando sessão nova
            if full_response and final_thread_id:
                try:
                    async with async_session_maker() as db:
                        await _save_conversation(
                            db=db,
                            thread_id=final_thread_id,
                            config_id=request.config_id,
                            user_id=1,
                            user_message=request.message,
                            assistant_message=full_response,
                        )
                except Exception as save_error:
                    logger.error("Erro ao salvar conversa", error=str(save_error))

        except Exception as e:
            logger.error("Erro no streaming", error=str(e))
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


# ==========================================
# Analysis & Suggestions
# ==========================================

@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
    summary="Análise rápida",
    description="Executa uma análise rápida sem persistir a conversa."
)
async def analyze(
    request: AnalyzeRequest,
):
    """
    Executa uma análise rápida usando o agente sem salvar conversa.
    """
    try:
        agent_service = await get_agent_service()

        result = await agent_service.chat(
            message=request.query,
            config_id=request.config_id,
            user_id=1,
            thread_id=None,
        )

        thread_id = result.get("thread_id")
        if thread_id:
            await agent_service.clear_conversation(thread_id)

        return AnalyzeResponse(
            success=result.get("success", False),
            response=result.get("response", ""),
            intent=result.get("intent"),
            tool_calls_count=result.get("tool_calls_count", 0),
            error=result.get("error"),
        )

    except Exception as e:
        logger.error("Erro no analyze", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get(
    "/suggestions/{config_id}",
    response_model=SuggestionsResponse,
    summary="Sugestões proativas",
    description="Retorna uma lista de sugestões de perguntas para o agente."
)
async def get_suggestions(
    config_id: int,
):
    """
    Retorna sugestões de perguntas para o usuário.
    """
    suggestions = [
        SuggestionItem(
            id="best_cpl",
            text="Qual campanha está com o melhor CPL?",
            category="performance",
            priority=5,
        ),
        SuggestionItem(
            id="critical_anomalies",
            text="Mostre anomalias críticas de hoje",
            category="anomalies",
            priority=4,
        ),
        SuggestionItem(
            id="recommendations",
            text="Quais recomendações devo seguir?",
            category="recommendations",
            priority=4,
        ),
        SuggestionItem(
            id="compare_top_campaigns",
            text="Compare minhas top 3 campanhas",
            category="comparison",
            priority=3,
        ),
        SuggestionItem(
            id="forecast_leads",
            text="Previsão de leads para próxima semana",
            category="forecast",
            priority=3,
        ),
        SuggestionItem(
            id="pause_campaigns",
            text="Qual campanha devo pausar?",
            category="actions",
            priority=3,
        ),
    ]

    return SuggestionsResponse(
        config_id=config_id,
        suggestions=suggestions,
    )


# ==========================================
# Conversation Endpoints
# ==========================================

@router.get(
    "/conversations",
    response_model=ConversationListResponse,
    summary="Listar conversas",
    description="Lista todas as conversas do usuário para uma configuração."
)
async def list_conversations(
    config_id: int,
    limit: int = 20,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
):
    """
    Lista conversas do usuário ordenadas por data de atualização.
    """
    try:
        # Contar total
        count_query = select(func.count()).select_from(AgentConversation).where(
            AgentConversation.user_id == 1,
            AgentConversation.config_id == config_id,
        )
        total_result = await db.execute(count_query)
        total = total_result.scalar() or 0

        # Buscar conversas
        query = (
            select(AgentConversation)
            .where(
                AgentConversation.user_id == 1,
                AgentConversation.config_id == config_id,
            )
            .order_by(desc(AgentConversation.updated_at))
            .offset(offset)
            .limit(limit)
        )

        result = await db.execute(query)
        conversations = result.scalars().all()

        return ConversationListResponse(
            conversations=[
                ConversationListItem(
                    thread_id=c.thread_id,
                    title=c.title,
                    message_count=c.message_count,
                    created_at=c.created_at,
                    updated_at=c.updated_at,
                )
                for c in conversations
            ],
            total=total
        )

    except Exception as e:
        logger.error("Erro ao listar conversas", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get(
    "/conversations/{thread_id}",
    response_model=ConversationHistoryResponse,
    summary="Obter histórico da conversa",
    description="Retorna todas as mensagens de uma conversa."
)
async def get_conversation_history(
    thread_id: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Retorna o histórico completo de uma conversa.
    """
    try:
        # Verificar se a conversa pertence ao usuário
        conv_query = select(AgentConversation).where(
            AgentConversation.thread_id == thread_id,
            AgentConversation.user_id == 1,
        )
        conv_result = await db.execute(conv_query)
        conversation = conv_result.scalar_one_or_none()

        if not conversation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversa não encontrada"
            )

        # Buscar mensagens
        msg_query = (
            select(AgentMessage)
            .where(AgentMessage.conversation_id == conversation.id)
            .order_by(AgentMessage.created_at)
        )
        msg_result = await db.execute(msg_query)
        messages = msg_result.scalars().all()

        return ConversationHistoryResponse(
            thread_id=thread_id,
            messages=[
                MessageResponse(
                    role=m.role.value,
                    content=m.content,
                    created_at=m.created_at,
                )
                for m in messages
            ],
            message_count=len(messages),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Erro ao obter histórico", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.delete(
    "/conversations/{thread_id}",
    response_model=ClearConversationResponse,
    summary="Limpar conversa",
    description="Remove todo o histórico de uma conversa."
)
async def clear_conversation(
    thread_id: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Limpa o histórico de uma conversa.
    """
    try:
        # Verificar se a conversa pertence ao usuário
        conv_query = select(AgentConversation).where(
            AgentConversation.thread_id == thread_id,
            AgentConversation.user_id == 1,
        )
        conv_result = await db.execute(conv_query)
        conversation = conv_result.scalar_one_or_none()

        if not conversation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversa não encontrada"
            )

        # Deletar conversa (cascade deleta mensagens)
        await db.delete(conversation)
        await db.commit()

        # Limpar checkpoints do agente
        agent_service = await get_agent_service()
        await agent_service.clear_conversation(thread_id)

        return ClearConversationResponse(
            success=True,
            message="Conversa limpa com sucesso"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Erro ao limpar conversa", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# ==========================================
# Feedback Endpoints
# ==========================================

@router.post(
    "/feedback",
    response_model=FeedbackResponse,
    summary="Enviar feedback",
    description="Envia feedback sobre uma resposta do agente."
)
async def submit_feedback(
    request: FeedbackRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Registra feedback do usuário sobre uma resposta.
    """
    try:
        # Verificar se a mensagem existe e pertence ao usuário
        msg_query = (
            select(AgentMessage)
            .join(AgentConversation)
            .where(
                AgentMessage.id == request.message_id,
                AgentConversation.user_id == 1,
            )
        )
        msg_result = await db.execute(msg_query)
        message = msg_result.scalar_one_or_none()

        if not message:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Mensagem não encontrada"
            )

        # Criar ou atualizar feedback
        feedback = AgentFeedback(
            message_id=request.message_id,
            user_id=1,
            rating=request.rating,
            feedback_text=request.feedback_text,
        )

        db.add(feedback)
        await db.commit()

        return FeedbackResponse(
            success=True,
            message="Feedback registrado com sucesso"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Erro ao registrar feedback", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# ==========================================
# Status Endpoints
# ==========================================

@router.get(
    "/status",
    response_model=AgentStatusResponse,
    summary="Status do agente",
    description="Retorna o status atual do agente."
)
async def get_agent_status():
    """
    Retorna informações de status do agente.
    """
    try:
        settings = get_agent_settings()

        return AgentStatusResponse(
            status="online",
            llm_provider=settings.llm_provider,
            model=settings.llm_model,
            version="1.0.0",
        )

    except Exception as e:
        logger.error("Erro ao obter status", error=str(e))
        return AgentStatusResponse(
            status="error",
            llm_provider="unknown",
            model="unknown",
            version="1.0.0",
        )


# ==========================================
# Multi-Agent Endpoints
# ==========================================

@router.post(
    "/multi-agent/chat",
    response_model=MultiAgentChatResponse,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
    summary="Chat com sistema multi-agente",
    description="Envia mensagem usando o sistema multi-agente orquestrado."
)
async def chat_multi_agent(
    request: MultiAgentChatRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Processa mensagem usando múltiplos agentes especializados.

    O sistema multi-agente utiliza um orquestrador que coordena vários
    subagentes especializados para fornecer respostas mais completas.

    Subagentes disponíveis:
    - classification: Análise de tiers de performance
    - anomaly: Detecção de problemas e alertas
    - forecast: Previsões de CPL/Leads
    - recommendation: Recomendações de ações
    - campaign: Dados de campanhas
    - analysis: Análises avançadas
    """
    try:
        multi_agent_service = await get_multi_agent_service()

        result = await multi_agent_service.chat_multi_agent(
            message=request.message,
            config_id=request.config_id,
            user_id=1,
            thread_id=request.thread_id,
        )

        # Salvar conversa e mensagens no banco
        await _save_conversation(
            db=db,
            thread_id=result["thread_id"],
            config_id=request.config_id,
            user_id=1,
            user_message=request.message,
            assistant_message=result.get("response", ""),
        )

        return MultiAgentChatResponse(
            success=result.get("success", False),
            thread_id=result.get("thread_id", ""),
            response=result.get("response", ""),
            confidence_score=result.get("confidence_score", 0.0),
            intent=result.get("intent"),
            agents_used=result.get("agents_used", []),
            agent_results=result.get("agent_results", {}),
            error=result.get("error"),
        )

    except Exception as e:
        logger.error("Erro no chat multi-agente", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post(
    "/multi-agent/chat/stream",
    summary="Chat multi-agente com streaming",
    description="Envia mensagem e recebe eventos em streaming (SSE) usando sistema multi-agente."
)
async def chat_multi_agent_stream(
    request: MultiAgentChatRequest,
):
    """
    Processa mensagem com streaming usando sistema multi-agente.

    Emite eventos SSE para acompanhamento em tempo real:
    - orchestrator_start: Início do processamento
    - intent_detected: Quando a intenção é identificada
    - plan_created: Quando o plano de execução é criado
    - agent_start: Quando um subagente começa
    - agent_end: Quando um subagente termina
    - synthesis_start: Início da síntese
    - text: Chunks da resposta sintetizada
    - done: Finalização com metadados
    - error: Em caso de erro
    """
    async def generate() -> AsyncGenerator[str, None]:
        try:
            multi_agent_service = await get_multi_agent_service()
            full_response = ""
            final_thread_id = ""

            async for event in multi_agent_service.stream_chat_multi_agent(
                message=request.message,
                config_id=request.config_id,
                user_id=1,
                thread_id=request.thread_id,
            ):
                # Acumular resposta para salvar depois
                if event.get("type") == "text":
                    full_response += event.get("content", "")

                # Capturar thread_id
                if event.get("thread_id"):
                    final_thread_id = event.get("thread_id")

                # Enviar como SSE
                yield f"data: {json.dumps(event)}\n\n"

            # Salvar conversa após streaming completo usando sessão nova
            if full_response and final_thread_id:
                try:
                    async with async_session_maker() as db:
                        await _save_conversation(
                            db=db,
                            thread_id=final_thread_id,
                            config_id=request.config_id,
                            user_id=1,
                            user_message=request.message,
                            assistant_message=full_response,
                        )
                except Exception as save_error:
                    logger.error("Erro ao salvar conversa multi-agente", error=str(save_error))

        except Exception as e:
            logger.error("Erro no streaming multi-agente", error=str(e))
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


@router.get(
    "/multi-agent/status",
    response_model=MultiAgentStatusResponse,
    summary="Status do sistema multi-agente",
    description="Retorna status e agentes disponíveis do sistema multi-agente."
)
async def get_multi_agent_status():
    """
    Retorna informações do sistema multi-agente.

    Inclui:
    - Status de operação (online/offline/error)
    - Modo atual (multi)
    - Lista de subagentes disponíveis
    - Versão do sistema
    """
    try:
        # Lista de agentes disponíveis do registry
        available_agents = list(SUBAGENT_REGISTRY.keys())

        return MultiAgentStatusResponse(
            status="online",
            mode="multi",
            available_agents=available_agents,
            version="1.0.0",
        )

    except Exception as e:
        logger.error("Erro ao obter status multi-agente", error=str(e))
        return MultiAgentStatusResponse(
            status="error",
            mode="multi",
            available_agents=[],
            version="1.0.0",
        )


@router.get(
    "/multi-agent/agents",
    response_model=ListAgentsResponse,
    summary="Lista de subagentes",
    description="Retorna lista detalhada de subagentes disponíveis com descrições."
)
async def list_available_agents():
    """
    Lista todos os subagentes disponíveis com suas descrições.

    Cada subagente tem:
    - name: Nome identificador único
    - description: Descrição das capacidades
    - timeout: Timeout de execução em segundos
    """
    try:
        agents_info = []

        for name, agent_cls in SUBAGENT_REGISTRY.items():
            # Instanciar para obter descrição e timeout
            agent_instance = agent_cls()

            agents_info.append(AgentInfo(
                name=name,
                description=getattr(agent_instance, "AGENT_DESCRIPTION", "Sem descrição"),
                timeout=agent_instance.get_timeout(),
            ))

        return ListAgentsResponse(
            total=len(agents_info),
            agents=agents_info,
        )

    except Exception as e:
        logger.error("Erro ao listar agentes", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# ==========================================
# Subagent Endpoints
# ==========================================

@router.get("/subagents", response_model=SubagentsListResponse)
async def list_subagents():
    """Lista todos os subagentes disponíveis."""
    from projects.agent.subagents import SUBAGENT_REGISTRY
    from projects.agent.config import get_agent_settings

    settings = get_agent_settings()

    subagents = []
    for name, cls in SUBAGENT_REGISTRY.items():
        agent = cls()
        subagents.append(SubagentInfo(
            name=name,
            description=agent.AGENT_DESCRIPTION,
            tools_count=len(agent.get_tools()),
            timeout=agent.get_timeout()
        ))

    return SubagentsListResponse(
        subagents=subagents,
        total=len(subagents),
        multi_agent_enabled=settings.multi_agent_enabled
    )


@router.get("/subagents/{name}/status", response_model=SubagentStatusResponse)
async def get_subagent_status(
    name: str,
):
    """Retorna status de um subagente específico."""
    from projects.agent.subagents import SUBAGENT_REGISTRY

    if name not in SUBAGENT_REGISTRY:
        raise HTTPException(
            status_code=404,
            detail=f"Subagente '{name}' não encontrado"
        )

    return SubagentStatusResponse(
        name=name,
        status="ready",
        last_execution_ms=None,
        total_executions=0,
        success_rate=1.0
    )


@router.post("/chat/detailed", response_model=ChatDetailedResponse)
async def chat_detailed(
    request: ChatRequest,
    db: AsyncSession = Depends(get_db),
):
    """Chat com resposta detalhada incluindo info de subagentes."""
    from projects.agent.service import get_agent_service, should_use_multiagent
    import time

    if not should_use_multiagent():
        raise HTTPException(
            status_code=400,
            detail="Multi-agent system não está habilitado"
        )

    start_time = time.time()

    try:
        service = await get_agent_service()
        result = await service._chat_multiagent(
            message=request.message,
            config_id=request.config_id,
            user_id=1,
            thread_id=request.thread_id or str(uuid.uuid4()),
            db=db
        )

        duration_ms = int((time.time() - start_time) * 1000)

        # Formatar resultados dos agentes
        agent_details = []
        for name, res in result.get("agent_results", {}).items():
            if isinstance(res, dict):
                agent_details.append(AgentResultDetail(
                    agent_name=name,
                    success=res.get("success", False),
                    duration_ms=res.get("duration_ms", 0),
                    tool_calls=res.get("tool_calls", []),
                    error=res.get("error")
                ))

        return ChatDetailedResponse(
            success=True,
            thread_id=result.get("thread_id", ""),
            response=result.get("response", ""),
            intent=result.get("intent", "general"),
            confidence_score=result.get("confidence", 0.0),
            agent_results=agent_details,
            total_duration_ms=duration_ms
        )

    except Exception as e:
        logger.error("Erro no chat detailed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ==========================================
# Helper Functions
# ==========================================

async def _save_conversation(
    db: AsyncSession,
    thread_id: str,
    config_id: int,
    user_id: int,
    user_message: str,
    assistant_message: str,
):
    """
    Salva ou atualiza conversa e mensagens no banco.
    """
    try:
        # Buscar ou criar conversa
        conv_query = select(AgentConversation).where(
            AgentConversation.thread_id == thread_id
        )
        conv_result = await db.execute(conv_query)
        conversation = conv_result.scalar_one_or_none()

        if not conversation:
            # Criar nova conversa
            # Usar primeiras palavras da mensagem como título
            title = user_message[:50] + "..." if len(user_message) > 50 else user_message

            conversation = AgentConversation(
                thread_id=thread_id,
                config_id=config_id,
                user_id=user_id,
                title=title,
                message_count=0,
            )
            db.add(conversation)
            await db.flush()

        # Adicionar mensagem do usuário
        user_msg = AgentMessage(
            conversation_id=conversation.id,
            role=MessageRole.USER,
            content=user_message,
        )
        db.add(user_msg)

        # Adicionar mensagem do assistente
        assistant_msg = AgentMessage(
            conversation_id=conversation.id,
            role=MessageRole.ASSISTANT,
            content=assistant_message,
        )
        db.add(assistant_msg)

        # Atualizar contador
        conversation.message_count += 2

        await db.commit()

    except Exception as e:
        logger.error("Erro ao salvar conversa", error=str(e))
        await db.rollback()
