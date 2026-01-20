"""
Router da API do Agente de Tráfego Pago.
"""

from typing import AsyncGenerator
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
import json

from app.api.v1.agent.schemas import (
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
)
from app.agent import get_agent_service, get_agent_settings
from app.db.session import get_db, async_session_maker
from app.db.models.agent_models import AgentConversation, AgentMessage, AgentFeedback, MessageRole
from app.core.security import get_current_user
from app.core.logging import get_logger
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
    current_user: dict = Depends(get_current_user),
):
    """
    Processa uma mensagem do usuário e retorna a resposta do agente.
    """
    try:
        agent_service = await get_agent_service()

        result = await agent_service.chat(
            message=request.message,
            config_id=request.config_id,
            user_id=current_user["id"],
            thread_id=request.thread_id,
        )

        # Salvar conversa e mensagens no banco
        await _save_conversation(
            db=db,
            thread_id=result["thread_id"],
            config_id=request.config_id,
            user_id=current_user["id"],
            user_message=request.message,
            assistant_message=result.get("response", ""),
        )

        return ChatResponse(**result)

    except Exception as e:
        logger.error(f"Erro no chat: {e}")
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
    current_user: dict = Depends(get_current_user),
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
                user_id=current_user["id"],
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
                            user_id=current_user["id"],
                            user_message=request.message,
                            assistant_message=full_response,
                        )
                except Exception as save_error:
                    logger.error(f"Erro ao salvar conversa: {save_error}")

        except Exception as e:
            logger.error(f"Erro no streaming: {e}")
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
    current_user: dict = Depends(get_current_user),
):
    """
    Executa uma análise rápida usando o agente sem salvar conversa.
    """
    try:
        agent_service = await get_agent_service()

        result = await agent_service.chat(
            message=request.query,
            config_id=request.config_id,
            user_id=current_user["id"],
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
        logger.error(f"Erro no analyze: {e}")
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
    current_user: dict = Depends(get_current_user),
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
    current_user: dict = Depends(get_current_user),
):
    """
    Lista conversas do usuário ordenadas por data de atualização.
    """
    try:
        # Contar total
        count_query = select(func.count()).select_from(AgentConversation).where(
            AgentConversation.user_id == current_user["id"],
            AgentConversation.config_id == config_id,
        )
        total_result = await db.execute(count_query)
        total = total_result.scalar() or 0

        # Buscar conversas
        query = (
            select(AgentConversation)
            .where(
                AgentConversation.user_id == current_user["id"],
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
        logger.error(f"Erro ao listar conversas: {e}")
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
    current_user: dict = Depends(get_current_user),
):
    """
    Retorna o histórico completo de uma conversa.
    """
    try:
        # Verificar se a conversa pertence ao usuário
        conv_query = select(AgentConversation).where(
            AgentConversation.thread_id == thread_id,
            AgentConversation.user_id == current_user["id"],
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
        logger.error(f"Erro ao obter histórico: {e}")
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
    current_user: dict = Depends(get_current_user),
):
    """
    Limpa o histórico de uma conversa.
    """
    try:
        # Verificar se a conversa pertence ao usuário
        conv_query = select(AgentConversation).where(
            AgentConversation.thread_id == thread_id,
            AgentConversation.user_id == current_user["id"],
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
        logger.error(f"Erro ao limpar conversa: {e}")
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
    current_user: dict = Depends(get_current_user),
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
                AgentConversation.user_id == current_user["id"],
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
            user_id=current_user["id"],
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
        logger.error(f"Erro ao registrar feedback: {e}")
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
        logger.error(f"Erro ao obter status: {e}")
        return AgentStatusResponse(
            status="error",
            llm_provider="unknown",
            model="unknown",
            version="1.0.0",
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
        logger.error(f"Erro ao salvar conversa: {e}")
        await db.rollback()
