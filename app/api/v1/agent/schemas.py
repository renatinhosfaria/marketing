"""
Schemas para a API do Agente de Tráfego Pago.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime


# ==========================================
# Requests
# ==========================================

class ChatRequest(BaseModel):
    """Request para enviar mensagem ao agente."""
    message: str = Field(..., min_length=1, max_length=4000, description="Mensagem do usuário")
    thread_id: Optional[str] = Field(None, description="ID da conversa (opcional para nova)")
    config_id: int = Field(..., description="ID da configuração Facebook Ads")

    class Config:
        json_schema_extra = {
            "example": {
                "message": "Como estão minhas campanhas hoje?",
                "config_id": 1
            }
        }


class AnalyzeRequest(BaseModel):
    """Request para análise rápida sem chat."""
    query: str = Field(..., min_length=1, max_length=4000, description="Pergunta ou análise solicitada")
    config_id: int = Field(..., description="ID da configuração Facebook Ads")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "Quais campanhas devo pausar?",
                "config_id": 1
            }
        }


class FeedbackRequest(BaseModel):
    """Request para enviar feedback sobre resposta do agente."""
    message_id: int = Field(..., description="ID da mensagem avaliada")
    rating: int = Field(..., ge=1, le=5, description="Nota de 1 a 5")
    feedback_text: Optional[str] = Field(None, max_length=1000, description="Comentário opcional")


class ClearConversationRequest(BaseModel):
    """Request para limpar uma conversa."""
    thread_id: str = Field(..., description="ID da conversa a limpar")


# ==========================================
# Responses
# ==========================================

class ChatResponse(BaseModel):
    """Response do chat com o agente."""
    success: bool = Field(..., description="Se a requisição foi bem-sucedida")
    thread_id: str = Field(..., description="ID da conversa")
    response: str = Field(..., description="Resposta do agente")
    intent: Optional[str] = Field(None, description="Intenção detectada")
    tool_calls_count: int = Field(default=0, description="Número de tools chamadas")
    error: Optional[str] = Field(None, description="Mensagem de erro se houver")

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "thread_id": "abc-123-def",
                "response": "Suas campanhas estão com bom desempenho...",
                "intent": "analyze",
                "tool_calls_count": 2
            }
        }


class AnalyzeResponse(BaseModel):
    """Resposta da análise rápida."""
    success: bool = Field(..., description="Se a requisição foi bem-sucedida")
    response: str = Field(..., description="Resposta do agente")
    intent: Optional[str] = Field(None, description="Intenção detectada")
    tool_calls_count: int = Field(default=0, description="Número de tools chamadas")
    error: Optional[str] = Field(None, description="Mensagem de erro se houver")


class SuggestionItem(BaseModel):
    """Sugestão de pergunta para o agente."""
    id: str
    text: str
    category: Optional[str] = None
    priority: int = 0


class StreamChunk(BaseModel):
    """Chunk de resposta em streaming."""
    type: str = Field(..., description="Tipo do chunk (text, tool_start, tool_end, done, error)")
    content: Optional[str] = Field(None, description="Conteúdo do chunk")
    tool: Optional[str] = Field(None, description="Nome da ferramenta (para tool_start/tool_end)")
    thread_id: str = Field(..., description="ID da conversa")
    error: Optional[str] = Field(None, description="Mensagem de erro")


class MessageResponse(BaseModel):
    """Resposta representando uma mensagem."""
    role: str = Field(..., description="Papel (user, assistant, tool)")
    content: str = Field(..., description="Conteúdo da mensagem")
    created_at: Optional[datetime] = Field(None, description="Data de criação")


class ConversationHistoryResponse(BaseModel):
    """Resposta com histórico da conversa."""
    thread_id: str = Field(..., description="ID da conversa")
    messages: List[MessageResponse] = Field(default_factory=list)
    message_count: int = Field(default=0, description="Total de mensagens")


class ConversationListItem(BaseModel):
    """Item da lista de conversas."""
    thread_id: str
    title: Optional[str] = None
    message_count: int = 0
    created_at: datetime
    updated_at: datetime


class ConversationListResponse(BaseModel):
    """Resposta com lista de conversas."""
    conversations: List[ConversationListItem] = Field(default_factory=list)
    total: int = Field(default=0)


class FeedbackResponse(BaseModel):
    """Resposta após envio de feedback."""
    success: bool
    message: str


class ClearConversationResponse(BaseModel):
    """Resposta após limpar conversa."""
    success: bool
    message: str


class AgentStatusResponse(BaseModel):
    """Status do agente."""
    status: str = Field(..., description="Status (online, offline, error)")
    llm_provider: str = Field(..., description="Provedor LLM configurado")
    model: str = Field(..., description="Modelo em uso")
    version: str = Field(default="1.0.0", description="Versão do agente")


class ErrorResponse(BaseModel):
    """Resposta de erro."""
    success: bool = Field(default=False)
    error: str = Field(..., description="Mensagem de erro")
    detail: Optional[str] = Field(None, description="Detalhes adicionais")


class SuggestionsResponse(BaseModel):
    """Resposta com sugestões proativas."""
    config_id: int
    suggestions: List[SuggestionItem] = Field(default_factory=list)


# ==========================================
# Multi-Agent Schemas
# ==========================================

class MultiAgentChatRequest(BaseModel):
    """Request para chat com sistema multi-agente."""
    message: str = Field(..., min_length=1, max_length=4000, description="Mensagem do usuário")
    thread_id: Optional[str] = Field(None, description="ID da conversa (opcional para nova)")
    config_id: int = Field(..., description="ID da configuração Facebook Ads")

    class Config:
        json_schema_extra = {
            "example": {
                "message": "Analise minhas campanhas e sugira otimizações",
                "config_id": 1
            }
        }


class AgentResultSchema(BaseModel):
    """Resultado de um subagente."""
    agent_name: str = Field(..., description="Nome do subagente")
    success: bool = Field(..., description="Se a execução foi bem-sucedida")
    data: Optional[Dict[str, Any]] = Field(None, description="Dados retornados pelo agente")
    error: Optional[str] = Field(None, description="Mensagem de erro se houver")
    duration_ms: int = Field(default=0, description="Duração da execução em milissegundos")
    tool_calls: List[str] = Field(default_factory=list, description="Lista de ferramentas chamadas")


class MultiAgentChatResponse(BaseModel):
    """Response do chat multi-agente."""
    success: bool = Field(..., description="Se a requisição foi bem-sucedida")
    thread_id: str = Field(..., description="ID da conversa")
    response: str = Field(..., description="Resposta sintetizada do orquestrador")
    confidence_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Score de confiança da resposta (0-1)"
    )
    intent: Optional[str] = Field(None, description="Intenção do usuário detectada")
    agents_used: List[str] = Field(
        default_factory=list,
        description="Lista de subagentes utilizados"
    )
    agent_results: Dict[str, Any] = Field(
        default_factory=dict,
        description="Resultados de cada subagente"
    )
    error: Optional[str] = Field(None, description="Mensagem de erro se houver")

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "thread_id": "abc-123-def",
                "response": "Análise completa das suas campanhas...",
                "confidence_score": 0.85,
                "intent": "analyze_campaigns",
                "agents_used": ["classification", "anomaly", "recommendation"],
                "agent_results": {
                    "classification": {"success": True, "data": {}},
                    "anomaly": {"success": True, "data": {}}
                }
            }
        }


class MultiAgentStatusResponse(BaseModel):
    """Status do sistema multi-agente."""
    status: str = Field(..., description="Status do sistema (online, offline, error)")
    mode: str = Field(..., description="Modo de operação (single ou multi)")
    available_agents: List[str] = Field(
        default_factory=list,
        description="Lista de subagentes disponíveis"
    )
    version: str = Field(default="1.0.0", description="Versão do sistema")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "online",
                "mode": "multi",
                "available_agents": [
                    "classification",
                    "anomaly",
                    "forecast",
                    "recommendation",
                    "campaign",
                    "analysis"
                ],
                "version": "1.0.0"
            }
        }


class AgentInfo(BaseModel):
    """Informações de um subagente."""
    name: str = Field(..., description="Nome identificador do subagente")
    description: str = Field(..., description="Descrição das capacidades do subagente")
    timeout: int = Field(default=30, description="Timeout em segundos")


class ListAgentsResponse(BaseModel):
    """Resposta com lista de subagentes disponíveis."""
    total: int = Field(..., description="Total de subagentes")
    agents: List[AgentInfo] = Field(
        default_factory=list,
        description="Lista de informações dos subagentes"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "total": 6,
                "agents": [
                    {
                        "name": "classification",
                        "description": "Analisa e classifica performance de campanhas em tiers",
                        "timeout": 30
                    }
                ]
            }
        }
