"""
Configurações do Agente de IA.
"""

from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings
from typing import Literal


class AgentSettings(BaseSettings):
    """Configurações do agente de tráfego pago."""

    # LLM Provider
    llm_provider: Literal["anthropic", "openai"] = Field(
        default="anthropic",
        description="Provedor do LLM (anthropic ou openai)"
    )
    llm_model: str = Field(
        default="claude-sonnet-4-20250514",
        description="Modelo do LLM a ser usado"
    )
    temperature: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Temperatura do LLM (0.0-1.0)"
    )
    max_tokens: int = Field(
        default=4096,
        ge=100,
        le=8192,
        description="Máximo de tokens por resposta"
    )

    # API Keys
    anthropic_api_key: str | None = Field(
        default=None,
        description="API Key da Anthropic"
    )
    openai_api_key: str | None = Field(
        default=None,
        description="API Key da OpenAI"
    )

    # Performance
    timeout_seconds: int = Field(
        default=60,
        ge=10,
        le=300,
        description="Timeout para requisições em segundos"
    )
    max_tool_calls: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Máximo de tool calls por turno"
    )
    max_conversation_messages: int = Field(
        default=100,
        ge=10,
        le=500,
        description="Máximo de mensagens por conversa"
    )

    # Rate Limiting
    rate_limit_per_minute: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Requisições por minuto por usuário"
    )
    rate_limit_per_day: int = Field(
        default=500,
        ge=10,
        le=10000,
        description="Requisições por dia por usuário"
    )

    # Persistência
    checkpoint_enabled: bool = Field(
        default=True,
        description="Habilitar checkpoints para persistência"
    )
    conversation_ttl_days: int = Field(
        default=30,
        ge=1,
        le=365,
        description="TTL de conversas em dias"
    )

    class Config:
        env_prefix = "AGENT_"
        env_file = ".env"
        extra = "ignore"


@lru_cache()
def get_agent_settings() -> AgentSettings:
    """
    Retorna instância cacheada das configurações do agente.
    """
    return AgentSettings()


# Instância global das configurações
agent_settings = get_agent_settings()
