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

    # Multi-Agent System
    multi_agent_enabled: bool = Field(
        default=False,
        description="Habilitar sistema multi-agente"
    )
    orchestrator_timeout: int = Field(
        default=120,
        ge=30,
        le=600,
        description="Timeout total do orchestrator em segundos"
    )
    max_parallel_subagents: int = Field(
        default=4,
        ge=1,
        le=10,
        description="Máximo de subagentes em paralelo"
    )

    # Subagent Timeouts
    timeout_classification: int = Field(
        default=30,
        ge=5,
        le=120,
        description="Timeout ClassificationAgent em segundos"
    )
    timeout_anomaly: int = Field(
        default=30,
        ge=5,
        le=120,
        description="Timeout AnomalyAgent em segundos"
    )
    timeout_forecast: int = Field(
        default=45,
        ge=5,
        le=180,
        description="Timeout ForecastAgent em segundos"
    )
    timeout_recommendation: int = Field(
        default=30,
        ge=5,
        le=120,
        description="Timeout RecommendationAgent em segundos"
    )
    timeout_campaign: int = Field(
        default=20,
        ge=5,
        le=60,
        description="Timeout CampaignAgent em segundos"
    )
    timeout_analysis: int = Field(
        default=45,
        ge=5,
        le=180,
        description="Timeout AnalysisAgent em segundos"
    )

    # Synthesis
    synthesis_max_tokens: int = Field(
        default=4096,
        ge=256,
        le=8192,
        description="Máximo de tokens para síntese de respostas"
    )
    synthesis_temperature: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Temperature para síntese de respostas"
    )

    # Subagent Retry
    subagent_max_retries: int = Field(
        default=2,
        ge=0,
        le=5,
        description="Máximo de retries por subagente"
    )
    subagent_retry_delay: float = Field(
        default=1.0,
        ge=0.1,
        le=10.0,
        description="Delay entre retries em segundos"
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
