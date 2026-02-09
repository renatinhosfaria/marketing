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

    # Auth
    allow_unauthenticated: bool = Field(
        default=False,
        description="Permite acesso sem credencial (modo single-tenant controlado por rede)"
    )
    default_user_id: int = Field(
        default=1,
        ge=1,
        description="User ID padrão quando allow_unauthenticated=True"
    )
    trusted_proxy_user_header: str = Field(
        default="X-Agent-User-Id",
        description="Header de usuário injetado por proxy confiável"
    )
    trusted_proxy_secret_header: str = Field(
        default="X-Agent-Proxy-Secret",
        description="Header com segredo compartilhado do proxy"
    )
    trusted_proxy_secret: str | None = Field(
        default=None,
        description="Segredo esperado para aceitar trusted_proxy_user_header"
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
    rate_limit_enabled: bool = Field(
        default=True,
        description="Habilitar rate limiting na API"
    )
    rate_limit_per_minute: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Requisições por minuto por usuário"
    )
    rate_limit_per_hour: int = Field(
        default=200,
        ge=10,
        le=5000,
        description="Requisições por hora por usuário"
    )
    rate_limit_per_day: int = Field(
        default=500,
        ge=10,
        le=10000,
        description="Requisições por dia por usuário"
    )

    # Summarization Memory
    summarization_enabled: bool = Field(
        default=True,
        description="Habilitar sumarizacao automatica de conversas longas"
    )
    summarization_threshold: int = Field(
        default=20,
        ge=5,
        le=100,
        description="Numero de mensagens nao-system que triggera sumarizacao"
    )
    summarization_keep_recent: int = Field(
        default=10,
        ge=3,
        le=50,
        description="Quantas mensagens recentes manter intactas apos sumarizacao"
    )
    summarization_max_tokens: int = Field(
        default=600,
        ge=100,
        le=2000,
        description="Maximo de tokens para o sumario gerado"
    )

    # Vector Store / RAG
    vector_store_enabled: bool = Field(
        default=False,
        description="Habilitar armazenamento e busca vetorial"
    )
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="Modelo de embeddings (OpenAI)"
    )
    embedding_dimensions: int = Field(
        default=1536,
        description="Dimensoes do vetor de embedding"
    )
    rag_top_k: int = Field(
        default=3,
        description="Numero de resultados RAG a injetar no contexto"
    )
    rag_min_similarity: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="Similaridade minima para resultados RAG"
    )

    # Entity Memory
    entity_memory_enabled: bool = Field(
        default=False,
        description="Habilitar extracao e persistencia de entidades"
    )
    entity_max_per_user: int = Field(
        default=50,
        ge=10,
        le=200,
        description="Maximo de entidades por usuario"
    )

    # Cross-thread Memory
    cross_thread_enabled: bool = Field(
        default=False,
        description="Habilitar memoria entre threads diferentes do mesmo usuario"
    )
    cross_thread_max_results: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximo de resultados cross-thread a injetar"
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

    # Orchestrator
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

    # Tracing / Logging de conteúdo sensível
    log_full_prompts: bool = Field(
        default=False,
        description="Se True, registra prompt completo no tracing"
    )
    log_full_responses: bool = Field(
        default=False,
        description="Se True, registra resposta completa do LLM no tracing"
    )
    log_full_tool_data: bool = Field(
        default=False,
        description="Se True, registra payload completo de params/result de tools"
    )
    log_preview_chars: int = Field(
        default=500,
        ge=50,
        le=5000,
        description="Quantidade máxima de caracteres em previews de logs"
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
