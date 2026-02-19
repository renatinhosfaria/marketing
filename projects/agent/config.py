"""
Configuracoes do modulo Agent (AI Agent Ecosystem).
Carrega variaveis de ambiente com prefixo AGENT_.
"""

from pydantic_settings import BaseSettings
from typing import Optional


class AgentSettings(BaseSettings):
    """Configuracoes do Agent carregadas do ambiente."""

    # LLM Models
    supervisor_model: str = "gpt-4o-mini"
    analyst_model: str = "gpt-4o"
    synthesizer_model: str = "gpt-4o"
    operations_model: str = "gpt-4o"
    title_generator_model: str = "gpt-4o-mini"
    default_provider: str = "openai"

    # API Keys
    anthropic_api_key: str = ""
    openai_api_key: Optional[str] = None

    # Versao
    agent_version: str = "1.0.0"

    # ML API
    ml_api_url: str = "http://marketing-api:8000"
    ml_api_timeout: int = 30

    # Memory (Store)
    store_embedding_model: str = "openai:text-embedding-3-small"
    store_embedding_dims: int = 1536

    # Supervisor
    supervisor_max_history_messages: int = 20
    agent_context_messages: int = 5  # Mensagens de historico enviadas aos agentes

    # LLM Timeouts (segundos)
    llm_timeout: int = 60  # Timeout padrao para chamadas LLM
    supervisor_timeout: int = 15  # Supervisor usa modelo leve — timeout curto

    # Streaming
    sse_keepalive_interval: int = 15

    # Auth
    api_key_hash: str = ""  # SHA-256 hex do AGENT_API_KEY. Vazio = auth desabilitada.
    require_auth: bool = True  # False permite requests sem key (dev local)
    runtime_user_id: str = "system"  # Identidade estavel no modo single-user.
    runtime_user_name: str = "System User"

    # Safety
    max_budget_change_pct: float = 50.0
    auto_approve_threshold: float = 0.0
    approval_token_secret: str = "change-me"

    # Rollout flags
    enable_strict_write_path: bool = True
    enable_ml_endpoint_fixes: bool = True
    enable_agent_jobs: bool = True

    # Rate limit do Agent API
    rate_limit_enabled: bool = True
    rate_limit_requests_per_minute: int = 120
    rate_limit_requests_per_hour: int = 3000
    rate_limit_burst: int = 20

    # LangSmith
    langsmith_tracing: bool = False
    langsmith_project: str = "famachat-agent"

    class Config:
        env_prefix = "AGENT_"
        env_file = ".env"


agent_settings = AgentSettings()
