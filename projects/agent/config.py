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

    # Streaming
    sse_keepalive_interval: int = 15

    # Safety
    max_budget_change_pct: float = 50.0
    auto_approve_threshold: float = 0.0
    approval_token_secret: str = "change-me-in-env"

    # LangSmith
    langsmith_tracing: bool = False
    langsmith_project: str = "famachat-agent"

    class Config:
        env_prefix = "AGENT_"
        env_file = ".env"


agent_settings = AgentSettings()
