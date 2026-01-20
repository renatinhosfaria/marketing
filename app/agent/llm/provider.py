"""
Factory para criação de LLM (Claude/OpenAI).
"""

from typing import Optional
from enum import Enum

from langchain_core.language_models import BaseChatModel

from app.agent.config import agent_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class LLMProvider(str, Enum):
    """Provedores de LLM suportados."""
    ANTHROPIC = "anthropic"
    OPENAI = "openai"


def get_llm(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> BaseChatModel:
    """
    Cria instância do LLM baseado nas configurações.

    Args:
        provider: Provedor (anthropic/openai), usa config se não especificado
        model: Modelo a usar, usa config se não especificado
        temperature: Temperatura (0.0-1.0), usa config se não especificado
        max_tokens: Máximo de tokens, usa config se não especificado

    Returns:
        Instância do LLM configurado

    Raises:
        ValueError: Se provedor inválido ou API key não configurada
    """
    # Usar valores das configurações se não especificados
    provider = provider or agent_settings.llm_provider
    model = model or agent_settings.llm_model
    temperature = temperature if temperature is not None else agent_settings.temperature
    max_tokens = max_tokens or agent_settings.max_tokens

    logger.info(
        "Criando LLM",
        provider=provider,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens
    )

    if provider == LLMProvider.ANTHROPIC or provider == "anthropic":
        return _create_anthropic_llm(model, temperature, max_tokens)
    elif provider == LLMProvider.OPENAI or provider == "openai":
        return _create_openai_llm(model, temperature, max_tokens)
    else:
        raise ValueError(f"Provedor de LLM não suportado: {provider}")


def _create_anthropic_llm(
    model: str,
    temperature: float,
    max_tokens: int
) -> BaseChatModel:
    """Cria LLM da Anthropic (Claude)."""
    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError:
        raise ImportError(
            "langchain-anthropic não instalado. Execute: pip install langchain-anthropic"
        )

    api_key = agent_settings.anthropic_api_key
    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY não configurada. "
            "Configure via variável de ambiente ou arquivo .env"
        )

    return ChatAnthropic(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        anthropic_api_key=api_key,
    )


def _create_openai_llm(
    model: str,
    temperature: float,
    max_tokens: int
) -> BaseChatModel:
    """Cria LLM da OpenAI (GPT)."""
    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        raise ImportError(
            "langchain-openai não instalado. Execute: pip install langchain-openai"
        )

    api_key = agent_settings.openai_api_key
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY não configurada. "
            "Configure via variável de ambiente ou arquivo .env"
        )

    return ChatOpenAI(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        openai_api_key=api_key,
    )


def get_llm_with_tools(
    tools: list,
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> BaseChatModel:
    """
    Cria LLM com tools vinculadas.

    Args:
        tools: Lista de tools LangChain
        provider: Provedor de LLM
        model: Modelo a usar

    Returns:
        LLM com tools bound
    """
    llm = get_llm(provider=provider, model=model)
    return llm.bind_tools(tools)
