"""
Gerenciamento de LLMs por papel (role) com fallback automatico.

get_model(role, config): retorna o modelo adequado para o papel.

Roles disponiveis:
  - supervisor: gpt-4o-mini (rapido, roteamento)
  - analyst: gpt-4o (analise profunda)
  - synthesizer: gpt-4o (sintese de respostas)
  - operations: gpt-4o (operacoes de escrita)
"""

from langchain.chat_models import init_chat_model

from projects.agent.config import agent_settings


def get_model(role: str, config: dict = None):
    """Retorna o modelo para o papel.

    O modelo pode ser sobrescrito via config["configurable"]["{role}_model"].

    Args:
        role: Papel do modelo (supervisor, analyst, synthesizer, operations).
        config: RunnableConfig com configurable opcional.

    Returns:
        ChatModel configurado.
    """
    config = config or {}
    configurable = config.get("configurable", {})

    model_map = {
        "supervisor": configurable.get(
            "supervisor_model", agent_settings.supervisor_model
        ),
        "analyst": configurable.get(
            "analyst_model", agent_settings.analyst_model
        ),
        "synthesizer": configurable.get(
            "synthesizer_model", agent_settings.synthesizer_model
        ),
        "operations": configurable.get(
            "operations_model", agent_settings.operations_model
        ),
        "title_generator": configurable.get(
            "title_generator_model", agent_settings.title_generator_model
        ),
    }

    model_name = model_map.get(role, agent_settings.analyst_model)
    provider = agent_settings.default_provider

    # Timeout por role: supervisor precisa de resposta rapida
    timeout = (
        agent_settings.supervisor_timeout
        if role == "supervisor"
        else agent_settings.llm_timeout
    )

    # Streaming habilitado apenas para synthesizer (tokens fluem via SSE)
    streaming = role == "synthesizer"

    kwargs = {"timeout": timeout}
    if provider == "openai" and agent_settings.openai_api_key:
        kwargs["api_key"] = agent_settings.openai_api_key
    elif provider == "anthropic" and agent_settings.anthropic_api_key:
        kwargs["api_key"] = agent_settings.anthropic_api_key

    return init_chat_model(
        model_name, model_provider=provider, streaming=streaming, **kwargs
    )
