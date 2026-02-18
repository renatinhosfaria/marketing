"""
Testes do provider de LLM.

Testa:
  - get_model: selecao de modelo por role
  - Provider default (openai) usado quando nao ha override
  - api_key passado quando configurado
  - Modelo default para role desconhecido
  - Override via config["configurable"]
"""

import pytest
from unittest.mock import patch, MagicMock


@pytest.mark.asyncio
async def test_get_model_supervisor_role():
    """get_model para role 'supervisor' usa modelo configurado."""
    mock_settings = MagicMock()
    mock_settings.supervisor_model = "gpt-4o-mini"
    mock_settings.analyst_model = "gpt-4o"
    mock_settings.synthesizer_model = "gpt-4o"
    mock_settings.operations_model = "gpt-4o"
    mock_settings.title_generator_model = "gpt-4o-mini"
    mock_settings.default_provider = "openai"
    mock_settings.openai_api_key = "sk-test"
    mock_settings.anthropic_api_key = ""

    mock_primary = MagicMock()

    with patch("projects.agent.llm.provider.agent_settings", mock_settings), \
         patch("projects.agent.llm.provider.init_chat_model", return_value=mock_primary) as mock_init:
        from projects.agent.llm.provider import get_model
        result = get_model("supervisor")

    mock_init.assert_called_once_with("gpt-4o-mini", model_provider="openai", api_key="sk-test")
    assert result == mock_primary


@pytest.mark.asyncio
async def test_get_model_analyst_role():
    """get_model para role 'analyst' usa modelo correto."""
    mock_settings = MagicMock()
    mock_settings.supervisor_model = "gpt-4o-mini"
    mock_settings.analyst_model = "gpt-4o"
    mock_settings.synthesizer_model = "gpt-4o"
    mock_settings.operations_model = "gpt-4o"
    mock_settings.title_generator_model = "gpt-4o-mini"
    mock_settings.default_provider = "openai"
    mock_settings.openai_api_key = None
    mock_settings.anthropic_api_key = ""

    mock_primary = MagicMock()

    with patch("projects.agent.llm.provider.agent_settings", mock_settings), \
         patch("projects.agent.llm.provider.init_chat_model", return_value=mock_primary):
        from projects.agent.llm.provider import get_model
        result = get_model("analyst")

    assert result == mock_primary


@pytest.mark.asyncio
async def test_get_model_with_anthropic_provider():
    """get_model com provider anthropic passa api_key quando configurado."""
    mock_settings = MagicMock()
    mock_settings.supervisor_model = "claude-3-5-haiku-latest"
    mock_settings.analyst_model = "claude-sonnet-4-5-20250929"
    mock_settings.synthesizer_model = "claude-sonnet-4-5-20250929"
    mock_settings.operations_model = "claude-sonnet-4-5-20250929"
    mock_settings.title_generator_model = "claude-3-5-haiku-latest"
    mock_settings.default_provider = "anthropic"
    mock_settings.openai_api_key = None
    mock_settings.anthropic_api_key = "sk-ant-test"

    mock_primary = MagicMock()

    with patch("projects.agent.llm.provider.agent_settings", mock_settings), \
         patch("projects.agent.llm.provider.init_chat_model", return_value=mock_primary) as mock_init:
        from projects.agent.llm.provider import get_model
        result = get_model("supervisor")

    mock_init.assert_called_once_with(
        "claude-3-5-haiku-latest", model_provider="anthropic", api_key="sk-ant-test",
    )
    assert result == mock_primary


@pytest.mark.asyncio
async def test_get_model_without_api_key():
    """get_model sem api_key nao passa kwarg extra."""
    mock_settings = MagicMock()
    mock_settings.supervisor_model = "gpt-4o-mini"
    mock_settings.analyst_model = "gpt-4o"
    mock_settings.synthesizer_model = "gpt-4o"
    mock_settings.operations_model = "gpt-4o"
    mock_settings.title_generator_model = "gpt-4o-mini"
    mock_settings.default_provider = "openai"
    mock_settings.openai_api_key = None
    mock_settings.anthropic_api_key = ""

    mock_primary = MagicMock()

    with patch("projects.agent.llm.provider.agent_settings", mock_settings), \
         patch("projects.agent.llm.provider.init_chat_model", return_value=mock_primary) as mock_init:
        from projects.agent.llm.provider import get_model
        result = get_model("analyst")

    mock_init.assert_called_once_with("gpt-4o", model_provider="openai")
    mock_primary.with_fallbacks.assert_not_called()
    assert result == mock_primary


@pytest.mark.asyncio
async def test_get_model_unknown_role_uses_analyst_default():
    """get_model com role desconhecido usa analyst_model como default."""
    mock_settings = MagicMock()
    mock_settings.supervisor_model = "gpt-4o-mini"
    mock_settings.analyst_model = "gpt-4o"
    mock_settings.synthesizer_model = "gpt-4o"
    mock_settings.operations_model = "gpt-4o"
    mock_settings.title_generator_model = "gpt-4o-mini"
    mock_settings.default_provider = "openai"
    mock_settings.openai_api_key = None
    mock_settings.anthropic_api_key = ""

    mock_primary = MagicMock()

    with patch("projects.agent.llm.provider.agent_settings", mock_settings), \
         patch("projects.agent.llm.provider.init_chat_model", return_value=mock_primary) as mock_init:
        from projects.agent.llm.provider import get_model
        get_model("unknown_role")

    mock_init.assert_called_once_with("gpt-4o", model_provider="openai")


@pytest.mark.asyncio
async def test_get_model_config_override():
    """get_model permite override do modelo via config."""
    mock_settings = MagicMock()
    mock_settings.supervisor_model = "gpt-4o-mini"
    mock_settings.analyst_model = "gpt-4o"
    mock_settings.synthesizer_model = "gpt-4o"
    mock_settings.operations_model = "gpt-4o"
    mock_settings.title_generator_model = "gpt-4o-mini"
    mock_settings.default_provider = "openai"
    mock_settings.openai_api_key = None
    mock_settings.anthropic_api_key = ""

    mock_primary = MagicMock()

    config = {"configurable": {"supervisor_model": "gpt-4-turbo"}}

    with patch("projects.agent.llm.provider.agent_settings", mock_settings), \
         patch("projects.agent.llm.provider.init_chat_model", return_value=mock_primary) as mock_init:
        from projects.agent.llm.provider import get_model
        get_model("supervisor", config)

    mock_init.assert_called_once_with("gpt-4-turbo", model_provider="openai")
