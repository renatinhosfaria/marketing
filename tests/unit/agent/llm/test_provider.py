"""
Testes do provider de LLM.

Testa:
  - get_model: selecao de modelo por role
  - Provider default (openai) usado quando nao ha override
  - api_key passado quando configurado
  - Modelo default para role desconhecido
  - Override via config["configurable"]
  - Timeout por role (supervisor=15s, demais=60s)
  - Streaming habilitado apenas para synthesizer
"""

import pytest
from unittest.mock import patch, MagicMock


def _make_settings(**overrides):
    """Cria mock de AgentSettings com defaults padrao."""
    s = MagicMock()
    s.supervisor_model = "gpt-4o-mini"
    s.analyst_model = "gpt-4o"
    s.synthesizer_model = "gpt-4o"
    s.operations_model = "gpt-4o"
    s.title_generator_model = "gpt-4o-mini"
    s.default_provider = "openai"
    s.openai_api_key = None
    s.anthropic_api_key = ""
    s.llm_timeout = 60
    s.supervisor_timeout = 15
    for k, v in overrides.items():
        setattr(s, k, v)
    return s


@pytest.mark.asyncio
async def test_get_model_supervisor_role():
    """get_model para role 'supervisor' usa modelo configurado com timeout curto."""
    mock_settings = _make_settings(openai_api_key="sk-test")
    mock_primary = MagicMock()

    with patch("projects.agent.llm.provider.agent_settings", mock_settings), \
         patch("projects.agent.llm.provider.init_chat_model", return_value=mock_primary) as mock_init:
        from projects.agent.llm.provider import get_model
        result = get_model("supervisor")

    mock_init.assert_called_once_with(
        "gpt-4o-mini", model_provider="openai", streaming=False,
        timeout=15, api_key="sk-test",
    )
    assert result == mock_primary


@pytest.mark.asyncio
async def test_get_model_analyst_role():
    """get_model para role 'analyst' usa timeout padrao e streaming=False."""
    mock_settings = _make_settings()
    mock_primary = MagicMock()

    with patch("projects.agent.llm.provider.agent_settings", mock_settings), \
         patch("projects.agent.llm.provider.init_chat_model", return_value=mock_primary) as mock_init:
        from projects.agent.llm.provider import get_model
        result = get_model("analyst")

    mock_init.assert_called_once_with(
        "gpt-4o", model_provider="openai", streaming=False, timeout=60,
    )
    assert result == mock_primary


@pytest.mark.asyncio
async def test_get_model_with_anthropic_provider():
    """get_model com provider anthropic passa api_key e timeout."""
    mock_settings = _make_settings(
        supervisor_model="claude-3-5-haiku-latest",
        analyst_model="claude-sonnet-4-5-20250929",
        synthesizer_model="claude-sonnet-4-5-20250929",
        operations_model="claude-sonnet-4-5-20250929",
        title_generator_model="claude-3-5-haiku-latest",
        default_provider="anthropic",
        anthropic_api_key="sk-ant-test",
    )
    mock_primary = MagicMock()

    with patch("projects.agent.llm.provider.agent_settings", mock_settings), \
         patch("projects.agent.llm.provider.init_chat_model", return_value=mock_primary) as mock_init:
        from projects.agent.llm.provider import get_model
        result = get_model("supervisor")

    mock_init.assert_called_once_with(
        "claude-3-5-haiku-latest", model_provider="anthropic", streaming=False,
        timeout=15, api_key="sk-ant-test",
    )
    assert result == mock_primary


@pytest.mark.asyncio
async def test_get_model_without_api_key():
    """get_model sem api_key passa apenas timeout e streaming."""
    mock_settings = _make_settings()
    mock_primary = MagicMock()

    with patch("projects.agent.llm.provider.agent_settings", mock_settings), \
         patch("projects.agent.llm.provider.init_chat_model", return_value=mock_primary) as mock_init:
        from projects.agent.llm.provider import get_model
        result = get_model("analyst")

    mock_init.assert_called_once_with(
        "gpt-4o", model_provider="openai", streaming=False, timeout=60,
    )
    assert result == mock_primary


@pytest.mark.asyncio
async def test_get_model_unknown_role_uses_analyst_default():
    """get_model com role desconhecido usa analyst_model como default."""
    mock_settings = _make_settings()
    mock_primary = MagicMock()

    with patch("projects.agent.llm.provider.agent_settings", mock_settings), \
         patch("projects.agent.llm.provider.init_chat_model", return_value=mock_primary) as mock_init:
        from projects.agent.llm.provider import get_model
        get_model("unknown_role")

    mock_init.assert_called_once_with(
        "gpt-4o", model_provider="openai", streaming=False, timeout=60,
    )


@pytest.mark.asyncio
async def test_get_model_config_override():
    """get_model permite override do modelo via config."""
    mock_settings = _make_settings()
    mock_primary = MagicMock()

    config = {"configurable": {"supervisor_model": "gpt-4-turbo"}}

    with patch("projects.agent.llm.provider.agent_settings", mock_settings), \
         patch("projects.agent.llm.provider.init_chat_model", return_value=mock_primary) as mock_init:
        from projects.agent.llm.provider import get_model
        get_model("supervisor", config)

    mock_init.assert_called_once_with(
        "gpt-4-turbo", model_provider="openai", streaming=False, timeout=15,
    )


# --- P1: Testes de timeout e streaming ---


@pytest.mark.asyncio
async def test_get_model_synthesizer_streaming_enabled():
    """Synthesizer e o unico role com streaming=True (tokens SSE em tempo real)."""
    mock_settings = _make_settings()
    mock_primary = MagicMock()

    with patch("projects.agent.llm.provider.agent_settings", mock_settings), \
         patch("projects.agent.llm.provider.init_chat_model", return_value=mock_primary) as mock_init:
        from projects.agent.llm.provider import get_model
        get_model("synthesizer")

    mock_init.assert_called_once_with(
        "gpt-4o", model_provider="openai", streaming=True, timeout=60,
    )


@pytest.mark.asyncio
async def test_get_model_supervisor_short_timeout():
    """Supervisor usa timeout curto (15s) para roteamento rapido."""
    mock_settings = _make_settings(supervisor_timeout=10)
    mock_primary = MagicMock()

    with patch("projects.agent.llm.provider.agent_settings", mock_settings), \
         patch("projects.agent.llm.provider.init_chat_model", return_value=mock_primary) as mock_init:
        from projects.agent.llm.provider import get_model
        get_model("supervisor")

    _, kwargs = mock_init.call_args
    assert kwargs["timeout"] == 10
    assert kwargs["streaming"] is False


@pytest.mark.asyncio
async def test_get_model_operations_uses_default_timeout():
    """Operations usa timeout padrao (llm_timeout), nao o do supervisor."""
    mock_settings = _make_settings(llm_timeout=45)
    mock_primary = MagicMock()

    with patch("projects.agent.llm.provider.agent_settings", mock_settings), \
         patch("projects.agent.llm.provider.init_chat_model", return_value=mock_primary) as mock_init:
        from projects.agent.llm.provider import get_model
        get_model("operations")

    _, kwargs = mock_init.call_args
    assert kwargs["timeout"] == 45
    assert kwargs["streaming"] is False


@pytest.mark.asyncio
async def test_get_model_non_synthesizer_roles_no_streaming():
    """Nenhum role alem de synthesizer tem streaming habilitado."""
    mock_settings = _make_settings()
    mock_primary = MagicMock()

    for role in ["supervisor", "analyst", "operations", "title_generator"]:
        with patch("projects.agent.llm.provider.agent_settings", mock_settings), \
             patch("projects.agent.llm.provider.init_chat_model", return_value=mock_primary) as mock_init:
            from projects.agent.llm.provider import get_model
            get_model(role)

        _, kwargs = mock_init.call_args
        assert kwargs["streaming"] is False, f"Role '{role}' nao deveria ter streaming"
