"""Testes para ForecastAgent."""
import sys
import os
import importlib.util
import types

import pytest
from unittest.mock import Mock, patch


# Adicionar o diretorio raiz ao path para permitir imports
root_path = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    '..', '..', '..', '..'
))
if root_path not in sys.path:
    sys.path.insert(0, root_path)


# Carregar o modulo state diretamente para evitar dependencias
state_path = os.path.join(
    os.path.dirname(__file__),
    '..', '..', '..', '..',
    'app', 'agent', 'subagents', 'state.py'
)
state_path = os.path.abspath(state_path)

spec = importlib.util.spec_from_file_location("subagent_state", state_path)
subagent_state = importlib.util.module_from_spec(spec)
spec.loader.exec_module(subagent_state)


def _create_mock_config_module():
    """Cria um modulo de configuracao mock."""
    config_mod = types.ModuleType('app.agent.config')

    class MockAgentSettings:
        timeout_test = 30
        timeout_classification = 30
        timeout_anomaly = 30
        timeout_forecast = 45
        timeout_recommendation = 30
        timeout_campaign = 20
        timeout_analysis = 45

    def get_agent_settings():
        return MockAgentSettings()

    config_mod.get_agent_settings = get_agent_settings
    config_mod.AgentSettings = MockAgentSettings
    return config_mod


def _create_mock_forecast_tools():
    """Cria tools mock para previsao."""
    from langchain_core.tools import tool

    @tool
    def get_forecasts(
        config_id: int,
        forecast_type: str = None,
        days_ahead: int = 7,
        limit: int = 30
    ) -> dict:
        """Lista previsoes de metricas para as campanhas."""
        return {"total": 0, "forecasts": []}

    @tool
    def predict_campaign_cpl(
        config_id: int,
        campaign_id: str,
        days_ahead: int = 7
    ) -> dict:
        """Retorna previsao de CPL para uma campanha especifica."""
        return {"found": False}

    @tool
    def predict_campaign_leads(
        config_id: int,
        campaign_id: str,
        days_ahead: int = 7
    ) -> dict:
        """Retorna previsao de leads para uma campanha especifica."""
        return {"found": False}

    tools_mod = types.ModuleType('app.agent.tools.forecast_tools')
    tools_mod.get_forecasts = get_forecasts
    tools_mod.predict_campaign_cpl = predict_campaign_cpl
    tools_mod.predict_campaign_leads = predict_campaign_leads
    return tools_mod


def _setup_mock_modules():
    """Configura modulos mock necessarios."""
    # Criar pacotes necessarios
    if 'app' not in sys.modules:
        sys.modules['app'] = types.ModuleType('app')
    if 'app.agent' not in sys.modules:
        agent_mod = types.ModuleType('app.agent')
        sys.modules['app.agent'] = agent_mod
    if 'app.agent.subagents' not in sys.modules:
        subagents_mod = types.ModuleType('app.agent.subagents')
        sys.modules['app.agent.subagents'] = subagents_mod
    if 'app.agent.tools' not in sys.modules:
        tools_mod = types.ModuleType('app.agent.tools')
        sys.modules['app.agent.tools'] = tools_mod

    # Registrar modulos
    sys.modules['app.agent.subagents.state'] = subagent_state
    sys.modules['app.agent.config'] = _create_mock_config_module()
    sys.modules['app.agent.tools.forecast_tools'] = _create_mock_forecast_tools()


def load_base_module():
    """Carrega o modulo base dinamicamente."""
    _setup_mock_modules()

    base_path = os.path.join(
        root_path, 'app', 'agent', 'subagents', 'base.py'
    )

    if 'app.agent.subagents.base' in sys.modules:
        return sys.modules['app.agent.subagents.base']

    spec = importlib.util.spec_from_file_location(
        "app.agent.subagents.base",
        base_path,
        submodule_search_locations=[os.path.dirname(base_path)]
    )
    base_module = importlib.util.module_from_spec(spec)
    base_module.__package__ = 'app.agent.subagents'
    spec.loader.exec_module(base_module)
    sys.modules['app.agent.subagents.base'] = base_module

    return base_module


def load_forecast_prompts_module():
    """Carrega o modulo de prompts do forecast."""
    _setup_mock_modules()

    prompts_path = os.path.join(
        root_path, 'app', 'agent', 'subagents', 'forecast', 'prompts.py'
    )

    if 'app.agent.subagents.forecast.prompts' in sys.modules:
        return sys.modules['app.agent.subagents.forecast.prompts']

    # Criar pacote forecast
    if 'app.agent.subagents.forecast' not in sys.modules:
        forecast_mod = types.ModuleType('app.agent.subagents.forecast')
        sys.modules['app.agent.subagents.forecast'] = forecast_mod

    spec = importlib.util.spec_from_file_location(
        "app.agent.subagents.forecast.prompts",
        prompts_path,
        submodule_search_locations=[os.path.dirname(prompts_path)]
    )
    prompts_module = importlib.util.module_from_spec(spec)
    prompts_module.__package__ = 'app.agent.subagents.forecast'
    spec.loader.exec_module(prompts_module)
    sys.modules['app.agent.subagents.forecast.prompts'] = prompts_module

    return prompts_module


def load_forecast_agent_module():
    """Carrega o modulo do ForecastAgent."""
    _setup_mock_modules()

    # Carregar dependencias primeiro
    load_base_module()
    load_forecast_prompts_module()

    agent_path = os.path.join(
        root_path, 'app', 'agent', 'subagents', 'forecast', 'agent.py'
    )

    if 'app.agent.subagents.forecast.agent' in sys.modules:
        return sys.modules['app.agent.subagents.forecast.agent']

    spec = importlib.util.spec_from_file_location(
        "app.agent.subagents.forecast.agent",
        agent_path,
        submodule_search_locations=[os.path.dirname(agent_path)]
    )
    agent_module = importlib.util.module_from_spec(spec)
    agent_module.__package__ = 'app.agent.subagents.forecast'
    spec.loader.exec_module(agent_module)
    sys.modules['app.agent.subagents.forecast.agent'] = agent_module

    return agent_module


def load_forecast_init_module():
    """Carrega o modulo __init__ do forecast."""
    _setup_mock_modules()

    # Carregar dependencias primeiro
    load_base_module()
    load_forecast_prompts_module()
    load_forecast_agent_module()

    init_path = os.path.join(
        root_path, 'app', 'agent', 'subagents', 'forecast', '__init__.py'
    )

    # Recarregar para testar exports
    if 'app.agent.subagents.forecast' in sys.modules:
        del sys.modules['app.agent.subagents.forecast']

    spec = importlib.util.spec_from_file_location(
        "app.agent.subagents.forecast",
        init_path,
        submodule_search_locations=[os.path.dirname(init_path)]
    )
    init_module = importlib.util.module_from_spec(spec)
    init_module.__package__ = 'app.agent.subagents'
    spec.loader.exec_module(init_module)
    sys.modules['app.agent.subagents.forecast'] = init_module

    return init_module


class TestForecastAgent:
    """Testes para o agente de previsao."""

    def test_forecast_agent_import(self):
        """ForecastAgent deve ser importavel."""
        agent_module = load_forecast_agent_module()
        assert hasattr(agent_module, 'ForecastAgent')
        assert agent_module.ForecastAgent is not None

    def test_forecast_agent_name(self):
        """ForecastAgent deve ter AGENT_NAME == 'forecast'."""
        agent_module = load_forecast_agent_module()
        ForecastAgent = agent_module.ForecastAgent
        agent = ForecastAgent()
        assert agent.AGENT_NAME == "forecast"

    def test_forecast_agent_has_three_tools(self):
        """ForecastAgent deve ter 3 tools."""
        agent_module = load_forecast_agent_module()
        ForecastAgent = agent_module.ForecastAgent
        agent = ForecastAgent()
        tools = agent.get_tools()
        assert len(tools) == 3

    def test_forecast_agent_tool_names(self):
        """ForecastAgent deve ter tools corretas."""
        agent_module = load_forecast_agent_module()
        ForecastAgent = agent_module.ForecastAgent
        agent = ForecastAgent()
        tools = agent.get_tools()
        tool_names = [t.name for t in tools]

        assert "get_forecasts" in tool_names
        assert "predict_campaign_cpl" in tool_names
        assert "predict_campaign_leads" in tool_names

    def test_forecast_agent_system_prompt(self):
        """ForecastAgent deve ter system prompt com 'previsao' ou 'forecast'."""
        agent_module = load_forecast_agent_module()
        ForecastAgent = agent_module.ForecastAgent
        agent = ForecastAgent()
        prompt = agent.get_system_prompt()

        assert isinstance(prompt, str)
        assert len(prompt) > 100
        # Deve conter referencia a previsao ou forecast
        prompt_lower = prompt.lower()
        assert "previs" in prompt_lower or "forecast" in prompt_lower

    def test_forecast_agent_timeout_45s(self):
        """ForecastAgent deve ter timeout de 45s."""
        agent_module = load_forecast_agent_module()
        ForecastAgent = agent_module.ForecastAgent
        agent = ForecastAgent()
        assert agent.get_timeout() == 45

    def test_forecast_agent_builds_graph(self):
        """ForecastAgent deve construir grafo."""
        agent_module = load_forecast_agent_module()
        ForecastAgent = agent_module.ForecastAgent
        agent = ForecastAgent()
        graph = agent.build_graph()
        assert graph is not None

    def test_forecast_agent_description(self):
        """ForecastAgent deve ter descricao."""
        agent_module = load_forecast_agent_module()
        ForecastAgent = agent_module.ForecastAgent
        agent = ForecastAgent()
        assert hasattr(agent, 'AGENT_DESCRIPTION')
        assert len(agent.AGENT_DESCRIPTION) > 10


class TestForecastPrompts:
    """Testes para prompts do ForecastAgent."""

    def test_get_forecast_prompt_import(self):
        """get_forecast_prompt deve ser importavel."""
        prompts_module = load_forecast_prompts_module()
        assert hasattr(prompts_module, 'get_forecast_prompt')
        assert prompts_module.get_forecast_prompt is not None

    def test_get_forecast_prompt_returns_string(self):
        """get_forecast_prompt deve retornar string."""
        prompts_module = load_forecast_prompts_module()
        get_forecast_prompt = prompts_module.get_forecast_prompt
        prompt = get_forecast_prompt()
        assert isinstance(prompt, str)

    def test_forecast_prompt_has_forecast_types(self):
        """Prompt deve mencionar tipos de previsao."""
        prompts_module = load_forecast_prompts_module()
        get_forecast_prompt = prompts_module.get_forecast_prompt
        prompt = get_forecast_prompt()

        assert "CPL_FORECAST" in prompt or "CPL" in prompt
        assert "LEADS_FORECAST" in prompt or "Leads" in prompt

    def test_forecast_prompt_constant(self):
        """FORECAST_SYSTEM_PROMPT deve existir."""
        prompts_module = load_forecast_prompts_module()
        assert hasattr(prompts_module, 'FORECAST_SYSTEM_PROMPT')
        assert prompts_module.FORECAST_SYSTEM_PROMPT is not None
        assert len(prompts_module.FORECAST_SYSTEM_PROMPT) > 100


class TestForecastInit:
    """Testes para __init__.py do forecast."""

    def test_forecast_agent_exported(self):
        """ForecastAgent deve ser exportado."""
        init_module = load_forecast_init_module()
        assert hasattr(init_module, 'ForecastAgent')
        assert init_module.ForecastAgent is not None

    def test_get_forecast_prompt_exported(self):
        """get_forecast_prompt deve ser exportado."""
        init_module = load_forecast_init_module()
        assert hasattr(init_module, 'get_forecast_prompt')
        assert init_module.get_forecast_prompt is not None
