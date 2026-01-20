"""Testes para RecommendationAgent."""
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


def _create_mock_recommendation_tools():
    """Cria tools mock para recomendacao."""
    from langchain_core.tools import tool

    @tool
    def get_recommendations(
        config_id: int,
        limit: int = 20,
        active_only: bool = True
    ) -> dict:
        """Lista todas as recomendacoes de otimizacao ativas."""
        return {"total": 0, "recommendations": []}

    @tool
    def get_recommendations_by_type(
        config_id: int,
        recommendation_type: str,
        limit: int = 10
    ) -> dict:
        """Filtra recomendacoes por tipo especifico."""
        return {"total": 0, "recommendations": []}

    @tool
    def get_high_priority_recommendations(
        config_id: int,
        min_priority: int = 7,
        limit: int = 10
    ) -> dict:
        """Retorna recomendacoes de alta prioridade que requerem acao urgente."""
        return {"total": 0, "recommendations": []}

    tools_mod = types.ModuleType('app.agent.tools.recommendation_tools')
    tools_mod.get_recommendations = get_recommendations
    tools_mod.get_recommendations_by_type = get_recommendations_by_type
    tools_mod.get_high_priority_recommendations = get_high_priority_recommendations
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
    sys.modules['app.agent.tools.recommendation_tools'] = _create_mock_recommendation_tools()


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


def load_recommendation_prompts_module():
    """Carrega o modulo de prompts do recommendation."""
    _setup_mock_modules()

    prompts_path = os.path.join(
        root_path, 'app', 'agent', 'subagents', 'recommendation', 'prompts.py'
    )

    if 'app.agent.subagents.recommendation.prompts' in sys.modules:
        return sys.modules['app.agent.subagents.recommendation.prompts']

    # Criar pacote recommendation
    if 'app.agent.subagents.recommendation' not in sys.modules:
        recommendation_mod = types.ModuleType('app.agent.subagents.recommendation')
        sys.modules['app.agent.subagents.recommendation'] = recommendation_mod

    spec = importlib.util.spec_from_file_location(
        "app.agent.subagents.recommendation.prompts",
        prompts_path,
        submodule_search_locations=[os.path.dirname(prompts_path)]
    )
    prompts_module = importlib.util.module_from_spec(spec)
    prompts_module.__package__ = 'app.agent.subagents.recommendation'
    spec.loader.exec_module(prompts_module)
    sys.modules['app.agent.subagents.recommendation.prompts'] = prompts_module

    return prompts_module


def load_recommendation_agent_module():
    """Carrega o modulo do RecommendationAgent."""
    _setup_mock_modules()

    # Carregar dependencias primeiro
    load_base_module()
    load_recommendation_prompts_module()

    agent_path = os.path.join(
        root_path, 'app', 'agent', 'subagents', 'recommendation', 'agent.py'
    )

    if 'app.agent.subagents.recommendation.agent' in sys.modules:
        return sys.modules['app.agent.subagents.recommendation.agent']

    spec = importlib.util.spec_from_file_location(
        "app.agent.subagents.recommendation.agent",
        agent_path,
        submodule_search_locations=[os.path.dirname(agent_path)]
    )
    agent_module = importlib.util.module_from_spec(spec)
    agent_module.__package__ = 'app.agent.subagents.recommendation'
    spec.loader.exec_module(agent_module)
    sys.modules['app.agent.subagents.recommendation.agent'] = agent_module

    return agent_module


def load_recommendation_init_module():
    """Carrega o modulo __init__ do recommendation."""
    _setup_mock_modules()

    # Carregar dependencias primeiro
    load_base_module()
    load_recommendation_prompts_module()
    load_recommendation_agent_module()

    init_path = os.path.join(
        root_path, 'app', 'agent', 'subagents', 'recommendation', '__init__.py'
    )

    # Recarregar para testar exports
    if 'app.agent.subagents.recommendation' in sys.modules:
        del sys.modules['app.agent.subagents.recommendation']

    spec = importlib.util.spec_from_file_location(
        "app.agent.subagents.recommendation",
        init_path,
        submodule_search_locations=[os.path.dirname(init_path)]
    )
    init_module = importlib.util.module_from_spec(spec)
    init_module.__package__ = 'app.agent.subagents'
    spec.loader.exec_module(init_module)
    sys.modules['app.agent.subagents.recommendation'] = init_module

    return init_module


class TestRecommendationAgent:
    """Testes para o agente de recomendacao."""

    def test_recommendation_agent_import(self):
        """RecommendationAgent deve ser importavel."""
        agent_module = load_recommendation_agent_module()
        assert hasattr(agent_module, 'RecommendationAgent')
        assert agent_module.RecommendationAgent is not None

    def test_recommendation_agent_name(self):
        """RecommendationAgent deve ter AGENT_NAME == 'recommendation'."""
        agent_module = load_recommendation_agent_module()
        RecommendationAgent = agent_module.RecommendationAgent
        agent = RecommendationAgent()
        assert agent.AGENT_NAME == "recommendation"

    def test_recommendation_agent_has_three_tools(self):
        """RecommendationAgent deve ter 3 tools."""
        agent_module = load_recommendation_agent_module()
        RecommendationAgent = agent_module.RecommendationAgent
        agent = RecommendationAgent()
        tools = agent.get_tools()
        assert len(tools) == 3

    def test_recommendation_agent_tool_names(self):
        """RecommendationAgent deve ter tools corretas."""
        agent_module = load_recommendation_agent_module()
        RecommendationAgent = agent_module.RecommendationAgent
        agent = RecommendationAgent()
        tools = agent.get_tools()
        tool_names = [t.name for t in tools]

        assert "get_recommendations" in tool_names
        assert "get_recommendations_by_type" in tool_names
        assert "get_high_priority_recommendations" in tool_names

    def test_recommendation_agent_system_prompt(self):
        """RecommendationAgent deve ter system prompt com 'recomendacao' ou 'acao'."""
        agent_module = load_recommendation_agent_module()
        RecommendationAgent = agent_module.RecommendationAgent
        agent = RecommendationAgent()
        prompt = agent.get_system_prompt()

        assert isinstance(prompt, str)
        assert len(prompt) > 100
        # Deve conter referencia a recomendacao ou acao
        prompt_lower = prompt.lower()
        assert "recomenda" in prompt_lower or "acao" in prompt_lower or "acao" in prompt_lower

    def test_recommendation_agent_timeout_30s(self):
        """RecommendationAgent deve ter timeout de 30s."""
        agent_module = load_recommendation_agent_module()
        RecommendationAgent = agent_module.RecommendationAgent
        agent = RecommendationAgent()
        assert agent.get_timeout() == 30

    def test_recommendation_agent_builds_graph(self):
        """RecommendationAgent deve construir grafo."""
        agent_module = load_recommendation_agent_module()
        RecommendationAgent = agent_module.RecommendationAgent
        agent = RecommendationAgent()
        graph = agent.build_graph()
        assert graph is not None

    def test_recommendation_agent_description(self):
        """RecommendationAgent deve ter descricao."""
        agent_module = load_recommendation_agent_module()
        RecommendationAgent = agent_module.RecommendationAgent
        agent = RecommendationAgent()
        assert hasattr(agent, 'AGENT_DESCRIPTION')
        assert len(agent.AGENT_DESCRIPTION) > 10


class TestRecommendationPrompts:
    """Testes para prompts do RecommendationAgent."""

    def test_get_recommendation_prompt_import(self):
        """get_recommendation_prompt deve ser importavel."""
        prompts_module = load_recommendation_prompts_module()
        assert hasattr(prompts_module, 'get_recommendation_prompt')
        assert prompts_module.get_recommendation_prompt is not None

    def test_get_recommendation_prompt_returns_string(self):
        """get_recommendation_prompt deve retornar string."""
        prompts_module = load_recommendation_prompts_module()
        get_recommendation_prompt = prompts_module.get_recommendation_prompt
        prompt = get_recommendation_prompt()
        assert isinstance(prompt, str)

    def test_recommendation_prompt_has_recommendation_types(self):
        """Prompt deve mencionar tipos de recomendacao."""
        prompts_module = load_recommendation_prompts_module()
        get_recommendation_prompt = prompts_module.get_recommendation_prompt
        prompt = get_recommendation_prompt()

        assert "SCALE_UP" in prompt
        assert "BUDGET_INCREASE" in prompt or "budget" in prompt.lower()
        assert "PAUSE_CAMPAIGN" in prompt or "pausar" in prompt.lower()

    def test_recommendation_prompt_constant(self):
        """RECOMMENDATION_SYSTEM_PROMPT deve existir."""
        prompts_module = load_recommendation_prompts_module()
        assert hasattr(prompts_module, 'RECOMMENDATION_SYSTEM_PROMPT')
        assert prompts_module.RECOMMENDATION_SYSTEM_PROMPT is not None
        assert len(prompts_module.RECOMMENDATION_SYSTEM_PROMPT) > 100


class TestRecommendationInit:
    """Testes para __init__.py do recommendation."""

    def test_recommendation_agent_exported(self):
        """RecommendationAgent deve ser exportado."""
        init_module = load_recommendation_init_module()
        assert hasattr(init_module, 'RecommendationAgent')
        assert init_module.RecommendationAgent is not None

    def test_get_recommendation_prompt_exported(self):
        """get_recommendation_prompt deve ser exportado."""
        init_module = load_recommendation_init_module()
        assert hasattr(init_module, 'get_recommendation_prompt')
        assert init_module.get_recommendation_prompt is not None
