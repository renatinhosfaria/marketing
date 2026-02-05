"""Testes para ClassificationAgent."""
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
    'projects', 'agent', 'subagents', 'state.py'
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


def _create_mock_classification_tools():
    """Cria tools mock para classificacao."""
    from langchain_core.tools import tool

    @tool
    def get_classifications(config_id: int, limit: int = 50, active_only: bool = True) -> dict:
        """Lista todas as classificacoes de campanhas."""
        return {"total": 0, "classifications": []}

    @tool
    def get_campaign_tier(config_id: int, campaign_id: str) -> dict:
        """Retorna o tier de classificacao de uma campanha especifica."""
        return {"found": False}

    @tool
    def get_high_performers(config_id: int, limit: int = 10) -> dict:
        """Lista as campanhas classificadas como HIGH_PERFORMER."""
        return {"total": 0, "campaigns": []}

    @tool
    def get_underperformers(config_id: int, limit: int = 10) -> dict:
        """Lista as campanhas classificadas como UNDERPERFORMER."""
        return {"total": 0, "campaigns": []}

    tools_mod = types.ModuleType('app.agent.tools.classification_tools')
    tools_mod.get_classifications = get_classifications
    tools_mod.get_campaign_tier = get_campaign_tier
    tools_mod.get_high_performers = get_high_performers
    tools_mod.get_underperformers = get_underperformers
    return tools_mod


def _setup_mock_modules():
    """Configura modulos mock necessarios."""
    # Criar pacotes necessarios
    if 'projects' not in sys.modules:
        sys.modules['projects'] = types.ModuleType('projects')
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
    sys.modules['app.agent.tools.classification_tools'] = _create_mock_classification_tools()


def load_base_module():
    """Carrega o modulo base dinamicamente."""
    _setup_mock_modules()

    base_path = os.path.join(
        root_path, 'projects', 'agent', 'subagents', 'base.py'
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


def load_classification_prompts_module():
    """Carrega o modulo de prompts do classification."""
    _setup_mock_modules()

    prompts_path = os.path.join(
        root_path, 'projects', 'agent', 'subagents', 'classification', 'prompts.py'
    )

    if 'app.agent.subagents.classification.prompts' in sys.modules:
        return sys.modules['app.agent.subagents.classification.prompts']

    # Criar pacote classification
    if 'app.agent.subagents.classification' not in sys.modules:
        classification_mod = types.ModuleType('app.agent.subagents.classification')
        sys.modules['app.agent.subagents.classification'] = classification_mod

    spec = importlib.util.spec_from_file_location(
        "app.agent.subagents.classification.prompts",
        prompts_path,
        submodule_search_locations=[os.path.dirname(prompts_path)]
    )
    prompts_module = importlib.util.module_from_spec(spec)
    prompts_module.__package__ = 'app.agent.subagents.classification'
    spec.loader.exec_module(prompts_module)
    sys.modules['app.agent.subagents.classification.prompts'] = prompts_module

    return prompts_module


def load_classification_agent_module():
    """Carrega o modulo do ClassificationAgent."""
    _setup_mock_modules()

    # Carregar dependencias primeiro
    load_base_module()
    load_classification_prompts_module()

    agent_path = os.path.join(
        root_path, 'projects', 'agent', 'subagents', 'classification', 'agent.py'
    )

    if 'app.agent.subagents.classification.agent' in sys.modules:
        return sys.modules['app.agent.subagents.classification.agent']

    spec = importlib.util.spec_from_file_location(
        "app.agent.subagents.classification.agent",
        agent_path,
        submodule_search_locations=[os.path.dirname(agent_path)]
    )
    agent_module = importlib.util.module_from_spec(spec)
    agent_module.__package__ = 'app.agent.subagents.classification'
    spec.loader.exec_module(agent_module)
    sys.modules['app.agent.subagents.classification.agent'] = agent_module

    return agent_module


def load_classification_init_module():
    """Carrega o modulo __init__ do classification."""
    _setup_mock_modules()

    # Carregar dependencias primeiro
    load_base_module()
    load_classification_prompts_module()
    load_classification_agent_module()

    init_path = os.path.join(
        root_path, 'projects', 'agent', 'subagents', 'classification', '__init__.py'
    )

    # Recarregar para testar exports
    if 'app.agent.subagents.classification' in sys.modules:
        del sys.modules['app.agent.subagents.classification']

    spec = importlib.util.spec_from_file_location(
        "app.agent.subagents.classification",
        init_path,
        submodule_search_locations=[os.path.dirname(init_path)]
    )
    init_module = importlib.util.module_from_spec(spec)
    init_module.__package__ = 'app.agent.subagents'
    spec.loader.exec_module(init_module)
    sys.modules['app.agent.subagents.classification'] = init_module

    return init_module


class TestClassificationAgent:
    """Testes para o agente de classificacao."""

    def test_classification_agent_import(self):
        """ClassificationAgent deve ser importavel."""
        agent_module = load_classification_agent_module()
        assert hasattr(agent_module, 'ClassificationAgent')
        assert agent_module.ClassificationAgent is not None

    def test_classification_agent_name(self):
        """ClassificationAgent deve ter nome correto."""
        agent_module = load_classification_agent_module()
        ClassificationAgent = agent_module.ClassificationAgent
        agent = ClassificationAgent()
        assert agent.AGENT_NAME == "classification"

    def test_classification_agent_has_tools(self):
        """ClassificationAgent deve ter 4 tools."""
        agent_module = load_classification_agent_module()
        ClassificationAgent = agent_module.ClassificationAgent
        agent = ClassificationAgent()
        tools = agent.get_tools()
        assert len(tools) == 4

    def test_classification_agent_tool_names(self):
        """ClassificationAgent deve ter tools corretas."""
        agent_module = load_classification_agent_module()
        ClassificationAgent = agent_module.ClassificationAgent
        agent = ClassificationAgent()
        tools = agent.get_tools()
        tool_names = [t.name for t in tools]

        assert "get_classifications" in tool_names
        assert "get_campaign_tier" in tool_names
        assert "get_high_performers" in tool_names
        assert "get_underperformers" in tool_names

    def test_classification_agent_system_prompt(self):
        """ClassificationAgent deve ter system prompt."""
        agent_module = load_classification_agent_module()
        ClassificationAgent = agent_module.ClassificationAgent
        agent = ClassificationAgent()
        prompt = agent.get_system_prompt()

        assert isinstance(prompt, str)
        assert len(prompt) > 100
        assert "classifica" in prompt.lower() or "tier" in prompt.lower()

    def test_classification_agent_builds_graph(self):
        """ClassificationAgent deve construir grafo."""
        agent_module = load_classification_agent_module()
        ClassificationAgent = agent_module.ClassificationAgent
        agent = ClassificationAgent()
        graph = agent.build_graph()
        assert graph is not None

    def test_classification_agent_timeout(self):
        """ClassificationAgent deve ter timeout de 30s."""
        agent_module = load_classification_agent_module()
        ClassificationAgent = agent_module.ClassificationAgent
        agent = ClassificationAgent()
        assert agent.get_timeout() == 30

    def test_classification_agent_description(self):
        """ClassificationAgent deve ter descricao."""
        agent_module = load_classification_agent_module()
        ClassificationAgent = agent_module.ClassificationAgent
        agent = ClassificationAgent()
        assert hasattr(agent, 'AGENT_DESCRIPTION')
        assert len(agent.AGENT_DESCRIPTION) > 10


class TestClassificationPrompts:
    """Testes para prompts do ClassificationAgent."""

    def test_get_classification_prompt_import(self):
        """get_classification_prompt deve ser importavel."""
        prompts_module = load_classification_prompts_module()
        assert hasattr(prompts_module, 'get_classification_prompt')
        assert prompts_module.get_classification_prompt is not None

    def test_get_classification_prompt_returns_string(self):
        """get_classification_prompt deve retornar string."""
        prompts_module = load_classification_prompts_module()
        get_classification_prompt = prompts_module.get_classification_prompt
        prompt = get_classification_prompt()
        assert isinstance(prompt, str)

    def test_classification_prompt_has_tier_info(self):
        """Prompt deve mencionar tiers de performance."""
        prompts_module = load_classification_prompts_module()
        get_classification_prompt = prompts_module.get_classification_prompt
        prompt = get_classification_prompt()

        assert "HIGH_PERFORMER" in prompt
        assert "MODERATE" in prompt
        assert "LOW" in prompt
        assert "UNDERPERFORMER" in prompt

    def test_classification_prompt_constant(self):
        """CLASSIFICATION_SYSTEM_PROMPT deve existir."""
        prompts_module = load_classification_prompts_module()
        assert hasattr(prompts_module, 'CLASSIFICATION_SYSTEM_PROMPT')
        assert prompts_module.CLASSIFICATION_SYSTEM_PROMPT is not None
        assert len(prompts_module.CLASSIFICATION_SYSTEM_PROMPT) > 100


class TestClassificationInit:
    """Testes para __init__.py do classification."""

    def test_classification_agent_exported(self):
        """ClassificationAgent deve ser exportado."""
        init_module = load_classification_init_module()
        assert hasattr(init_module, 'ClassificationAgent')
        assert init_module.ClassificationAgent is not None

    def test_get_classification_prompt_exported(self):
        """get_classification_prompt deve ser exportado."""
        init_module = load_classification_init_module()
        assert hasattr(init_module, 'get_classification_prompt')
        assert init_module.get_classification_prompt is not None

    def test_classification_system_prompt_exported(self):
        """CLASSIFICATION_SYSTEM_PROMPT deve ser exportado."""
        init_module = load_classification_init_module()
        assert hasattr(init_module, 'CLASSIFICATION_SYSTEM_PROMPT')
