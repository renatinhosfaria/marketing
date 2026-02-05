"""Testes para AnomalyAgent."""
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


def _create_mock_anomaly_tools():
    """Cria tools mock para anomalias."""
    from langchain_core.tools import tool

    @tool
    def get_anomalies(config_id: int, days: int = 7, limit: int = 50) -> dict:
        """Lista anomalias detectadas nas campanhas."""
        return {"total": 0, "anomalies": []}

    @tool
    def get_critical_anomalies(config_id: int, days: int = 3, limit: int = 20) -> dict:
        """Retorna anomalias de severidade CRITICAL e HIGH."""
        return {"total": 0, "anomalies": []}

    @tool
    def get_anomalies_by_type(config_id: int, anomaly_type: str, days: int = 7, limit: int = 20) -> dict:
        """Filtra anomalias por tipo especifico."""
        return {"total": 0, "anomalies": []}

    tools_mod = types.ModuleType('app.agent.tools.anomaly_tools')
    tools_mod.get_anomalies = get_anomalies
    tools_mod.get_critical_anomalies = get_critical_anomalies
    tools_mod.get_anomalies_by_type = get_anomalies_by_type
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
    sys.modules['app.agent.tools.anomaly_tools'] = _create_mock_anomaly_tools()


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


def load_anomaly_prompts_module():
    """Carrega o modulo de prompts do anomaly."""
    _setup_mock_modules()

    prompts_path = os.path.join(
        root_path, 'projects', 'agent', 'subagents', 'anomaly', 'prompts.py'
    )

    if 'app.agent.subagents.anomaly.prompts' in sys.modules:
        return sys.modules['app.agent.subagents.anomaly.prompts']

    # Criar pacote anomaly
    if 'app.agent.subagents.anomaly' not in sys.modules:
        anomaly_mod = types.ModuleType('app.agent.subagents.anomaly')
        sys.modules['app.agent.subagents.anomaly'] = anomaly_mod

    spec = importlib.util.spec_from_file_location(
        "app.agent.subagents.anomaly.prompts",
        prompts_path,
        submodule_search_locations=[os.path.dirname(prompts_path)]
    )
    prompts_module = importlib.util.module_from_spec(spec)
    prompts_module.__package__ = 'app.agent.subagents.anomaly'
    spec.loader.exec_module(prompts_module)
    sys.modules['app.agent.subagents.anomaly.prompts'] = prompts_module

    return prompts_module


def load_anomaly_agent_module():
    """Carrega o modulo do AnomalyAgent."""
    _setup_mock_modules()

    # Carregar dependencias primeiro
    load_base_module()
    load_anomaly_prompts_module()

    agent_path = os.path.join(
        root_path, 'projects', 'agent', 'subagents', 'anomaly', 'agent.py'
    )

    if 'app.agent.subagents.anomaly.agent' in sys.modules:
        return sys.modules['app.agent.subagents.anomaly.agent']

    spec = importlib.util.spec_from_file_location(
        "app.agent.subagents.anomaly.agent",
        agent_path,
        submodule_search_locations=[os.path.dirname(agent_path)]
    )
    agent_module = importlib.util.module_from_spec(spec)
    agent_module.__package__ = 'app.agent.subagents.anomaly'
    spec.loader.exec_module(agent_module)
    sys.modules['app.agent.subagents.anomaly.agent'] = agent_module

    return agent_module


def load_anomaly_init_module():
    """Carrega o modulo __init__ do anomaly."""
    _setup_mock_modules()

    # Carregar dependencias primeiro
    load_base_module()
    load_anomaly_prompts_module()
    load_anomaly_agent_module()

    init_path = os.path.join(
        root_path, 'projects', 'agent', 'subagents', 'anomaly', '__init__.py'
    )

    # Recarregar para testar exports
    if 'app.agent.subagents.anomaly' in sys.modules:
        del sys.modules['app.agent.subagents.anomaly']

    spec = importlib.util.spec_from_file_location(
        "app.agent.subagents.anomaly",
        init_path,
        submodule_search_locations=[os.path.dirname(init_path)]
    )
    init_module = importlib.util.module_from_spec(spec)
    init_module.__package__ = 'app.agent.subagents'
    spec.loader.exec_module(init_module)
    sys.modules['app.agent.subagents.anomaly'] = init_module

    return init_module


class TestAnomalyAgent:
    """Testes para o agente de anomalias."""

    def test_anomaly_agent_import(self):
        """AnomalyAgent deve ser importavel."""
        agent_module = load_anomaly_agent_module()
        assert hasattr(agent_module, 'AnomalyAgent')
        assert agent_module.AnomalyAgent is not None

    def test_anomaly_agent_name(self):
        """AnomalyAgent deve ter nome correto."""
        agent_module = load_anomaly_agent_module()
        AnomalyAgent = agent_module.AnomalyAgent
        agent = AnomalyAgent()
        assert agent.AGENT_NAME == "anomaly"

    def test_anomaly_agent_has_tools(self):
        """AnomalyAgent deve ter 3 tools."""
        agent_module = load_anomaly_agent_module()
        AnomalyAgent = agent_module.AnomalyAgent
        agent = AnomalyAgent()
        tools = agent.get_tools()
        assert len(tools) == 3

    def test_anomaly_agent_tool_names(self):
        """AnomalyAgent deve ter tools corretas."""
        agent_module = load_anomaly_agent_module()
        AnomalyAgent = agent_module.AnomalyAgent
        agent = AnomalyAgent()
        tools = agent.get_tools()
        tool_names = [t.name for t in tools]

        assert "get_anomalies" in tool_names
        assert "get_critical_anomalies" in tool_names
        assert "get_anomalies_by_type" in tool_names

    def test_anomaly_agent_system_prompt(self):
        """AnomalyAgent deve ter system prompt sobre anomalias."""
        agent_module = load_anomaly_agent_module()
        AnomalyAgent = agent_module.AnomalyAgent
        agent = AnomalyAgent()
        prompt = agent.get_system_prompt()

        assert isinstance(prompt, str)
        assert "anomalia" in prompt.lower() or "problema" in prompt.lower()

    def test_anomaly_agent_builds_graph(self):
        """AnomalyAgent deve construir grafo."""
        agent_module = load_anomaly_agent_module()
        AnomalyAgent = agent_module.AnomalyAgent
        agent = AnomalyAgent()
        graph = agent.build_graph()
        assert graph is not None

    def test_anomaly_agent_timeout(self):
        """AnomalyAgent deve ter timeout de 30s."""
        agent_module = load_anomaly_agent_module()
        AnomalyAgent = agent_module.AnomalyAgent
        agent = AnomalyAgent()
        assert agent.get_timeout() == 30

    def test_anomaly_agent_description(self):
        """AnomalyAgent deve ter descricao."""
        agent_module = load_anomaly_agent_module()
        AnomalyAgent = agent_module.AnomalyAgent
        agent = AnomalyAgent()
        assert hasattr(agent, "AGENT_DESCRIPTION")
        assert isinstance(agent.AGENT_DESCRIPTION, str)
        assert len(agent.AGENT_DESCRIPTION) > 0

    def test_anomaly_agent_inherits_from_base(self):
        """AnomalyAgent deve herdar de BaseSubagent."""
        agent_module = load_anomaly_agent_module()
        base_module = load_base_module()
        AnomalyAgent = agent_module.AnomalyAgent
        BaseSubagent = base_module.BaseSubagent
        assert issubclass(AnomalyAgent, BaseSubagent)


class TestAnomalyPrompts:
    """Testes para prompts do AnomalyAgent."""

    def test_get_anomaly_prompt_import(self):
        """get_anomaly_prompt deve ser importavel."""
        prompts_module = load_anomaly_prompts_module()
        assert hasattr(prompts_module, 'get_anomaly_prompt')
        assert prompts_module.get_anomaly_prompt is not None

    def test_get_anomaly_prompt_returns_string(self):
        """get_anomaly_prompt deve retornar string."""
        prompts_module = load_anomaly_prompts_module()
        get_anomaly_prompt = prompts_module.get_anomaly_prompt
        prompt = get_anomaly_prompt()
        assert isinstance(prompt, str)

    def test_anomaly_prompt_has_severity_info(self):
        """Prompt deve mencionar severidades."""
        prompts_module = load_anomaly_prompts_module()
        get_anomaly_prompt = prompts_module.get_anomaly_prompt
        prompt = get_anomaly_prompt()

        assert "CRITICAL" in prompt
        assert "HIGH" in prompt
        assert "MEDIUM" in prompt
        assert "LOW" in prompt

    def test_anomaly_prompt_has_type_info(self):
        """Prompt deve mencionar tipos de anomalias."""
        prompts_module = load_anomaly_prompts_module()
        get_anomaly_prompt = prompts_module.get_anomaly_prompt
        prompt = get_anomaly_prompt()

        assert "CPL_SPIKE" in prompt
        assert "CTR_DROP" in prompt

    def test_anomaly_prompt_constant(self):
        """ANOMALY_SYSTEM_PROMPT deve existir."""
        prompts_module = load_anomaly_prompts_module()
        assert hasattr(prompts_module, 'ANOMALY_SYSTEM_PROMPT')
        assert prompts_module.ANOMALY_SYSTEM_PROMPT is not None
        assert len(prompts_module.ANOMALY_SYSTEM_PROMPT) > 100


class TestAnomalyInit:
    """Testes para __init__.py do anomaly."""

    def test_anomaly_agent_exported(self):
        """AnomalyAgent deve ser exportado."""
        init_module = load_anomaly_init_module()
        assert hasattr(init_module, 'AnomalyAgent')
        assert init_module.AnomalyAgent is not None

    def test_get_anomaly_prompt_exported(self):
        """get_anomaly_prompt deve ser exportado."""
        init_module = load_anomaly_init_module()
        assert hasattr(init_module, 'get_anomaly_prompt')
        assert init_module.get_anomaly_prompt is not None
