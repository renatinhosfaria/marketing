"""Testes para CampaignAgent."""
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


def _create_mock_campaign_tools():
    """Cria tools mock para campanha."""
    from langchain_core.tools import tool

    @tool
    def get_campaign_details(
        config_id: int,
        campaign_id: str
    ) -> dict:
        """Retorna detalhes completos de uma campanha especifica."""
        return {"found": False}

    @tool
    def list_campaigns(
        config_id: int,
        status: str = None,
        limit: int = 50
    ) -> dict:
        """Lista todas as campanhas da conta."""
        return {"total": 0, "campaigns": []}

    tools_mod = types.ModuleType('app.agent.tools.campaign_tools')
    tools_mod.get_campaign_details = get_campaign_details
    tools_mod.list_campaigns = list_campaigns
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
    sys.modules['app.agent.tools.campaign_tools'] = _create_mock_campaign_tools()


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


def load_campaign_prompts_module():
    """Carrega o modulo de prompts do campaign."""
    _setup_mock_modules()

    prompts_path = os.path.join(
        root_path, 'projects', 'agent', 'subagents', 'campaign', 'prompts.py'
    )

    if 'app.agent.subagents.campaign.prompts' in sys.modules:
        return sys.modules['app.agent.subagents.campaign.prompts']

    # Criar pacote campaign
    if 'app.agent.subagents.campaign' not in sys.modules:
        campaign_mod = types.ModuleType('app.agent.subagents.campaign')
        sys.modules['app.agent.subagents.campaign'] = campaign_mod

    spec = importlib.util.spec_from_file_location(
        "app.agent.subagents.campaign.prompts",
        prompts_path,
        submodule_search_locations=[os.path.dirname(prompts_path)]
    )
    prompts_module = importlib.util.module_from_spec(spec)
    prompts_module.__package__ = 'app.agent.subagents.campaign'
    spec.loader.exec_module(prompts_module)
    sys.modules['app.agent.subagents.campaign.prompts'] = prompts_module

    return prompts_module


def load_campaign_agent_module():
    """Carrega o modulo do CampaignAgent."""
    _setup_mock_modules()

    # Carregar dependencias primeiro
    load_base_module()
    load_campaign_prompts_module()

    agent_path = os.path.join(
        root_path, 'projects', 'agent', 'subagents', 'campaign', 'agent.py'
    )

    if 'app.agent.subagents.campaign.agent' in sys.modules:
        return sys.modules['app.agent.subagents.campaign.agent']

    spec = importlib.util.spec_from_file_location(
        "app.agent.subagents.campaign.agent",
        agent_path,
        submodule_search_locations=[os.path.dirname(agent_path)]
    )
    agent_module = importlib.util.module_from_spec(spec)
    agent_module.__package__ = 'app.agent.subagents.campaign'
    spec.loader.exec_module(agent_module)
    sys.modules['app.agent.subagents.campaign.agent'] = agent_module

    return agent_module


def load_campaign_init_module():
    """Carrega o modulo __init__ do campaign."""
    _setup_mock_modules()

    # Carregar dependencias primeiro
    load_base_module()
    load_campaign_prompts_module()
    load_campaign_agent_module()

    init_path = os.path.join(
        root_path, 'projects', 'agent', 'subagents', 'campaign', '__init__.py'
    )

    # Recarregar para testar exports
    if 'app.agent.subagents.campaign' in sys.modules:
        del sys.modules['app.agent.subagents.campaign']

    spec = importlib.util.spec_from_file_location(
        "app.agent.subagents.campaign",
        init_path,
        submodule_search_locations=[os.path.dirname(init_path)]
    )
    init_module = importlib.util.module_from_spec(spec)
    init_module.__package__ = 'app.agent.subagents'
    spec.loader.exec_module(init_module)
    sys.modules['app.agent.subagents.campaign'] = init_module

    return init_module


class TestCampaignAgent:
    """Testes para o agente de campanha."""

    def test_campaign_agent_import(self):
        """CampaignAgent deve ser importavel."""
        agent_module = load_campaign_agent_module()
        assert hasattr(agent_module, 'CampaignAgent')
        assert agent_module.CampaignAgent is not None

    def test_campaign_agent_name(self):
        """CampaignAgent deve ter AGENT_NAME == 'campaign'."""
        agent_module = load_campaign_agent_module()
        CampaignAgent = agent_module.CampaignAgent
        agent = CampaignAgent()
        assert agent.AGENT_NAME == "campaign"

    def test_campaign_agent_has_two_tools(self):
        """CampaignAgent deve ter 2 tools."""
        agent_module = load_campaign_agent_module()
        CampaignAgent = agent_module.CampaignAgent
        agent = CampaignAgent()
        tools = agent.get_tools()
        assert len(tools) == 2

    def test_campaign_agent_tool_names(self):
        """CampaignAgent deve ter tools corretas."""
        agent_module = load_campaign_agent_module()
        CampaignAgent = agent_module.CampaignAgent
        agent = CampaignAgent()
        tools = agent.get_tools()
        tool_names = [t.name for t in tools]

        assert "get_campaign_details" in tool_names
        assert "list_campaigns" in tool_names

    def test_campaign_agent_system_prompt(self):
        """CampaignAgent deve ter system prompt com 'campanha'."""
        agent_module = load_campaign_agent_module()
        CampaignAgent = agent_module.CampaignAgent
        agent = CampaignAgent()
        prompt = agent.get_system_prompt()

        assert isinstance(prompt, str)
        assert len(prompt) > 100
        # Deve conter referencia a campanha
        prompt_lower = prompt.lower()
        assert "campanha" in prompt_lower

    def test_campaign_agent_timeout_20s(self):
        """CampaignAgent deve ter timeout de 20s."""
        agent_module = load_campaign_agent_module()
        CampaignAgent = agent_module.CampaignAgent
        agent = CampaignAgent()
        assert agent.get_timeout() == 20

    def test_campaign_agent_builds_graph(self):
        """CampaignAgent deve construir grafo."""
        agent_module = load_campaign_agent_module()
        CampaignAgent = agent_module.CampaignAgent
        agent = CampaignAgent()
        graph = agent.build_graph()
        assert graph is not None

    def test_campaign_agent_description(self):
        """CampaignAgent deve ter descricao."""
        agent_module = load_campaign_agent_module()
        CampaignAgent = agent_module.CampaignAgent
        agent = CampaignAgent()
        assert hasattr(agent, 'AGENT_DESCRIPTION')
        assert len(agent.AGENT_DESCRIPTION) > 10


class TestCampaignPrompts:
    """Testes para prompts do CampaignAgent."""

    def test_get_campaign_prompt_import(self):
        """get_campaign_prompt deve ser importavel."""
        prompts_module = load_campaign_prompts_module()
        assert hasattr(prompts_module, 'get_campaign_prompt')
        assert prompts_module.get_campaign_prompt is not None

    def test_get_campaign_prompt_returns_string(self):
        """get_campaign_prompt deve retornar string."""
        prompts_module = load_campaign_prompts_module()
        get_campaign_prompt = prompts_module.get_campaign_prompt
        prompt = get_campaign_prompt()
        assert isinstance(prompt, str)

    def test_campaign_prompt_has_campaign_details(self):
        """Prompt deve mencionar dados de campanha."""
        prompts_module = load_campaign_prompts_module()
        get_campaign_prompt = prompts_module.get_campaign_prompt
        prompt = get_campaign_prompt()

        # Deve conter referencia a ID, status, budget ou performance
        prompt_lower = prompt.lower()
        assert any(term in prompt_lower for term in ["id", "status", "budget", "performance"])

    def test_campaign_prompt_constant(self):
        """CAMPAIGN_SYSTEM_PROMPT deve existir."""
        prompts_module = load_campaign_prompts_module()
        assert hasattr(prompts_module, 'CAMPAIGN_SYSTEM_PROMPT')
        assert prompts_module.CAMPAIGN_SYSTEM_PROMPT is not None
        assert len(prompts_module.CAMPAIGN_SYSTEM_PROMPT) > 100


class TestCampaignInit:
    """Testes para __init__.py do campaign."""

    def test_campaign_agent_exported(self):
        """CampaignAgent deve ser exportado."""
        init_module = load_campaign_init_module()
        assert hasattr(init_module, 'CampaignAgent')
        assert init_module.CampaignAgent is not None

    def test_get_campaign_prompt_exported(self):
        """get_campaign_prompt deve ser exportado."""
        init_module = load_campaign_init_module()
        assert hasattr(init_module, 'get_campaign_prompt')
        assert init_module.get_campaign_prompt is not None
