"""Testes para o registro de subagentes."""
import sys
import os
import importlib.util
import types

import pytest
from unittest.mock import Mock


# Adicionar o diretorio raiz ao path para permitir imports
root_path = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    '..', '..', '..', '..'
))
if root_path not in sys.path:
    sys.path.insert(0, root_path)


def _create_mock_config_module():
    """Cria um modulo de configuracao mock."""
    config_mod = types.ModuleType('app.agent.config')

    class MockAgentSettings:
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


def _create_mock_tool(name: str, description: str):
    """Cria uma tool mock."""
    from langchain_core.tools import tool

    @tool
    def mock_tool(config_id: int) -> dict:
        """Tool mock."""
        return {"status": "ok"}

    mock_tool.name = name
    mock_tool.description = description
    return mock_tool


def _setup_mocks():
    """Configura todos os mocks necessarios."""
    # Mock config module
    config_mod = _create_mock_config_module()
    sys.modules['app.agent.config'] = config_mod

    # Mock tools modules
    for tool_module in [
        'app.agent.tools.classification_tools',
        'app.agent.tools.anomaly_tools',
        'app.agent.tools.forecast_tools',
        'app.agent.tools.recommendation_tools',
        'app.agent.tools.campaign_tools',
        'app.agent.tools.analysis_tools',
    ]:
        mock_mod = types.ModuleType(tool_module)

        # Adicionar tools mock
        mock_mod.get_classifications = _create_mock_tool("get_classifications", "Get classifications")
        mock_mod.get_campaign_tier = _create_mock_tool("get_campaign_tier", "Get campaign tier")
        mock_mod.get_high_performers = _create_mock_tool("get_high_performers", "Get high performers")
        mock_mod.get_underperformers = _create_mock_tool("get_underperformers", "Get underperformers")
        mock_mod.get_anomalies = _create_mock_tool("get_anomalies", "Get anomalies")
        mock_mod.get_critical_anomalies = _create_mock_tool("get_critical_anomalies", "Get critical")
        mock_mod.get_anomalies_by_type = _create_mock_tool("get_anomalies_by_type", "Get by type")
        mock_mod.get_forecasts = _create_mock_tool("get_forecasts", "Get forecasts")
        mock_mod.predict_campaign_cpl = _create_mock_tool("predict_campaign_cpl", "Predict CPL")
        mock_mod.predict_campaign_leads = _create_mock_tool("predict_campaign_leads", "Predict leads")
        mock_mod.get_recommendations = _create_mock_tool("get_recommendations", "Get recs")
        mock_mod.get_recommendations_by_type = _create_mock_tool("get_recommendations_by_type", "Get by type")
        mock_mod.get_high_priority_recommendations = _create_mock_tool("get_high_priority", "Get high priority")
        mock_mod.get_campaign_details = _create_mock_tool("get_campaign_details", "Get details")
        mock_mod.list_campaigns = _create_mock_tool("list_campaigns", "List campaigns")
        mock_mod.compare_campaigns = _create_mock_tool("compare_campaigns", "Compare campaigns")
        mock_mod.analyze_trends = _create_mock_tool("analyze_trends", "Analyze trends")
        mock_mod.get_account_summary = _create_mock_tool("get_account_summary", "Get summary")
        mock_mod.calculate_roi = _create_mock_tool("calculate_roi", "Calculate ROI")
        mock_mod.get_top_campaigns = _create_mock_tool("get_top_campaigns", "Get top campaigns")

        sys.modules[tool_module] = mock_mod


# Setup mocks antes de importar os agentes
_setup_mocks()


def _load_module_directly(module_name: str, file_path: str):
    """Carrega um modulo Python diretamente de arquivo."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# Carregar state
state_path = os.path.join(root_path, 'app', 'agent', 'subagents', 'state.py')
subagent_state = _load_module_directly('app.agent.subagents.state', state_path)

# Carregar base
base_path = os.path.join(root_path, 'app', 'agent', 'subagents', 'base.py')
subagent_base = _load_module_directly('app.agent.subagents.base', base_path)

# Carregar prompts de cada agente
for agent_name in ['classification', 'anomaly', 'forecast', 'recommendation', 'campaign', 'analysis']:
    prompts_path = os.path.join(root_path, 'app', 'agent', 'subagents', agent_name, 'prompts.py')
    _load_module_directly(f'app.agent.subagents.{agent_name}.prompts', prompts_path)

    agent_path = os.path.join(root_path, 'app', 'agent', 'subagents', agent_name, 'agent.py')
    _load_module_directly(f'app.agent.subagents.{agent_name}.agent', agent_path)

# Agora carregamos o __init__ que contém o registry
init_path = os.path.join(root_path, 'app', 'agent', 'subagents', '__init__.py')
subagents_module = _load_module_directly('app.agent.subagents', init_path)


class TestSubagentRegistry:
    """Testes para o registro de subagentes."""

    def test_get_subagent_by_name(self):
        """get_subagent deve retornar agente correto."""
        get_subagent = subagents_module.get_subagent

        agent = get_subagent("classification")
        assert agent.AGENT_NAME == "classification"

        agent = get_subagent("anomaly")
        assert agent.AGENT_NAME == "anomaly"

    def test_get_subagent_invalid_name(self):
        """get_subagent deve levantar erro para nome invalido."""
        get_subagent = subagents_module.get_subagent

        with pytest.raises(ValueError):
            get_subagent("invalid_agent")

    def test_get_all_subagents(self):
        """get_all_subagents deve retornar todos os 6 agentes."""
        get_all_subagents = subagents_module.get_all_subagents

        agents = get_all_subagents()
        assert len(agents) == 6

        names = [a.AGENT_NAME for a in agents]
        assert "classification" in names
        assert "anomaly" in names
        assert "forecast" in names
        assert "recommendation" in names
        assert "campaign" in names
        assert "analysis" in names

    def test_subagent_registry_constant(self):
        """SUBAGENT_REGISTRY deve conter todos os agentes."""
        SUBAGENT_REGISTRY = subagents_module.SUBAGENT_REGISTRY

        assert "classification" in SUBAGENT_REGISTRY
        assert "anomaly" in SUBAGENT_REGISTRY
        assert "forecast" in SUBAGENT_REGISTRY
        assert "recommendation" in SUBAGENT_REGISTRY
        assert "campaign" in SUBAGENT_REGISTRY
        assert "analysis" in SUBAGENT_REGISTRY

    def test_get_subagent_returns_new_instance(self):
        """get_subagent deve retornar nova instancia a cada chamada."""
        get_subagent = subagents_module.get_subagent

        agent1 = get_subagent("classification")
        agent2 = get_subagent("classification")

        assert agent1 is not agent2
        assert agent1.AGENT_NAME == agent2.AGENT_NAME

    def test_get_subagent_all_valid_names(self):
        """get_subagent deve funcionar para todos os nomes validos."""
        get_subagent = subagents_module.get_subagent
        SUBAGENT_REGISTRY = subagents_module.SUBAGENT_REGISTRY

        for name in SUBAGENT_REGISTRY.keys():
            agent = get_subagent(name)
            assert agent.AGENT_NAME == name

    def test_registry_maps_to_correct_classes(self):
        """SUBAGENT_REGISTRY deve mapear nomes para classes corretas."""
        SUBAGENT_REGISTRY = subagents_module.SUBAGENT_REGISTRY
        ClassificationAgent = subagents_module.ClassificationAgent
        AnomalyAgent = subagents_module.AnomalyAgent
        ForecastAgent = subagents_module.ForecastAgent
        RecommendationAgent = subagents_module.RecommendationAgent
        CampaignAgent = subagents_module.CampaignAgent
        AnalysisAgent = subagents_module.AnalysisAgent

        assert SUBAGENT_REGISTRY["classification"] is ClassificationAgent
        assert SUBAGENT_REGISTRY["anomaly"] is AnomalyAgent
        assert SUBAGENT_REGISTRY["forecast"] is ForecastAgent
        assert SUBAGENT_REGISTRY["recommendation"] is RecommendationAgent
        assert SUBAGENT_REGISTRY["campaign"] is CampaignAgent
        assert SUBAGENT_REGISTRY["analysis"] is AnalysisAgent
