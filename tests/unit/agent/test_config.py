"""Testes para configurações do multi-agent system."""
import sys
import os
import importlib.util

# Carregar o módulo config diretamente sem passar pelo __init__.py
# Isso evita dependências de outros módulos que podem ter requirements não instalados
config_path = os.path.join(
    os.path.dirname(__file__),
    '..', '..', '..',
    'app', 'agent', 'config.py'
)
config_path = os.path.abspath(config_path)

spec = importlib.util.spec_from_file_location("agent_config", config_path)
agent_config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agent_config)

AgentSettings = agent_config.AgentSettings
get_agent_settings = agent_config.get_agent_settings

import pytest


class TestMultiAgentConfig:
    """Testes para configurações multi-agente."""

    def test_multi_agent_enabled_default_false(self):
        """Multi-agent deve estar desabilitado por padrão."""
        settings = AgentSettings()
        assert settings.multi_agent_enabled is False

    def test_orchestrator_timeout_default(self):
        """Timeout do orchestrator deve ter valor padrão."""
        settings = AgentSettings()
        assert settings.orchestrator_timeout == 120

    def test_max_parallel_subagents_default(self):
        """Max parallel subagents deve ter valor padrão."""
        settings = AgentSettings()
        assert settings.max_parallel_subagents == 4

    def test_subagent_timeouts_exist(self):
        """Timeouts de subagentes devem existir."""
        settings = AgentSettings()
        assert hasattr(settings, 'timeout_classification')
        assert hasattr(settings, 'timeout_anomaly')
        assert hasattr(settings, 'timeout_forecast')
        assert hasattr(settings, 'timeout_recommendation')
        assert hasattr(settings, 'timeout_campaign')
        assert hasattr(settings, 'timeout_analysis')

    def test_subagent_timeout_values(self):
        """Timeouts devem ter valores corretos."""
        settings = AgentSettings()
        assert settings.timeout_classification == 30
        assert settings.timeout_anomaly == 30
        assert settings.timeout_forecast == 45
        assert settings.timeout_recommendation == 30
        assert settings.timeout_campaign == 20
        assert settings.timeout_analysis == 45

    def test_synthesis_config(self):
        """Configurações de síntese devem existir."""
        settings = AgentSettings()
        assert settings.synthesis_max_tokens == 4096
        assert settings.synthesis_temperature == 0.3

    def test_subagent_retry_config(self):
        """Configurações de retry devem existir."""
        settings = AgentSettings()
        assert settings.subagent_max_retries == 2
        assert settings.subagent_retry_delay == 1.0
