"""
Módulo do Agente de IA para Gestão de Tráfego Pago.

Este módulo implementa um agente conversacional usando LangGraph Python
que acessa os dados de Machine Learning para auxiliar na tomada de decisão
sobre campanhas de Facebook Ads.
"""

from app.agent.service import TrafficAgentService, get_agent_service
from app.agent.config import AgentSettings, get_agent_settings

__all__ = [
    "TrafficAgentService",
    "get_agent_service",
    "AgentSettings",
    "get_agent_settings",
]
