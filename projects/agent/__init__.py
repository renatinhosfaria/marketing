"""
Módulo do Agente de IA para Gestão de Tráfego Pago.

Este módulo implementa um agente conversacional usando LangGraph Python
que acessa os dados de Machine Learning para auxiliar na tomada de decisão
sobre campanhas de Facebook Ads.

Nota: usar imports lazy para evitar efeitos colaterais em processos que
apenas precisam de submódulos (ex.: db.models).
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from projects.agent.service import TrafficAgentService, get_agent_service
    from projects.agent.config import AgentSettings, get_agent_settings

__all__ = [
    "TrafficAgentService",
    "get_agent_service",
    "AgentSettings",
    "get_agent_settings",
]

_LAZY_IMPORTS = {
    "TrafficAgentService": ("projects.agent.service", "TrafficAgentService"),
    "get_agent_service": ("projects.agent.service", "get_agent_service"),
    "AgentSettings": ("projects.agent.config", "AgentSettings"),
    "get_agent_settings": ("projects.agent.config", "get_agent_settings"),
}


def __getattr__(name: str):
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_IMPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(_LAZY_IMPORTS.keys()))
