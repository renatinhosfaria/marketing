"""
Observabilidade do Agent — metricas Prometheus e logging estruturado.

Exporta todas as metricas e helpers de logging para uso nos modulos do Agent.
"""

from projects.agent.observability.metrics import (
    agent_requests_total,
    agent_dispatches_total,
    agent_interrupts_total,
    agent_response_duration,
    agent_subgraph_duration,
    agent_active_streams,
    agent_store_memories,
    forecast_accuracy,
    recommendation_resolution,
    action_impact,
    tool_errors_total,
    ml_api_latency,
    store_operations,
    approval_token_failures,
)

from projects.agent.observability.logging import (
    log_agent_event,
    log_agent_error,
)

__all__ = [
    # Contadores
    "agent_requests_total",
    "agent_dispatches_total",
    "agent_interrupts_total",
    # Histogramas
    "agent_response_duration",
    "agent_subgraph_duration",
    # Gauges
    "agent_active_streams",
    "agent_store_memories",
    # Qualidade
    "forecast_accuracy",
    "recommendation_resolution",
    "action_impact",
    # Operacionais
    "tool_errors_total",
    "ml_api_latency",
    "store_operations",
    "approval_token_failures",
    # Logging
    "log_agent_event",
    "log_agent_error",
]
