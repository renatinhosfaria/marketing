"""
Metricas Prometheus do Agent.

Define todas as metricas de negocio, qualidade e operacionais
do modulo Agent para monitoramento via Prometheus/Grafana.
"""

from prometheus_client import Counter, Histogram, Gauge

# --- Contadores ---

agent_requests_total = Counter(
    "agent_requests_total", "Total de requests ao Agent API",
    ["endpoint", "status"],
)

agent_dispatches_total = Counter(
    "agent_dispatches_total", "Total de despachos para agentes",
    ["agent_id"],
)

agent_interrupts_total = Counter(
    "agent_interrupts_total", "Total de interrupts disparados",
    ["interrupt_type", "resolution"],
)

# --- Histogramas ---

agent_response_duration = Histogram(
    "agent_response_duration_seconds", "Tempo total de resposta",
    ["routing_urgency"],
    buckets=[1, 2, 5, 10, 20, 30, 60],
)

agent_subgraph_duration = Histogram(
    "agent_subgraph_duration_seconds", "Tempo por subgraph",
    ["agent_id"],
    buckets=[0.5, 1, 2, 5, 10, 20],
)

# --- Gauges ---

agent_active_streams = Gauge(
    "agent_active_streams", "Streams SSE ativos",
)

agent_store_memories = Gauge(
    "agent_store_memories_total", "Total de memorias no Store",
    ["namespace_type"],
)

# --- Metricas de Qualidade ---

forecast_accuracy = Histogram(
    "agent_forecast_mape", "MAPE das previsoes",
    ["metric", "entity_type"],
    buckets=[0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0],
)

recommendation_resolution = Counter(
    "agent_recommendation_resolution_total",
    "Resolucao de recomendacoes",
    ["resolution", "recommendation_type"],
)

action_impact = Histogram(
    "agent_action_impact_pct",
    "Impacto percentual das acoes",
    ["action_type", "metric"],
    buckets=[-50, -30, -20, -10, -5, 0, 5, 10, 20, 30, 50],
)

# --- Metricas Operacionais ---

tool_errors_total = Counter(
    "agent_tool_errors_total", "Erros em tools",
    ["tool_name", "error_code"],
)

ml_api_latency = Histogram(
    "agent_ml_api_duration_seconds", "Latencia ML API",
    ["method", "path", "status"],
    buckets=[0.1, 0.25, 0.5, 1, 2, 5, 10, 30],
)

store_operations = Histogram(
    "agent_store_operation_seconds", "Latencia Store",
    ["operation", "namespace_type"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1],
)

approval_token_failures = Counter(
    "agent_approval_token_failures_total",
    "Token de aprovacao invalido",
    ["action_type"],
)
