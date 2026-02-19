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

# --- Metricas de SLO (semana 1 — confiabilidade operacional) ---

# Erros de stream SSE (taxa via rate(agent_stream_errors_total[5m]) no Grafana)
stream_errors_total = Counter(
    "agent_stream_errors_total",
    "Total de erros em streams SSE",
    ["error_type"],  # graph_error, timeout, cancelled
)

# Reconexoes SSE (suporte a replay — semana 2)
stream_reconnect_total = Counter(
    "agent_stream_reconnect_total",
    "Total de tentativas de reconexao SSE",
    ["status"],  # attempted, success, session_not_found
)

# Latencia ate o primeiro evento SSE (TTFB do stream)
stream_first_event_latency = Histogram(
    "agent_first_event_latency_seconds",
    "Latencia do primeiro evento SSE (time-to-first-byte)",
    buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0],
)

# Tempo total do stream ate evento done
stream_time_to_done = Histogram(
    "agent_time_to_done_seconds",
    "Tempo total do stream SSE ate evento done",
    buckets=[1, 2, 5, 10, 20, 30, 60, 120],
)

# Sessoes SSE orfas detectadas pelo reaper periodico
session_orphan_count = Gauge(
    "agent_session_orphan_count",
    "Sessoes SSE orfas detectadas",
)

# Circuit breakers abertos por dependencia (0=fechado, 1=aberto)
dependency_circuit_open = Gauge(
    "agent_dependency_circuit_open_total",
    "Circuit breakers abertos por dependencia",
    ["dependency"],  # ml_api, fb_api, store
)
