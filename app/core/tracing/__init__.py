"""
Sistema de tracing para logging detalhado do agente IA.
"""
from app.core.tracing.context import (
    generate_trace_id,
    generate_span_id,
    set_trace_context,
    get_trace_context,
    clear_trace_context
)

__all__ = [
    "generate_trace_id",
    "generate_span_id",
    "set_trace_context",
    "get_trace_context",
    "clear_trace_context"
]
