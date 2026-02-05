"""
Backwards compatibility layer.
Use shared.infrastructure.tracing instead.
"""
from shared.infrastructure.tracing import (
    generate_trace_id,
    generate_span_id,
    set_trace_context,
    get_trace_context,
    clear_trace_context,
)

__all__ = [
    "generate_trace_id",
    "generate_span_id",
    "set_trace_context",
    "get_trace_context",
    "clear_trace_context",
]
