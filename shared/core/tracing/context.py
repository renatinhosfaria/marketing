"""
Backwards compatibility layer.
Use shared.infrastructure.tracing.context instead.
"""
from shared.infrastructure.tracing.context import (
    generate_trace_id,
    generate_span_id,
    set_trace_context,
    get_trace_context,
    clear_trace_context,
    trace_id_var,
    span_id_var,
    parent_span_id_var,
)

__all__ = [
    "generate_trace_id",
    "generate_span_id",
    "set_trace_context",
    "get_trace_context",
    "clear_trace_context",
    "trace_id_var",
    "span_id_var",
    "parent_span_id_var",
]
