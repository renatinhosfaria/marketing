"""Tracing module for distributed observability."""
from .context import (
    generate_trace_id,
    generate_span_id,
    set_trace_context,
    get_trace_context,
    clear_trace_context,
)
from .decorators import log_span
from .middleware import TraceMiddleware
from .events import (
    log_tool_call,
    log_tool_call_error,
)

__all__ = [
    # Context
    "generate_trace_id",
    "generate_span_id",
    "set_trace_context",
    "get_trace_context",
    "clear_trace_context",
    # Decorators
    "log_span",
    # Middleware
    "TraceMiddleware",
    # Events
    "log_tool_call",
    "log_tool_call_error",
]
