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
    log_orchestrator_request_received,
    log_orchestrator_response_sent,
    log_orchestrator_request_failed,
    log_intent_detected,
    log_subagents_selected,
    log_subagent_dispatched,
    log_subagent_started,
    log_subagent_completed,
    log_subagent_failed,
    log_subagent_result_received,
    log_llm_call,
    log_tool_call,
    log_tool_call_error,
    log_results_collection_end,
    log_synthesis_start,
    log_synthesis_completed,
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
    "log_orchestrator_request_received",
    "log_orchestrator_response_sent",
    "log_orchestrator_request_failed",
    "log_intent_detected",
    "log_subagents_selected",
    "log_subagent_dispatched",
    "log_subagent_started",
    "log_subagent_completed",
    "log_subagent_failed",
    "log_subagent_result_received",
    "log_llm_call",
    "log_tool_call",
    "log_tool_call_error",
    "log_results_collection_end",
    "log_synthesis_start",
    "log_synthesis_completed",
]
