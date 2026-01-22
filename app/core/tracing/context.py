"""
Gerenciamento de contexto de trace usando contextvars para thread-safety.
"""
import contextvars
import uuid
from typing import Optional, Dict


# Context vars (thread-safe para async)
trace_id_var = contextvars.ContextVar('trace_id', default=None)
span_id_var = contextvars.ContextVar('span_id', default=None)
parent_span_id_var = contextvars.ContextVar('parent_span_id', default=None)


def generate_trace_id() -> str:
    """Gera trace_id único (UUID4)"""
    return str(uuid.uuid4())


def generate_span_id() -> str:
    """Gera span_id único (UUID4 curto - primeiros 8 chars)"""
    return str(uuid.uuid4())[:8]


def set_trace_context(
    trace_id: Optional[str] = None,
    span_id: Optional[str] = None,
    parent_span_id: Optional[str] = None
) -> None:
    """
    Define contexto de trace atual.

    Args:
        trace_id: ID do trace (requisição completa)
        span_id: ID do span (operação específica)
        parent_span_id: ID do span pai (hierarquia)
    """
    if trace_id is not None:
        trace_id_var.set(trace_id)
    if span_id is not None:
        span_id_var.set(span_id)
    if parent_span_id is not None:
        parent_span_id_var.set(parent_span_id)


def get_trace_context() -> Dict[str, Optional[str]]:
    """
    Retorna contexto de trace atual.

    Returns:
        Dict com trace_id, span_id, parent_span_id
    """
    return {
        "trace_id": trace_id_var.get(),
        "span_id": span_id_var.get(),
        "parent_span_id": parent_span_id_var.get()
    }


def clear_trace_context() -> None:
    """Limpa contexto de trace (útil para cleanup)"""
    trace_id_var.set(None)
    span_id_var.set(None)
    parent_span_id_var.set(None)
