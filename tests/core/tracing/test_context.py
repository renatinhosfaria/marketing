import pytest
from app.core.tracing.context import (
    generate_trace_id,
    generate_span_id,
    set_trace_context,
    get_trace_context,
    clear_trace_context
)


def test_generate_trace_id_returns_uuid():
    """Trace ID deve ser um UUID válido"""
    trace_id = generate_trace_id()
    assert len(trace_id) == 36
    assert trace_id.count('-') == 4


def test_generate_span_id_returns_short_id():
    """Span ID deve ser UUID curto (8 chars)"""
    span_id = generate_span_id()
    assert len(span_id) == 8


def test_set_and_get_trace_context():
    """Deve armazenar e recuperar contexto de trace"""
    set_trace_context(
        trace_id="test-trace-123",
        span_id="span-456",
        parent_span_id="parent-789"
    )

    context = get_trace_context()
    assert context["trace_id"] == "test-trace-123"
    assert context["span_id"] == "span-456"
    assert context["parent_span_id"] == "parent-789"


def test_clear_trace_context():
    """Deve limpar contexto de trace"""
    set_trace_context(trace_id="test", span_id="span")
    clear_trace_context()

    context = get_trace_context()
    assert context["trace_id"] is None
    assert context["span_id"] is None
    assert context["parent_span_id"] is None


def test_context_isolation_between_calls():
    """Contextos devem ser isolados entre chamadas"""
    # Contexto 1
    set_trace_context(trace_id="trace-1", span_id="span-1")
    context1 = get_trace_context()

    # Contexto 2
    set_trace_context(trace_id="trace-2", span_id="span-2")
    context2 = get_trace_context()

    # Contexto 2 deve ter sobrescrito contexto 1
    assert context2["trace_id"] == "trace-2"
    assert context2["span_id"] == "span-2"
