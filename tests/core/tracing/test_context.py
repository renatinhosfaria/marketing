import pytest
from shared.core.tracing.context import (
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


@pytest.mark.asyncio
async def test_concurrent_context_isolation():
    """Contextos devem ser isolados entre requisições concorrentes.

    Este teste valida que contextvars mantém isolamento entre
    tasks async executando concorrentemente, evitando vazamento
    de contexto entre requisições diferentes.
    """
    import asyncio

    async def simulate_request(request_id: int):
        """Simula uma requisição com seu próprio contexto"""
        # Definir contexto único para esta requisição
        trace_id = f"trace-{request_id}"
        span_id = f"span-{request_id}"
        set_trace_context(trace_id=trace_id, span_id=span_id)

        # Simular processamento com delay
        await asyncio.sleep(0.01)

        # Verificar que contexto ainda é o correto após await
        context = get_trace_context()
        assert context["trace_id"] == trace_id, \
            f"Request {request_id}: trace_id vazou! Esperado {trace_id}, obtido {context['trace_id']}"
        assert context["span_id"] == span_id, \
            f"Request {request_id}: span_id vazou! Esperado {span_id}, obtido {context['span_id']}"

        # Simular mais processamento
        await asyncio.sleep(0.01)

        # Verificar novamente
        context = get_trace_context()
        assert context["trace_id"] == trace_id
        assert context["span_id"] == span_id

        return request_id

    # Executar 10 requisições concorrentes
    tasks = [simulate_request(i) for i in range(10)]
    results = await asyncio.gather(*tasks)

    # Todas as 10 requisições devem ter completado com sucesso
    assert len(results) == 10
    assert results == list(range(10))


@pytest.mark.asyncio
async def test_concurrent_context_with_nested_spans():
    """Contextos aninhados devem ser isolados entre requisições concorrentes.

    Testa que hierarquias de spans (parent → child) são mantidas
    corretamente mesmo com execução concorrente.
    """
    import asyncio

    async def simulate_request_with_nested_spans(request_id: int):
        """Simula requisição com spans aninhados"""
        # Span raiz
        root_trace_id = f"trace-{request_id}"
        root_span_id = f"root-{request_id}"
        set_trace_context(trace_id=root_trace_id, span_id=root_span_id, parent_span_id=None)

        await asyncio.sleep(0.01)

        # Verificar span raiz
        context = get_trace_context()
        assert context["trace_id"] == root_trace_id
        assert context["span_id"] == root_span_id
        assert context["parent_span_id"] is None

        # Criar span filho
        child_span_id = f"child-{request_id}"
        set_trace_context(span_id=child_span_id, parent_span_id=root_span_id)

        await asyncio.sleep(0.01)

        # Verificar span filho
        context = get_trace_context()
        assert context["trace_id"] == root_trace_id  # trace_id não muda
        assert context["span_id"] == child_span_id
        assert context["parent_span_id"] == root_span_id

        # Voltar para span raiz
        set_trace_context(span_id=root_span_id, parent_span_id=None)

        await asyncio.sleep(0.01)

        # Verificar que voltou para span raiz
        context = get_trace_context()
        assert context["trace_id"] == root_trace_id
        assert context["span_id"] == root_span_id
        assert context["parent_span_id"] is None

        return request_id

    # Executar 5 requisições com spans aninhados concorrentemente
    tasks = [simulate_request_with_nested_spans(i) for i in range(5)]
    results = await asyncio.gather(*tasks)

    assert len(results) == 5
    assert results == list(range(5))
