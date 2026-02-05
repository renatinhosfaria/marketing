import pytest
import asyncio
from unittest.mock import patch, MagicMock
from shared.core.tracing.decorators import log_span
from shared.core.tracing.context import set_trace_context, get_trace_context


@pytest.mark.asyncio
async def test_log_span_decorator_async_success():
    """Decorador deve criar span e logar início/fim"""
    set_trace_context(trace_id="test-trace", span_id="parent-span")

    with patch('app.core.tracing.decorators.logger') as mock_logger:
        @log_span("test_operation")
        async def async_function(value: int):
            return value * 2

        result = await async_function(5)

        assert result == 10
        assert mock_logger.info.call_count == 2  # start + end

        # Verifica log start
        start_call = mock_logger.info.call_args_list[0]
        assert "test_operation_start" in str(start_call)

        # Verifica log end
        end_call = mock_logger.info.call_args_list[1]
        assert "test_operation_end" in str(end_call)
        assert "success" in str(end_call)


@pytest.mark.asyncio
async def test_log_span_decorator_async_error():
    """Decorador deve logar erro quando exceção ocorre"""
    set_trace_context(trace_id="test-trace", span_id="parent-span")

    with patch('app.core.tracing.decorators.logger') as mock_logger:
        @log_span("test_operation")
        async def failing_function():
            raise ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            await failing_function()

        # Verifica log error
        assert mock_logger.error.call_count == 1
        error_call = mock_logger.error.call_args_list[0]
        assert "test_operation_error" in str(error_call)
        assert "ValueError" in str(error_call)


def test_log_span_decorator_sync_success():
    """Decorador deve funcionar com funções síncronas"""
    set_trace_context(trace_id="test-trace", span_id="parent-span")

    with patch('app.core.tracing.decorators.logger') as mock_logger:
        @log_span("sync_operation")
        def sync_function(value: int):
            return value * 3

        result = sync_function(4)

        assert result == 12
        assert mock_logger.info.call_count == 2


def test_log_span_maintains_hierarchy():
    """Decorador deve manter hierarquia de spans"""
    set_trace_context(trace_id="test-trace", span_id="parent-span")

    with patch('app.core.tracing.decorators.logger'):
        @log_span("outer_operation")
        def outer():
            context_during = get_trace_context()
            return context_during

        context = outer()

        # Durante execução, deve ter criado novo span com parent correto
        assert context["trace_id"] == "test-trace"
        assert context["span_id"] != "parent-span"  # novo span criado
        assert context["parent_span_id"] == "parent-span"  # parent preservado


def test_log_span_with_args_logging():
    """Decorador deve logar argumentos quando log_args=True"""
    with patch('app.core.tracing.decorators.logger') as mock_logger:
        @log_span("operation_with_args", log_args=True)
        def func_with_args(x: int, y: str):
            return f"{y}-{x}"

        result = func_with_args(x=10, y="test")

        start_call = mock_logger.info.call_args_list[0]
        assert "function_args" in str(start_call)
