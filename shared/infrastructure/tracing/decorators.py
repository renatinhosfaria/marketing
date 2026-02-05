"""
Decorador para criar spans automáticos em funções.
"""
import time
import traceback
import functools
import asyncio
from typing import Callable, Any

from shared.infrastructure.logging.structlog_config import get_logger
from shared.infrastructure.tracing.context import (
    generate_span_id,
    get_trace_context,
    set_trace_context
)

logger = get_logger("tracing.decorator")


def log_span(
    event_name: str,
    log_args: bool = True,
    log_result: bool = True
):
    """
    Decorador que cria span automático para uma função.

    Args:
        event_name: Nome base do evento (ex: "subagent_classification_execution")
        log_args: Se deve logar argumentos da função
        log_result: Se deve logar resultado da função

    Exemplo:
        @log_span("subagent_execution")
        async def run_subagent(task):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # 1. Salva contexto anterior
            parent_context = get_trace_context()
            parent_span = parent_context["span_id"]

            # 2. Gera novo span
            span_id = generate_span_id()
            set_trace_context(span_id=span_id, parent_span_id=parent_span)

            # 3. Log start
            log_data = {
                "event": f"{event_name}_start",
                **get_trace_context(),
                "function": func.__name__
            }
            if log_args:
                log_data["function_args"] = kwargs

            logger.info(**log_data)

            start_time = time.time()
            try:
                # 4. Executa função
                result = await func(*args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000

                # 5. Log end (success)
                log_data = {
                    "event": f"{event_name}_end",
                    **get_trace_context(),
                    "status": "success",
                    "duration_ms": duration_ms
                }
                if log_result:
                    log_data["result_summary"] = _summarize_result(result)

                logger.info(**log_data)
                return result

            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000

                # 6. Log error
                logger.error(
                    event=f"{event_name}_error",
                    **get_trace_context(),
                    error_type=type(e).__name__,
                    error_message=str(e),
                    error_stack=traceback.format_exc(),
                    duration_before_error_ms=duration_ms
                )
                raise
            finally:
                # 7. Restaura span anterior
                set_trace_context(span_id=parent_span)

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Versão síncrona (mesmo código, sem async/await)
            parent_context = get_trace_context()
            parent_span = parent_context["span_id"]
            span_id = generate_span_id()
            set_trace_context(span_id=span_id, parent_span_id=parent_span)

            log_data = {
                "event": f"{event_name}_start",
                **get_trace_context(),
                "function": func.__name__
            }
            if log_args:
                log_data["function_args"] = kwargs

            logger.info(**log_data)

            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000

                log_data = {
                    "event": f"{event_name}_end",
                    **get_trace_context(),
                    "status": "success",
                    "duration_ms": duration_ms
                }
                if log_result:
                    log_data["result_summary"] = _summarize_result(result)

                logger.info(**log_data)
                return result

            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                logger.error(
                    event=f"{event_name}_error",
                    **get_trace_context(),
                    error_type=type(e).__name__,
                    error_message=str(e),
                    error_stack=traceback.format_exc(),
                    duration_before_error_ms=duration_ms
                )
                raise
            finally:
                set_trace_context(span_id=parent_span)

        # Retorna wrapper apropriado
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


def _summarize_result(result: Any) -> Any:
    """
    Cria resumo do resultado para logging.
    Evita logar objetos muito grandes.
    """
    if result is None:
        return None

    # Para strings/números, retorna direto
    if isinstance(result, (str, int, float, bool)):
        return result

    # Para listas, retorna tamanho + preview
    if isinstance(result, list):
        return {
            "type": "list",
            "length": len(result),
            "preview": result[:3] if len(result) > 3 else result
        }

    # Para dicts, retorna keys
    if isinstance(result, dict):
        return {
            "type": "dict",
            "keys": list(result.keys())
        }

    # Para objetos, retorna tipo
    return {
        "type": type(result).__name__
    }
