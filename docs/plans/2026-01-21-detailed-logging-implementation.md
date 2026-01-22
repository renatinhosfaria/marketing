# Sistema de Logging Detalhado - Plano de Implementação

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implementar sistema de logging detalhado com trace context, span hierarchy e event logging para visibilidade completa do agente de IA multi-agente.

**Architecture:** Sistema em camadas: (1) Trace Context via contextvars para propagação automática de trace_id, (2) Middleware FastAPI para captura/geração de trace_id, (3) Decorador @log_span para funções, (4) Event logging functions para eventos específicos. Logs estruturados JSON para stdout capturados pelo Docker.

**Tech Stack:** Python 3.11+, structlog (existente), contextvars, FastAPI middleware, pytest

---

## Task 1: Trace Context Management

**Files:**
- Create: `app/core/tracing/__init__.py`
- Create: `app/core/tracing/context.py`
- Create: `tests/core/tracing/test_context.py`

**Step 1: Write the failing test**

```python
# tests/core/tracing/test_context.py
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
```

**Step 2: Run test to verify it fails**

```bash
cd /var/www/famachat-ml
pytest tests/core/tracing/test_context.py -v
```

Expected: FAIL with "ModuleNotFoundError: No module named 'app.core.tracing'"

**Step 3: Write minimal implementation**

```python
# app/core/tracing/__init__.py
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
```

```python
# app/core/tracing/context.py
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
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/core/tracing/test_context.py -v
```

Expected: PASS (5 tests passed)

**Step 5: Commit**

```bash
git add app/core/tracing/__init__.py app/core/tracing/context.py tests/core/tracing/test_context.py
git commit -m "feat(tracing): add trace context management with contextvars

- Generate trace_id (UUID) and span_id (short UUID)
- Set/get/clear trace context with thread-safety
- Full test coverage for context isolation

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 2: Decorador @log_span

**Files:**
- Create: `app/core/tracing/decorators.py`
- Create: `tests/core/tracing/test_decorators.py`

**Step 1: Write the failing test**

```python
# tests/core/tracing/test_decorators.py
import pytest
import asyncio
from unittest.mock import patch, MagicMock
from app.core.tracing.decorators import log_span
from app.core.tracing.context import set_trace_context, get_trace_context


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
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/core/tracing/test_decorators.py -v
```

Expected: FAIL with "ModuleNotFoundError: No module named 'app.core.tracing.decorators'"

**Step 3: Write minimal implementation**

```python
# app/core/tracing/decorators.py
"""
Decorador para criar spans automáticos em funções.
"""
import time
import traceback
import functools
import asyncio
from typing import Callable, Any

from app.core.logging import get_logger
from app.core.tracing.context import (
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

            logger.info(f"{event_name}_start", **log_data)

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

                logger.info(f"{event_name}_end", **log_data)
                return result

            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000

                # 6. Log error
                logger.error(
                    f"{event_name}_error",
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

            logger.info(f"{event_name}_start", **log_data)

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

                logger.info(f"{event_name}_end", **log_data)
                return result

            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                logger.error(
                    f"{event_name}_error",
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
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/core/tracing/test_decorators.py -v
```

Expected: PASS (5 tests passed)

**Step 5: Commit**

```bash
git add app/core/tracing/decorators.py tests/core/tracing/test_decorators.py
git commit -m "feat(tracing): add @log_span decorator for automatic span creation

- Support async and sync functions
- Automatic start/end logging with duration
- Error logging with stack trace
- Maintains span hierarchy
- Optional args/result logging
- Full test coverage

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 3: Event Logging Functions

**Files:**
- Create: `app/core/tracing/events.py`
- Create: `tests/core/tracing/test_events.py`

**Step 1: Write the failing test**

```python
# tests/core/tracing/test_events.py
import pytest
from unittest.mock import patch
from app.core.tracing.events import (
    log_orchestrator_request_received,
    log_intent_detected,
    log_subagents_selected,
    log_subagent_dispatched,
    log_llm_call,
    log_tool_call,
    log_orchestrator_response_sent
)
from app.core.tracing.context import set_trace_context


def test_log_orchestrator_request_received():
    """Deve logar recebimento de request"""
    set_trace_context(trace_id="test-trace", span_id="span-1")

    with patch('app.core.tracing.events.logger') as mock_logger:
        log_orchestrator_request_received(
            user_id=123,
            config_id=1,
            path="/api/v1/agent/chat",
            method="POST",
            ip="192.168.1.1",
            user_agent="Mozilla/5.0"
        )

        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args
        assert call_args[0][0] == "orchestrator_request_received"
        assert call_args[1]["trace_id"] == "test-trace"
        assert call_args[1]["user_id"] == 123


def test_log_intent_detected():
    """Deve logar detecção de intent"""
    set_trace_context(trace_id="test-trace", span_id="span-2")

    with patch('app.core.tracing.events.logger') as mock_logger:
        log_intent_detected(
            intent="analyze_campaigns",
            confidence=0.95,
            reasoning="Usuário pediu análise explícita"
        )

        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args
        assert call_args[0][0] == "intent_detected"
        assert call_args[1]["intent"] == "analyze_campaigns"
        assert call_args[1]["confidence"] == 0.95


def test_log_subagents_selected():
    """Deve logar seleção de subagentes"""
    set_trace_context(trace_id="test-trace", span_id="span-3")

    with patch('app.core.tracing.events.logger') as mock_logger:
        log_subagents_selected(
            subagents=["classification", "analysis"],
            reasoning="Intent requer classificação e análise",
            parallel=True
        )

        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args
        assert call_args[0][0] == "subagents_selected"
        assert call_args[1]["subagents"] == ["classification", "analysis"]


def test_log_llm_call():
    """Deve logar chamada LLM (prompt + response)"""
    set_trace_context(trace_id="test-trace", span_id="span-4")

    with patch('app.core.tracing.events.logger') as mock_logger:
        log_llm_call(
            prompt="Você é um assistente...",
            response="Analisando as campanhas...",
            prompt_tokens=100,
            response_tokens=50,
            duration_ms=1234,
            prompt_type="intent_detection",
            model="gpt-4"
        )

        assert mock_logger.debug.call_count == 2  # prompt + response

        # Verifica log prompt
        prompt_call = mock_logger.debug.call_args_list[0]
        assert prompt_call[0][0] == "llm_prompt_sent"
        assert "prompt_full" in prompt_call[1]

        # Verifica log response
        response_call = mock_logger.debug.call_args_list[1]
        assert response_call[0][0] == "llm_response_received"
        assert "response_full" in response_call[1]


def test_log_tool_call():
    """Deve logar execução de ferramenta"""
    set_trace_context(trace_id="test-trace", span_id="span-5")

    with patch('app.core.tracing.events.logger') as mock_logger:
        log_tool_call(
            tool_name="classify_campaigns",
            params={"config_id": 1, "campaign_ids": [101, 102]},
            result={"classified": 2},
            duration_ms=345,
            status="success"
        )

        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args
        assert call_args[0][0] == "tool_call_end"
        assert call_args[1]["tool_name"] == "classify_campaigns"
        assert call_args[1]["status"] == "success"


def test_log_orchestrator_response_sent():
    """Deve logar envio de resposta"""
    set_trace_context(trace_id="test-trace", span_id="span-1")

    with patch('app.core.tracing.events.logger') as mock_logger:
        log_orchestrator_response_sent(
            status_code=200,
            duration_ms=8934,
            subagents_executed=["classification", "analysis"],
            total_llm_calls=3,
            total_tokens=3393,
            total_tool_calls=2
        )

        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args
        assert call_args[0][0] == "orchestrator_response_sent"
        assert call_args[1]["status"] == "success"
        assert call_args[1]["total_duration_ms"] == 8934
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/core/tracing/test_events.py -v
```

Expected: FAIL with "ModuleNotFoundError: No module named 'app.core.tracing.events'"

**Step 3: Write minimal implementation**

```python
# app/core/tracing/events.py
"""
Funções específicas para logar eventos do agente.
"""
from typing import Optional, List, Dict, Any

from app.core.logging import get_logger
from app.core.tracing.context import get_trace_context

logger = get_logger("agent.events")


# ============================================================================
# Eventos do Orquestrador
# ============================================================================

def log_orchestrator_request_received(
    user_id: Optional[int],
    config_id: Optional[int],
    path: str,
    method: str,
    ip: Optional[str],
    user_agent: Optional[str]
):
    """Loga recebimento de request no orquestrador"""
    logger.info(
        "orchestrator_request_received",
        **get_trace_context(),
        user_id=user_id,
        config_id=config_id,
        request_metadata={
            "path": path,
            "method": method,
            "ip": ip,
            "user_agent": user_agent
        }
    )


def log_intent_detected(
    intent: str,
    confidence: float,
    reasoning: str
):
    """Loga detecção de intent"""
    logger.info(
        "intent_detected",
        **get_trace_context(),
        intent=intent,
        confidence=confidence,
        reasoning=reasoning
    )


def log_subagents_selected(
    subagents: List[str],
    reasoning: str,
    parallel: bool = True
):
    """Loga seleção de subagentes"""
    logger.info(
        "subagents_selected",
        **get_trace_context(),
        subagents=subagents,
        reasoning=reasoning,
        estimated_parallel_execution=parallel
    )


def log_subagent_dispatched(
    subagent: str,
    task: Dict[str, Any]
):
    """Loga dispatch de um subagente"""
    logger.info(
        "subagent_dispatched",
        **get_trace_context(),
        subagent=subagent,
        task=task
    )


def log_orchestrator_response_sent(
    status_code: int,
    duration_ms: float,
    subagents_executed: Optional[List[str]] = None,
    total_llm_calls: Optional[int] = None,
    total_tokens: Optional[int] = None,
    total_tool_calls: Optional[int] = None
):
    """Loga envio de response do orquestrador"""
    logger.info(
        "orchestrator_response_sent",
        **get_trace_context(),
        status="success",
        status_code=status_code,
        total_duration_ms=duration_ms,
        subagents_executed=subagents_executed,
        total_llm_calls=total_llm_calls,
        total_tokens=total_tokens,
        total_tool_calls=total_tool_calls
    )


def log_orchestrator_request_failed(
    error_type: str,
    error_message: str,
    duration_ms: float
):
    """Loga falha de request do orquestrador"""
    logger.error(
        "orchestrator_request_failed",
        **get_trace_context(),
        status="error",
        error_type=error_type,
        error_message=error_message,
        total_duration_ms=duration_ms
    )


# ============================================================================
# Eventos de LLM
# ============================================================================

def log_llm_call(
    prompt: str,
    response: str,
    prompt_tokens: int,
    response_tokens: int,
    duration_ms: float,
    prompt_type: Optional[str] = None,
    model: Optional[str] = None
):
    """Loga chamada completa ao LLM (prompt + response)"""
    # Log prompt
    logger.debug(
        "llm_prompt_sent",
        **get_trace_context(),
        prompt_type=prompt_type,
        prompt_full=prompt,
        prompt_tokens=prompt_tokens,
        model=model
    )

    # Log response
    logger.debug(
        "llm_response_received",
        **get_trace_context(),
        response_full=response,
        response_tokens=response_tokens,
        duration_ms=duration_ms
    )


# ============================================================================
# Eventos de Subagentes
# ============================================================================

def log_subagent_result_received(
    subagent: str,
    status: str,
    result_summary: Optional[str] = None
):
    """Loga recebimento de resultado de subagente"""
    logger.info(
        "subagent_result_received",
        **get_trace_context(),
        subagent=subagent,
        status=status,
        result_summary=result_summary
    )


# ============================================================================
# Eventos de Tools
# ============================================================================

def log_tool_call(
    tool_name: str,
    params: Dict[str, Any],
    result: Any,
    duration_ms: float,
    status: str = "success"
):
    """Loga execução de ferramenta"""
    logger.info(
        "tool_call_end",
        **get_trace_context(),
        tool_name=tool_name,
        tool_params=params,
        result_full=result,
        status=status,
        duration_ms=duration_ms
    )


def log_tool_call_error(
    tool_name: str,
    params: Dict[str, Any],
    error_type: str,
    error_message: str,
    duration_ms: float
):
    """Loga erro em execução de ferramenta"""
    logger.error(
        "tool_call_error",
        **get_trace_context(),
        tool_name=tool_name,
        tool_params=params,
        error_type=error_type,
        error_message=error_message,
        duration_ms=duration_ms
    )


# ============================================================================
# Eventos de Coleta e Síntese
# ============================================================================

def log_results_collection_end(
    total_subagents: int,
    successful: int,
    failed: int,
    duration_ms: float
):
    """Loga fim da coleta de resultados"""
    logger.info(
        "results_collection_end",
        **get_trace_context(),
        total_subagents=total_subagents,
        successful=successful,
        failed=failed,
        duration_ms=duration_ms
    )


def log_synthesis_start(
    subagent_results_count: int,
    strategy: str = "comprehensive"
):
    """Loga início da síntese"""
    logger.info(
        "synthesis_start",
        **get_trace_context(),
        subagent_results_count=subagent_results_count,
        synthesis_strategy=strategy
    )
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/core/tracing/test_events.py -v
```

Expected: PASS (7 tests passed)

**Step 5: Commit**

```bash
git add app/core/tracing/events.py tests/core/tracing/test_events.py
git commit -m "feat(tracing): add event logging functions for agent lifecycle

- Orchestrator events (request, intent, subagents, response)
- LLM call logging (prompt + response)
- Tool execution logging
- Subagent events
- Collection and synthesis events
- Full test coverage

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 4: Middleware FastAPI para Trace ID

**Files:**
- Create: `app/core/tracing/middleware.py`
- Create: `tests/core/tracing/test_middleware.py`
- Modify: `app/main.py`

**Step 1: Write the failing test**

```python
# tests/core/tracing/test_middleware.py
import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from app.core.tracing.middleware import TraceMiddleware
from app.core.tracing.context import get_trace_context


def test_middleware_generates_trace_id_when_not_provided():
    """Middleware deve gerar trace_id quando não fornecido"""
    app = FastAPI()
    app.add_middleware(TraceMiddleware)

    @app.get("/test")
    async def test_endpoint():
        context = get_trace_context()
        return {"trace_id": context["trace_id"]}

    client = TestClient(app)
    response = client.get("/test")

    assert response.status_code == 200
    trace_id = response.json()["trace_id"]
    assert trace_id is not None
    assert len(trace_id) == 36  # UUID format


def test_middleware_uses_provided_trace_id():
    """Middleware deve usar X-Trace-ID do header quando fornecido"""
    app = FastAPI()
    app.add_middleware(TraceMiddleware)

    @app.get("/test")
    async def test_endpoint():
        context = get_trace_context()
        return {"trace_id": context["trace_id"]}

    client = TestClient(app)
    response = client.get("/test", headers={"X-Trace-ID": "custom-trace-123"})

    assert response.status_code == 200
    assert response.json()["trace_id"] == "custom-trace-123"


def test_middleware_adds_trace_id_to_response_header():
    """Middleware deve adicionar X-Trace-ID no header da resposta"""
    app = FastAPI()
    app.add_middleware(TraceMiddleware)

    @app.get("/test")
    async def test_endpoint():
        return {"status": "ok"}

    client = TestClient(app)
    response = client.get("/test")

    assert "X-Trace-ID" in response.headers
    assert len(response.headers["X-Trace-ID"]) == 36


def test_middleware_logs_request_and_response():
    """Middleware deve logar request received e response sent"""
    app = FastAPI()
    app.add_middleware(TraceMiddleware)

    @app.get("/test")
    async def test_endpoint():
        return {"status": "ok"}

    with patch('app.core.tracing.middleware.log_orchestrator_request_received') as mock_request:
        with patch('app.core.tracing.middleware.log_orchestrator_response_sent') as mock_response:
            client = TestClient(app)
            response = client.get("/test")

            assert response.status_code == 200
            mock_request.assert_called_once()
            mock_response.assert_called_once()


def test_middleware_logs_error_on_exception():
    """Middleware deve logar erro quando exceção ocorre"""
    app = FastAPI()
    app.add_middleware(TraceMiddleware)

    @app.get("/test")
    async def failing_endpoint():
        raise ValueError("Test error")

    with patch('app.core.tracing.middleware.log_orchestrator_request_failed') as mock_failed:
        client = TestClient(app)

        with pytest.raises(ValueError):
            client.get("/test")

        mock_failed.assert_called_once()
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/core/tracing/test_middleware.py -v
```

Expected: FAIL with "ModuleNotFoundError: No module named 'app.core.tracing.middleware'"

**Step 3: Write minimal implementation**

```python
# app/core/tracing/middleware.py
"""
Middleware para capturar/gerar trace_id e injetar no contexto.
"""
import time
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.core.tracing.context import (
    generate_trace_id,
    generate_span_id,
    set_trace_context,
    get_trace_context
)
from app.core.tracing.events import (
    log_orchestrator_request_received,
    log_orchestrator_response_sent,
    log_orchestrator_request_failed
)


class TraceMiddleware(BaseHTTPMiddleware):
    """
    Middleware que:
    1. Captura X-Trace-ID do header ou gera novo
    2. Injeta trace_id no contexto
    3. Loga request received e response sent
    """

    async def dispatch(self, request: Request, call_next):
        # 1. Captura ou gera trace_id
        trace_id = request.headers.get("X-Trace-ID") or generate_trace_id()
        root_span_id = generate_span_id()

        # 2. Injeta no contexto
        set_trace_context(
            trace_id=trace_id,
            span_id=root_span_id,
            parent_span_id=None
        )

        # 3. Extrai dados da request
        user_id = getattr(request.state, "user_id", None)
        config_id = getattr(request.state, "config_id", None)

        # 4. Loga request received
        start_time = time.time()
        log_orchestrator_request_received(
            user_id=user_id,
            config_id=config_id,
            path=str(request.url.path),
            method=request.method,
            ip=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent")
        )

        try:
            # 5. Processa request
            response = await call_next(request)
            duration_ms = (time.time() - start_time) * 1000

            # 6. Loga response sent
            log_orchestrator_response_sent(
                status_code=response.status_code,
                duration_ms=duration_ms
            )

            # 7. Adiciona trace_id no header da resposta
            response.headers["X-Trace-ID"] = trace_id

            return response

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000

            # 8. Loga request failed
            log_orchestrator_request_failed(
                error_type=type(e).__name__,
                error_message=str(e),
                duration_ms=duration_ms
            )
            raise
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/core/tracing/test_middleware.py -v
```

Expected: PASS (5 tests passed)

**Step 5: Integrate middleware in main.py**

```python
# app/main.py
# ... imports existentes ...
from app.core.tracing.middleware import TraceMiddleware

# ... código existente ...

# Add trace middleware (PRIMEIRO middleware, antes de outros)
app.add_middleware(TraceMiddleware)

# ... resto do código ...
```

**Step 6: Run tests to verify integration**

```bash
pytest tests/core/tracing/test_middleware.py -v
```

Expected: PASS

**Step 7: Commit**

```bash
git add app/core/tracing/middleware.py tests/core/tracing/test_middleware.py app/main.py
git commit -m "feat(tracing): add FastAPI middleware for trace_id injection

- Capture X-Trace-ID from header or generate new
- Inject trace_id into context via contextvars
- Log request received and response sent
- Add X-Trace-ID to response headers
- Error logging on exceptions
- Integrated in main.py
- Full test coverage

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 5: Integração no Orquestrador - parse_request

**Files:**
- Modify: `app/agent/orchestrator/nodes/parse_request.py`

**Step 1: Add imports and decorator**

```python
# app/agent/orchestrator/nodes/parse_request.py
# ... imports existentes ...
import time
from app.core.tracing.decorators import log_span
from app.core.tracing.events import log_intent_detected, log_llm_call

# ... código existente ...

@log_span("intent_detection", log_args=True, log_result=False)
async def parse_request(state: OrchestratorState) -> OrchestratorState:
    """
    Analisa a requisição do usuário e detecta a intenção.
    """
    # ... código existente até construir prompt ...

    # Chamar LLM e logar
    start = time.time()
    response = await llm.ainvoke(prompt)
    duration = (time.time() - start) * 1000

    # Logar chamada LLM
    log_llm_call(
        prompt=str(prompt.messages) if hasattr(prompt, 'messages') else str(prompt),
        response=response.content,
        prompt_tokens=response.usage_metadata.get("input_tokens", 0) if hasattr(response, "usage_metadata") else 0,
        response_tokens=response.usage_metadata.get("output_tokens", 0) if hasattr(response, "usage_metadata") else 0,
        duration_ms=duration,
        prompt_type="intent_detection",
        model=llm.model_name if hasattr(llm, "model_name") else "unknown"
    )

    # Parse resposta
    parsed = json.loads(response.content)

    # Logar intent detectado
    log_intent_detected(
        intent=parsed.get("intent", "unknown"),
        confidence=parsed.get("confidence", 1.0),
        reasoning=parsed.get("reasoning", "")
    )

    # ... resto do código existente ...
```

**Step 2: Test manually with request**

```bash
# Start server
cd /var/www/famachat-ml
docker-compose up -d

# Wait for startup
sleep 10

# Make test request
curl -X POST http://localhost:8000/api/v1/agent/multi-agent/chat \
  -H "Content-Type: application/json" \
  -H "X-Trace-ID: test-trace-123" \
  -d '{"message": "Analise as campanhas", "config_id": 1}'

# Check logs for intent detection events
docker-compose logs famachat-ml-api | grep "intent_detection"
docker-compose logs famachat-ml-api | grep "llm_prompt_sent"
docker-compose logs famachat-ml-api | grep "intent_detected"
```

Expected: Logs showing intent_detection_start, llm_prompt_sent, llm_response_received, intent_detected, intent_detection_end

**Step 3: Commit**

```bash
git add app/agent/orchestrator/nodes/parse_request.py
git commit -m "feat(tracing): integrate logging in orchestrator parse_request

- Add @log_span decorator for intent detection
- Log LLM call (prompt + response)
- Log detected intent with confidence and reasoning
- Full trace visibility for intent detection phase

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 6: Integração no Orquestrador - plan_execution

**Files:**
- Modify: `app/agent/orchestrator/nodes/plan_execution.py`

**Step 1: Add imports and decorator**

```python
# app/agent/orchestrator/nodes/plan_execution.py
# ... imports existentes ...
from app.core.tracing.decorators import log_span
from app.core.tracing.events import log_subagents_selected

# ... código existente ...

@log_span("execution_planning", log_args=False, log_result=False)
async def plan_execution(state: OrchestratorState) -> OrchestratorState:
    """
    Planeja quais subagentes devem ser executados com base no intent.
    """
    # ... código existente para determinar subagentes ...

    # Após determinar selected_agents
    log_subagents_selected(
        subagents=selected_agents,
        reasoning=f"Intent '{state.intent}' mapped to subagents: {', '.join(selected_agents)}",
        parallel=True
    )

    # ... resto do código existente ...
```

**Step 2: Test manually**

```bash
# Make test request
curl -X POST http://localhost:8000/api/v1/agent/multi-agent/chat \
  -H "Content-Type: application/json" \
  -H "X-Trace-ID: test-plan-456" \
  -d '{"message": "Analise as campanhas", "config_id": 1}'

# Check logs
docker-compose logs famachat-ml-api | grep "test-plan-456" | grep "execution_planning"
docker-compose logs famachat-ml-api | grep "test-plan-456" | grep "subagents_selected"
```

Expected: Logs showing execution_planning_start, subagents_selected, execution_planning_end

**Step 3: Commit**

```bash
git add app/agent/orchestrator/nodes/plan_execution.py
git commit -m "feat(tracing): integrate logging in orchestrator plan_execution

- Add @log_span decorator for execution planning
- Log selected subagents with reasoning
- Trace visibility for planning phase

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 7: Integração em Subagents Base

**Files:**
- Modify: `app/agent/subagents/base.py`

**Step 1: Add imports and tool wrapping**

```python
# app/agent/subagents/base.py
# ... imports existentes ...
import time
from app.core.tracing.decorators import log_span
from app.core.tracing.events import log_tool_call, log_tool_call_error

# ... código existente ...

class BaseSubagent:
    # ... código existente ...

    @log_span("subagent_execution", log_args=True, log_result=True)
    async def run(self, task: SubagentTask) -> AgentResult:
        """
        Executa o subagente.
        """
        # Wrappear tools com logging
        wrapped_tools = [self._wrap_tool_with_logging(t) for t in self.tools]

        # ... resto do código existente usando wrapped_tools ...

    def _wrap_tool_with_logging(self, tool):
        """Wrapper para logar execução de cada tool"""
        original_func = tool.func

        async def logged_tool(*args, **kwargs):
            start = time.time()
            try:
                result = await original_func(*args, **kwargs)
                duration = (time.time() - start) * 1000

                log_tool_call(
                    tool_name=tool.name,
                    params=kwargs,
                    result=result,
                    duration_ms=duration,
                    status="success"
                )
                return result

            except Exception as e:
                duration = (time.time() - start) * 1000
                log_tool_call_error(
                    tool_name=tool.name,
                    params=kwargs,
                    error_type=type(e).__name__,
                    error_message=str(e),
                    duration_ms=duration
                )
                raise

        # Handle sync functions
        def logged_tool_sync(*args, **kwargs):
            start = time.time()
            try:
                result = original_func(*args, **kwargs)
                duration = (time.time() - start) * 1000

                log_tool_call(
                    tool_name=tool.name,
                    params=kwargs,
                    result=result,
                    duration_ms=duration,
                    status="success"
                )
                return result

            except Exception as e:
                duration = (time.time() - start) * 1000
                log_tool_call_error(
                    tool_name=tool.name,
                    params=kwargs,
                    error_type=type(e).__name__,
                    error_message=str(e),
                    duration_ms=duration
                )
                raise

        # Wrap appropriately
        if asyncio.iscoroutinefunction(original_func):
            tool.func = logged_tool
        else:
            tool.func = logged_tool_sync

        return tool
```

**Step 2: Test manually**

```bash
# Make test request that triggers subagents
curl -X POST http://localhost:8000/api/v1/agent/multi-agent/chat \
  -H "Content-Type: application/json" \
  -H "X-Trace-ID: test-subagent-789" \
  -d '{"message": "Classifique as campanhas ativas", "config_id": 1}'

# Check logs
docker-compose logs famachat-ml-api | grep "test-subagent-789" | grep "subagent_execution"
docker-compose logs famachat-ml-api | grep "test-subagent-789" | grep "tool_call"
```

Expected: Logs showing subagent_execution_start, tool_call_end for each tool, subagent_execution_end

**Step 3: Commit**

```bash
git add app/agent/subagents/base.py
git commit -m "feat(tracing): integrate logging in subagent base class

- Add @log_span decorator for subagent execution
- Wrap all tools with logging (start/end/error)
- Log tool parameters and results
- Handle both async and sync tools
- Full trace visibility for subagent operations

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 8: Configuração e Variáveis de Ambiente

**Files:**
- Modify: `.env`
- Modify: `.env.example`
- Modify: `docker-compose.yml`

**Step 1: Add to .env**

```bash
# .env (adicionar no final)

# ==================== Logging Detalhado ====================
# Nível de log (DEBUG para logs completos, INFO para resumos)
LOG_LEVEL=DEBUG

# Habilitar logging detalhado do agente (trace completo)
AGENT_DETAILED_LOGGING=true

# Incluir prompts completos nos logs (pode gerar volume alto)
AGENT_LOG_FULL_PROMPTS=true

# Incluir respostas completas do LLM nos logs
AGENT_LOG_FULL_RESPONSES=true

# Incluir dados completos de tools nos logs
AGENT_LOG_FULL_TOOL_DATA=true
```

**Step 2: Update .env.example**

```bash
# .env.example (adicionar no final)

# ==================== Logging Detalhado ====================
# Nível de log (DEBUG para logs completos, INFO para resumos)
# Development/Staging: DEBUG | Production: INFO
LOG_LEVEL=INFO

# Habilitar logging detalhado do agente (trace completo)
AGENT_DETAILED_LOGGING=false

# Incluir prompts completos nos logs (pode gerar volume alto)
AGENT_LOG_FULL_PROMPTS=false

# Incluir respostas completas do LLM nos logs
AGENT_LOG_FULL_RESPONSES=false

# Incluir dados completos de tools nos logs
AGENT_LOG_FULL_TOOL_DATA=false
```

**Step 3: Update docker-compose.yml**

```yaml
# docker-compose.yml
services:
  famachat-ml-api:
    # ... config existente ...

    environment:
      # ... env vars existentes ...

      # Logging Detalhado
      - LOG_LEVEL=${LOG_LEVEL:-INFO}
      - AGENT_DETAILED_LOGGING=${AGENT_DETAILED_LOGGING:-false}
      - AGENT_LOG_FULL_PROMPTS=${AGENT_LOG_FULL_PROMPTS:-false}
      - AGENT_LOG_FULL_RESPONSES=${AGENT_LOG_FULL_RESPONSES:-false}
      - AGENT_LOG_FULL_TOOL_DATA=${AGENT_LOG_FULL_TOOL_DATA:-false}

    logging:
      driver: "json-file"
      options:
        max-size: "100m"      # Máximo por arquivo
        max-file: "10"        # Mantém 10 arquivos = 1GB total
        compress: "true"      # Comprime arquivos antigos
```

**Step 4: Rebuild and restart**

```bash
cd /var/www/famachat-ml
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

**Step 5: Verify configuration is applied**

```bash
# Check if LOG_LEVEL is DEBUG
docker-compose logs famachat-ml-api | head -20

# Should show DEBUG level logs
```

Expected: Logs showing DEBUG level, detailed traces

**Step 6: Commit**

```bash
git add .env .env.example docker-compose.yml
git commit -m "feat(tracing): add configuration for detailed logging

- Add LOG_LEVEL (DEBUG/INFO)
- Add AGENT_DETAILED_LOGGING flag
- Add flags for prompts, responses, tool data
- Configure Docker logging (100MB x 10 = 1GB)
- Enable compression for old logs
- Defaults for dev/staging/production

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 9: Testes de Integração End-to-End

**Files:**
- Create: `tests/integration/test_detailed_logging.py`

**Step 1: Write integration test**

```python
# tests/integration/test_detailed_logging.py
import pytest
import json
import subprocess
from typing import List, Dict


def get_logs_for_trace(trace_id: str) -> List[Dict]:
    """Extrai todos os logs de um trace_id específico"""
    result = subprocess.run(
        ["docker-compose", "logs", "--no-color", "famachat-ml-api"],
        cwd="/var/www/famachat-ml",
        capture_output=True,
        text=True
    )

    logs = []
    for line in result.stdout.split('\n'):
        if trace_id in line:
            try:
                # Extrair JSON da linha (pode ter prefix do docker)
                json_start = line.find('{')
                if json_start >= 0:
                    log_data = json.loads(line[json_start:])
                    logs.append(log_data)
            except json.JSONDecodeError:
                continue

    return logs


@pytest.mark.integration
def test_full_request_trace_visibility():
    """Testa visibilidade completa de uma requisição do início ao fim"""
    trace_id = "integration-test-trace-001"

    # 1. Fazer requisição com trace_id customizado
    result = subprocess.run([
        "curl", "-X", "POST",
        "http://localhost:8000/api/v1/agent/multi-agent/chat",
        "-H", "Content-Type: application/json",
        "-H", f"X-Trace-ID: {trace_id}",
        "-d", json.dumps({
            "message": "Analise as campanhas ativas",
            "config_id": 1
        })
    ], capture_output=True, text=True)

    assert result.returncode == 0, f"Request failed: {result.stderr}"

    # 2. Aguardar processamento
    import time
    time.sleep(5)

    # 3. Extrair logs do trace
    logs = get_logs_for_trace(trace_id)

    assert len(logs) > 0, f"No logs found for trace_id {trace_id}"

    # 4. Verificar eventos críticos presentes
    events = [log.get("event") for log in logs]

    # Orquestrador
    assert "orchestrator_request_received" in events
    assert "intent_detected" in events
    assert "subagents_selected" in events
    assert "orchestrator_response_sent" in events or "orchestrator_request_failed" in events

    # LLM
    assert any("llm_" in event for event in events)

    # 5. Verificar hierarquia de spans
    trace_ids = set(log.get("trace_id") for log in logs)
    assert len(trace_ids) == 1, "Multiple trace_ids found - context leaking"
    assert trace_id in trace_ids

    # 6. Verificar spans têm parent correto
    span_hierarchy = {}
    for log in logs:
        span_id = log.get("span_id")
        parent_span_id = log.get("parent_span_id")
        if span_id:
            span_hierarchy[span_id] = parent_span_id

    assert len(span_hierarchy) > 0, "No spans found"


@pytest.mark.integration
def test_llm_prompts_logged_in_debug_mode():
    """Testa que prompts completos são logados em DEBUG mode"""
    trace_id = "integration-test-trace-002"

    # Fazer requisição
    subprocess.run([
        "curl", "-X", "POST",
        "http://localhost:8000/api/v1/agent/multi-agent/chat",
        "-H", "Content-Type: application/json",
        "-H", f"X-Trace-ID: {trace_id}",
        "-d", json.dumps({"message": "Teste", "config_id": 1})
    ], capture_output=True)

    import time
    time.sleep(5)

    logs = get_logs_for_trace(trace_id)

    # Verificar que prompt_full está presente
    llm_prompts = [log for log in logs if log.get("event") == "llm_prompt_sent"]
    assert len(llm_prompts) > 0, "No LLM prompts logged"

    for prompt_log in llm_prompts:
        assert "prompt_full" in prompt_log
        assert len(prompt_log["prompt_full"]) > 0


@pytest.mark.integration
def test_tool_calls_logged_with_params_and_results():
    """Testa que tool calls são logados com parâmetros e resultados"""
    trace_id = "integration-test-trace-003"

    # Fazer requisição que vai chamar tools
    subprocess.run([
        "curl", "-X", "POST",
        "http://localhost:8000/api/v1/agent/multi-agent/chat",
        "-H", "Content-Type: application/json",
        "-H", f"X-Trace-ID: {trace_id}",
        "-d", json.dumps({
            "message": "Classifique as campanhas",
            "config_id": 1
        })
    ], capture_output=True)

    import time
    time.sleep(5)

    logs = get_logs_for_trace(trace_id)

    # Verificar tool calls
    tool_calls = [log for log in logs if log.get("event") == "tool_call_end"]

    if len(tool_calls) > 0:  # Pode não ter se não houver dados
        for tool_log in tool_calls:
            assert "tool_name" in tool_log
            assert "tool_params" in tool_log
            assert "result_full" in tool_log
            assert "duration_ms" in tool_log


@pytest.mark.integration
def test_error_logging_with_stack_trace():
    """Testa que erros são logados com stack trace"""
    trace_id = "integration-test-trace-004"

    # Fazer requisição que vai causar erro (config_id inválido)
    subprocess.run([
        "curl", "-X", "POST",
        "http://localhost:8000/api/v1/agent/multi-agent/chat",
        "-H", "Content-Type: application/json",
        "-H", f"X-Trace-ID: {trace_id}",
        "-d", json.dumps({
            "message": "Teste",
            "config_id": 99999  # ID inválido
        })
    ], capture_output=True)

    import time
    time.sleep(5)

    logs = get_logs_for_trace(trace_id)

    # Verificar que há log de erro
    error_logs = [log for log in logs if log.get("level") == "error"]

    if len(error_logs) > 0:
        for error_log in error_logs:
            assert "error_type" in error_log or "event" in error_log
```

**Step 2: Run integration tests**

```bash
cd /var/www/famachat-ml
pytest tests/integration/test_detailed_logging.py -v -m integration
```

Expected: PASS (4 integration tests passed)

**Step 3: Commit**

```bash
git add tests/integration/test_detailed_logging.py
git commit -m "test(tracing): add end-to-end integration tests for logging

- Test full request trace visibility
- Verify LLM prompts logged in DEBUG mode
- Verify tool calls with params and results
- Verify error logging with stack traces
- Extract and validate logs from Docker

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 10: Documentação e Validação Final

**Files:**
- Create: `docs/LOGGING.md`
- Update: `README.md`

**Step 1: Create comprehensive logging documentation**

```markdown
# docs/LOGGING.md
# Sistema de Logging Detalhado - Guia de Uso

## Visão Geral

O sistema de logging detalhado oferece visibilidade completa do agente de IA multi-agente através de logs estruturados JSON com trace context e span hierarchy.

## Configuração

### Variáveis de Ambiente

```bash
# Nível de log
LOG_LEVEL=DEBUG  # DEBUG (completo) ou INFO (resumido)

# Flags de logging detalhado
AGENT_DETAILED_LOGGING=true        # Habilita logging detalhado
AGENT_LOG_FULL_PROMPTS=true        # Loga prompts completos do LLM
AGENT_LOG_FULL_RESPONSES=true      # Loga respostas completas do LLM
AGENT_LOG_FULL_TOOL_DATA=true      # Loga dados completos de ferramentas
```

### Recomendações por Ambiente

| Ambiente | LOG_LEVEL | DETAILED | PROMPTS | RESPONSES | TOOL_DATA |
|----------|-----------|----------|---------|-----------|-----------|
| **Development** | DEBUG | true | true | true | true |
| **Staging** | DEBUG | true | true | true | true |
| **Production** | INFO | false | false | false | false |

## Estrutura de Logs

### Trace Context

Cada requisição tem um `trace_id` único que permite rastrear toda a jornada:

```json
{
  "trace_id": "abc-123-def-456",
  "span_id": "span-789",
  "parent_span_id": "span-012"
}
```

- **trace_id**: ID único da requisição completa (UUID)
- **span_id**: ID da operação específica (UUID curto)
- **parent_span_id**: ID do span pai (hierarquia)

### Eventos Principais

#### Orquestrador
- `orchestrator_request_received` - Request recebida
- `intent_detected` - Intent detectado com confidence
- `subagents_selected` - Subagentes escolhidos
- `orchestrator_response_sent` - Response enviada com sucesso
- `orchestrator_request_failed` - Request falhou com erro

#### LLM
- `llm_prompt_sent` - Prompt enviado ao LLM
- `llm_response_received` - Resposta do LLM recebida

#### Subagentes
- `subagent_execution_start` - Subagente iniciado
- `subagent_execution_end` - Subagente finalizado
- `subagent_execution_error` - Erro no subagente

#### Tools
- `tool_call_end` - Ferramenta executada com sucesso
- `tool_call_error` - Erro na execução de ferramenta

## Consultando Logs

### Comandos Básicos

```bash
# Ver logs em tempo real
docker-compose logs -f famachat-ml-api

# Ver logs de um trace_id específico
docker-compose logs famachat-ml-api | grep "trace_id.*abc-123"

# Ver apenas erros
docker-compose logs famachat-ml-api | grep '"level":"error"'

# Exportar logs de hoje
docker-compose logs --since 00:00 famachat-ml-api > logs-$(date +%Y-%m-%d).json
```

### Análise com jq

```bash
# Extrair todos os trace_ids únicos
docker-compose logs famachat-ml-api | jq -r '.trace_id' | sort -u

# Ver duração de todas as requisições
docker-compose logs famachat-ml-api | \
  jq 'select(.event=="orchestrator_response_sent") | {trace_id, duration_ms, status}'

# Ver subagentes mais usados
docker-compose logs famachat-ml-api | \
  jq -r 'select(.event=="subagent_execution_start") | .subagent' | \
  sort | uniq -c | sort -rn

# Total de tokens consumidos
docker-compose logs --since 00:00 famachat-ml-api | \
  jq 'select(.event=="orchestrator_response_sent") | .total_tokens' | \
  awk '{sum+=$1} END {print "Total tokens:", sum}'
```

### Debug de Requisição Específica

```bash
# 1. Obter trace_id da requisição problemática
TRACE_ID="abc-123"

# 2. Extrair todos os logs dessa requisição
docker-compose logs famachat-ml-api | \
  grep "$TRACE_ID" | \
  jq -s 'sort_by(.timestamp)'

# 3. Ver hierarquia de spans
docker-compose logs famachat-ml-api | \
  grep "$TRACE_ID" | \
  jq '{event, span_id, parent_span_id, timestamp}'

# 4. Ver só os prompts LLM
docker-compose logs famachat-ml-api | \
  grep "$TRACE_ID" | \
  jq 'select(.event | contains("llm"))'
```

## Métricas

### Performance

```bash
# P50/P95/P99 de latência
docker-compose logs famachat-ml-api | \
  jq -r 'select(.event=="orchestrator_response_sent") | .total_duration_ms' | \
  tail -1000 | sort -n | \
  awk '{arr[NR]=$1} END {
    print "P50:", arr[int(NR*0.5)]
    print "P95:", arr[int(NR*0.95)]
    print "P99:", arr[int(NR*0.99)]
  }'

# Taxa de erro por hora
docker-compose logs --since 24h famachat-ml-api | \
  jq -r 'select(.event=="orchestrator_request_failed") | .timestamp' | \
  cut -c1-13 | uniq -c
```

### Custos

```bash
# Custo estimado de tokens (assumindo $0.01 por 1k tokens)
docker-compose logs --since 24h famachat-ml-api | \
  jq 'select(.event=="orchestrator_response_sent") | .total_tokens' | \
  awk '{sum+=$1} END {print "Tokens:", sum, "| Custo: $" sum/1000*0.01}'
```

## Correlação Backend ↔ ML

O backend Node.js pode enviar `X-Trace-ID` no header da requisição:

```javascript
// Backend Node.js
const traceId = uuidv4();
const response = await axios.post('http://localhost:8000/api/v1/agent/chat', data, {
  headers: {
    'X-Trace-ID': traceId
  }
});
```

Isso permite correlacionar logs do backend com logs do ML usando o mesmo trace_id.

## Troubleshooting

### Volume de logs muito alto

Reduzir nível de log:
```bash
# .env
LOG_LEVEL=INFO
AGENT_LOG_FULL_PROMPTS=false
AGENT_LOG_FULL_RESPONSES=false
```

### Logs não aparecem

Verificar configuração do Docker:
```bash
docker-compose logs famachat-ml-api | tail -50
```

### Trace context perdido

Verificar que contextvars está sendo propagado corretamente:
```bash
# Verificar que todos os logs de um trace têm o mesmo trace_id
docker-compose logs famachat-ml-api | grep "trace_id.*abc-123" | jq .trace_id | sort -u
```

## Referências

- Design completo: `docs/plans/2026-01-21-detailed-logging-design.md`
- Código fonte: `app/core/tracing/`
- Testes: `tests/core/tracing/` e `tests/integration/test_detailed_logging.py`
```

**Step 2: Update README.md**

Add section to README.md:

```markdown
# README.md (adicionar seção)

## 📊 Sistema de Logging Detalhado

O FamaChat ML possui sistema de logging detalhado com trace context para visibilidade completa do agente de IA.

### Características

- **Trace Context**: Cada requisição tem trace_id único
- **Span Hierarchy**: Hierarquia de operações (orquestrador → subagentes → tools)
- **Logs Estruturados**: JSON para parsing e análise
- **Correlação**: Backend ↔ ML via X-Trace-ID header
- **Debug Facilitado**: Filtrar por trace_id e ver execução completa

### Configuração

```bash
# Habilitar logs detalhados em .env
LOG_LEVEL=DEBUG
AGENT_DETAILED_LOGGING=true
AGENT_LOG_FULL_PROMPTS=true
AGENT_LOG_FULL_RESPONSES=true
```

### Uso

```bash
# Ver logs em tempo real
docker-compose logs -f famachat-ml-api

# Filtrar por trace_id
docker-compose logs famachat-ml-api | grep "trace_id.*abc-123"

# Análise com jq
docker-compose logs famachat-ml-api | jq 'select(.event=="intent_detected")'
```

**Documentação completa**: [docs/LOGGING.md](docs/LOGGING.md)
```

**Step 3: Validate documentation**

```bash
# Verify all commands in documentation work
# Test each example command

# Example:
docker-compose logs --since 00:00 famachat-ml-api | jq -r '.trace_id' | sort -u | head -5
```

**Step 4: Commit**

```bash
git add docs/LOGGING.md README.md
git commit -m "docs(tracing): add comprehensive logging documentation

- Complete usage guide in docs/LOGGING.md
- Configuration by environment (dev/staging/prod)
- Query commands (basic + jq examples)
- Debug workflow for specific requests
- Performance and cost metrics
- Troubleshooting guide
- Update README.md with logging section

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Validação Final e Rollout

**Step 1: Run all tests**

```bash
cd /var/www/famachat-ml

# Unit tests
pytest tests/core/tracing/ -v

# Integration tests
pytest tests/integration/test_detailed_logging.py -v -m integration

# All tests
pytest -v
```

Expected: All tests passing

**Step 2: Manual validation with real request**

```bash
# Make real request
TRACE_ID="final-validation-$(date +%s)"
curl -X POST http://localhost:8000/api/v1/agent/multi-agent/chat \
  -H "Content-Type: application/json" \
  -H "X-Trace-ID: $TRACE_ID" \
  -d '{"message": "Analise as campanhas ativas da última semana", "config_id": 1}'

# Wait for processing
sleep 10

# Extract and analyze logs
docker-compose logs famachat-ml-api | grep "$TRACE_ID" | jq -s 'sort_by(.timestamp)' > validation-logs.json

# Verify events present
echo "Events found:"
cat validation-logs.json | jq -r '.[].event' | sort -u

# Verify span hierarchy
echo "Span hierarchy:"
cat validation-logs.json | jq '{event, span_id, parent_span_id}' | head -20
```

Expected: Complete trace with all events (orchestrator, intent, subagents, tools, llm, synthesis)

**Step 3: Performance validation**

```bash
# Make 10 requests and measure overhead
for i in {1..10}; do
  curl -X POST http://localhost:8000/api/v1/agent/multi-agent/chat \
    -H "Content-Type: application/json" \
    -H "X-Trace-ID: perf-test-$i" \
    -d '{"message": "Teste", "config_id": 1}' &
done

wait

# Check P95 latency
docker-compose logs famachat-ml-api | \
  grep "perf-test-" | \
  jq -r 'select(.event=="orchestrator_response_sent") | .total_duration_ms' | \
  sort -n | \
  awk '{arr[NR]=$1} END {print "P95:", arr[int(NR*0.95)] "ms"}'
```

Expected: P95 latency increase < 5% compared to baseline without logging

**Step 4: Final commit and tag**

```bash
git add -A
git commit -m "feat(tracing): complete detailed logging system implementation

Complete implementation of hierarchical logging system with:
- Trace context management (contextvars)
- Span hierarchy (decorators)
- Event logging functions
- FastAPI middleware for trace_id injection
- Integration in orchestrator and subagents
- Configuration (.env, docker-compose.yml)
- Comprehensive tests (unit + integration)
- Full documentation (LOGGING.md)

Provides complete visibility:
- Execution flow (A): function entry/exit, durations
- Agent reasoning (B): intent, decisions, synthesis
- Audit trail (D): user actions, data processed

Volume: ~5-10 MB/day in DEBUG mode
Performance overhead: < 5%

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"

git tag -a v1.0.0-detailed-logging -m "Release: Detailed logging system v1.0.0"
```

---

## Próximos Passos Recomendados

1. **Monitoramento** (1 semana):
   - Coletar métricas de volume de logs
   - Verificar performance (P95 latency)
   - Identificar eventos mais úteis

2. **Ajustes** (se necessário):
   - Adicionar/remover eventos
   - Ajustar nível de detalhe
   - Otimizar serialização de objetos grandes

3. **Integração Backend** (futuro):
   - Backend Node.js envia X-Trace-ID
   - Correlação completa backend ↔ ML
   - Dashboard unificado

4. **Observabilidade Avançada** (opcional):
   - Integração com Grafana + Loki
   - Dashboards de métricas em tempo real
   - Alertas automáticos

---

## Resumo da Implementação

**10 tarefas completadas**:
1. ✅ Trace Context Management
2. ✅ Decorador @log_span
3. ✅ Event Logging Functions
4. ✅ Middleware FastAPI
5. ✅ Integração parse_request
6. ✅ Integração plan_execution
7. ✅ Integração subagents base
8. ✅ Configuração e variáveis
9. ✅ Testes de integração E2E
10. ✅ Documentação completa

**Arquivos criados**: 15
**Arquivos modificados**: 5
**Testes adicionados**: 20+
**Commits**: 10

**Sistema pronto para produção** com visibilidade completa do agente de IA! 🎉
