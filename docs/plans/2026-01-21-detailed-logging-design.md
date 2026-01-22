# Sistema de Logging Detalhado para Agente IA

**Data**: 2026-01-21
**Status**: Design Aprovado
**Autor**: Sistema de Brainstorming

## Visão Geral

Implementação de sistema de logging detalhado para o agente de IA multi-agente, oferecendo visibilidade completa de:
- **Execução técnica** (A): Entrada/saída de funções, parâmetros, tempos de execução
- **Raciocínio do agente** (B): Detecção de intent, seleção de subagentes, decisões, síntese
- **Auditoria** (D): Histórico completo de interações, usuários, dados processados

## Requisitos Validados

### 1. Nível de Detalhe
- **Logs completos sempre** (opção A)
- Inclui prompts completos do LLM
- Inclui respostas completas do LLM
- Inclui todos os parâmetros e resultados de ferramentas
- Volume esperado: ~5-10 MB/dia em staging

### 2. Estrutura dos Logs
- **Logs hierárquicos com trace_id** (opção B)
- Cada requisição tem `trace_id` único
- Spans criam hierarquia (orquestrador → subagentes → tools)
- Formato JSON para parsing estruturado
- Compatível com OpenTelemetry para visualização futura

### 3. Correlação entre Sistemas
- **Abordagem híbrida** (opção D)
- Backend Node.js pode enviar `X-Trace-ID` via header
- Se não receber, ML gera novo trace_id
- Permite correlacionar logs backend + ML
- Funciona standalone ou integrado

### 4. Dados Sensíveis
- **Logs em texto claro** (opção A)
- Nenhum mascaramento de dados
- Máxima transparência para debugging
- ⚠️ Requer controle de acesso aos logs

### 5. Retenção
- **Docker gerencia rotação** (opção D)
- Logs enviados para stdout
- Docker Compose configurado: max 100MB por arquivo, 10 arquivos = 1GB
- Simplicidade: sem gestão manual de arquivos

---

## Arquitetura do Sistema

### Camadas

```
┌─────────────────────────────────────────────────────────────┐
│ Camada 4: Docker Logging (stdout → json-file driver)       │
└─────────────────────────────────────────────────────────────┘
                              ▲
┌─────────────────────────────────────────────────────────────┐
│ Camada 3: Event Logging (funções específicas por evento)   │
│   - log_intent_detected()                                   │
│   - log_subagents_selected()                                │
│   - log_tool_call()                                         │
│   - log_llm_call()                                          │
└─────────────────────────────────────────────────────────────┘
                              ▲
┌─────────────────────────────────────────────────────────────┐
│ Camada 2: Span Management (@log_span decorator)            │
│   - Gera span_id único                                      │
│   - Mantém hierarquia parent_span_id                        │
│   - Registra duração, status, exceções                      │
└─────────────────────────────────────────────────────────────┘
                              ▲
┌─────────────────────────────────────────────────────────────┐
│ Camada 1: Trace Context (middleware FastAPI)               │
│   - Captura/gera trace_id                                   │
│   - Injeta no contexto via contextvars                      │
│   - Propaga automaticamente para todos os logs             │
└─────────────────────────────────────────────────────────────┘
```

### Fluxo de Trace e Spans

```
Request → Backend Node.js
            ↓ (X-Trace-ID: abc-123)
            ↓
         ML API (FastAPI)
            ↓
    Trace Middleware
    ├─ trace_id: abc-123 (do header)
    └─ span_id: root
            ↓
    Orchestrator (span-1, parent=root)
    ├─ parse_request (span-2, parent=span-1)
    │  └─ LLM call (logs dentro do span-2)
    ├─ plan_execution (span-3, parent=span-1)
    ├─ dispatch (span-4, parent=span-1)
    ├─ subagent_classification (span-5, parent=span-4)
    │  ├─ tool: classify_campaigns (span-6, parent=span-5)
    │  └─ tool: get_tier_distribution (span-7, parent=span-5)
    ├─ subagent_analysis (span-8, parent=span-4)
    │  └─ tool: get_campaign_metrics (span-9, parent=span-8)
    ├─ collect_results (span-10, parent=span-1)
    └─ synthesize (span-11, parent=span-1)
       └─ LLM call (logs dentro do span-11)
```

---

## Eventos Detalhados

### Orquestrador

#### 1. Request Received
```json
{
  "event": "orchestrator_request_received",
  "trace_id": "abc-123",
  "span_id": "span-1",
  "timestamp": "2026-01-21T10:30:00.123Z",
  "user_id": 123,
  "config_id": 1,
  "user_query": "Analise as campanhas da última semana",
  "request_metadata": {
    "ip": "192.168.1.1",
    "user_agent": "Mozilla/5.0...",
    "session_id": "sess-456"
  }
}
```

#### 2. Intent Detection
```json
{
  "event": "intent_detection_start",
  "trace_id": "abc-123",
  "span_id": "span-2",
  "parent_span_id": "span-1",
  "timestamp": "2026-01-21T10:30:00.150Z",
  "llm_provider": "openai",
  "llm_model": "gpt-4.1-2025-04-14"
}
```

```json
{
  "event": "llm_prompt_sent",
  "trace_id": "abc-123",
  "span_id": "span-2",
  "timestamp": "2026-01-21T10:30:00.151Z",
  "prompt_type": "intent_detection",
  "prompt_full": "Você é um assistente especializado...\n\nQuery do usuário: Analise as campanhas da última semana",
  "prompt_tokens": 245
}
```

```json
{
  "event": "llm_response_received",
  "trace_id": "abc-123",
  "span_id": "span-2",
  "timestamp": "2026-01-21T10:30:01.385Z",
  "response_full": "{\"intent\": \"analyze_campaigns\", \"confidence\": 0.95, \"reasoning\": \"Usuário solicitou análise explícita...\"}",
  "response_tokens": 87,
  "duration_ms": 1234
}
```

```json
{
  "event": "intent_detected",
  "trace_id": "abc-123",
  "span_id": "span-2",
  "timestamp": "2026-01-21T10:30:01.390Z",
  "intent": "analyze_campaigns",
  "confidence": 0.95,
  "reasoning": "Usuário solicitou análise explícita de campanhas com período específico"
}
```

#### 3. Planning
```json
{
  "event": "execution_planning_start",
  "trace_id": "abc-123",
  "span_id": "span-3",
  "parent_span_id": "span-1",
  "timestamp": "2026-01-21T10:30:01.400Z",
  "detected_intent": "analyze_campaigns"
}
```

```json
{
  "event": "subagents_selected",
  "trace_id": "abc-123",
  "span_id": "span-3",
  "timestamp": "2026-01-21T10:30:01.420Z",
  "subagents": ["classification", "analysis"],
  "reasoning": "Intent analyze_campaigns requer classification para categorizar + analysis para métricas detalhadas",
  "estimated_parallel_execution": true
}
```

#### 4. Dispatch
```json
{
  "event": "subagents_dispatch_start",
  "trace_id": "abc-123",
  "span_id": "span-4",
  "parent_span_id": "span-1",
  "timestamp": "2026-01-21T10:30:01.430Z",
  "subagents_to_dispatch": ["classification", "analysis"],
  "parallel_execution": true
}
```

```json
{
  "event": "subagent_dispatched",
  "trace_id": "abc-123",
  "span_id": "span-4",
  "timestamp": "2026-01-21T10:30:01.435Z",
  "subagent": "classification",
  "task": {
    "agent_name": "classification",
    "query": "Analise as campanhas da última semana",
    "context": {...}
  }
}
```

### Subagentes

#### 5. Subagent Execution
```json
{
  "event": "subagent_execution_start",
  "trace_id": "abc-123",
  "span_id": "span-5",
  "parent_span_id": "span-4",
  "timestamp": "2026-01-21T10:30:01.440Z",
  "subagent": "classification",
  "task_received": {...},
  "available_tools": ["classify_campaigns", "get_campaign_tiers", "get_tier_distribution", "reclassify_campaign"]
}
```

#### 6. Tool Calls
```json
{
  "event": "tool_call_start",
  "trace_id": "abc-123",
  "span_id": "span-6",
  "parent_span_id": "span-5",
  "timestamp": "2026-01-21T10:30:01.450Z",
  "tool_name": "classify_campaigns",
  "tool_params": {
    "config_id": 1,
    "campaign_ids": [101, 102, 103],
    "window_days": 7
  }
}
```

```json
{
  "event": "tool_call_end",
  "trace_id": "abc-123",
  "span_id": "span-6",
  "timestamp": "2026-01-21T10:30:01.795Z",
  "tool_name": "classify_campaigns",
  "status": "success",
  "result_summary": "3 campanhas classificadas: 2 high-tier, 1 medium-tier",
  "result_full": [
    {"campaign_id": 101, "tier": "high", "confidence": 0.92},
    {"campaign_id": 102, "tier": "high", "confidence": 0.88},
    {"campaign_id": 103, "tier": "medium", "confidence": 0.75}
  ],
  "duration_ms": 345
}
```

#### 7. Subagent LLM Calls
```json
{
  "event": "subagent_llm_prompt_sent",
  "trace_id": "abc-123",
  "span_id": "span-5",
  "timestamp": "2026-01-21T10:30:02.000Z",
  "subagent": "classification",
  "prompt_full": "Você é um especialista em classificação de campanhas...",
  "prompt_tokens": 512
}
```

```json
{
  "event": "subagent_llm_response_received",
  "trace_id": "abc-123",
  "span_id": "span-5",
  "timestamp": "2026-01-21T10:30:04.100Z",
  "subagent": "classification",
  "response_full": "Baseado nas métricas fornecidas...",
  "response_tokens": 234,
  "duration_ms": 2100
}
```

#### 8. Subagent Completion
```json
{
  "event": "subagent_execution_end",
  "trace_id": "abc-123",
  "span_id": "span-5",
  "timestamp": "2026-01-21T10:30:04.330Z",
  "subagent": "classification",
  "status": "success",
  "result": {
    "agent": "classification",
    "data": {...},
    "summary": "Classificadas 3 campanhas: 2 high-tier (ROI médio 3.2x), 1 medium-tier (ROI 1.8x)"
  },
  "tools_used": ["classify_campaigns", "get_tier_distribution"],
  "total_duration_ms": 2890,
  "llm_calls": 1,
  "total_tokens": 746
}
```

### Coleta e Síntese

#### 9. Collect Results
```json
{
  "event": "results_collection_start",
  "trace_id": "abc-123",
  "span_id": "span-7",
  "parent_span_id": "span-1",
  "timestamp": "2026-01-21T10:30:04.600Z",
  "expected_subagents": ["classification", "analysis"]
}
```

```json
{
  "event": "subagent_result_received",
  "trace_id": "abc-123",
  "span_id": "span-7",
  "timestamp": "2026-01-21T10:30:04.605Z",
  "subagent": "classification",
  "status": "success",
  "result_summary": "3 campanhas classificadas",
  "received_at": "2026-01-21T10:30:04.605Z"
}
```

```json
{
  "event": "results_collection_end",
  "trace_id": "abc-123",
  "span_id": "span-7",
  "timestamp": "2026-01-21T10:30:07.800Z",
  "total_subagents": 2,
  "successful": 2,
  "failed": 0,
  "duration_ms": 3200
}
```

#### 10. Synthesis
```json
{
  "event": "synthesis_start",
  "trace_id": "abc-123",
  "span_id": "span-8",
  "parent_span_id": "span-1",
  "timestamp": "2026-01-21T10:30:07.810Z",
  "subagent_results_count": 2,
  "synthesis_strategy": "comprehensive"
}
```

```json
{
  "event": "synthesis_llm_prompt_sent",
  "trace_id": "abc-123",
  "span_id": "span-8",
  "timestamp": "2026-01-21T10:30:07.820Z",
  "prompt_type": "final_synthesis",
  "prompt_full": "Você é um assistente...\n\nResultados dos subagentes:\n\n**Classification:**\n...\n\n**Analysis:**\n...\n\nGere resposta final integrando todos os dados...",
  "prompt_tokens": 1834,
  "subagent_data_included": ["classification", "analysis"]
}
```

```json
{
  "event": "synthesis_llm_response_received",
  "trace_id": "abc-123",
  "span_id": "span-8",
  "timestamp": "2026-01-21T10:30:11.220Z",
  "response_full": "Com base na análise das campanhas da última semana, identificamos 3 campanhas ativas...",
  "response_tokens": 567,
  "duration_ms": 3400
}
```

```json
{
  "event": "synthesis_end",
  "trace_id": "abc-123",
  "span_id": "span-8",
  "timestamp": "2026-01-21T10:30:11.270Z",
  "final_response_preview": "Com base na análise das campanhas...",
  "total_tokens_used": 2401,
  "duration_ms": 3450
}
```

#### 11. Response Sent
```json
{
  "event": "orchestrator_response_sent",
  "trace_id": "abc-123",
  "span_id": "span-1",
  "timestamp": "2026-01-21T10:30:11.280Z",
  "status": "success",
  "total_duration_ms": 8934,
  "subagents_executed": ["classification", "analysis"],
  "total_llm_calls": 3,
  "total_tokens": 3393,
  "total_tool_calls": 2,
  "response_preview": "Com base na análise das campanhas da última semana..."
}
```

### Tratamento de Erros

#### Error in Subagent
```json
{
  "event": "subagent_execution_error",
  "trace_id": "abc-123",
  "span_id": "span-5",
  "timestamp": "2026-01-21T10:30:06.450Z",
  "subagent": "classification",
  "error_type": "ToolExecutionError",
  "error_message": "Database connection timeout",
  "error_stack": "Traceback (most recent call last):\n  File ...",
  "tool_being_executed": "classify_campaigns",
  "duration_before_error_ms": 5000,
  "will_retry": true,
  "retry_attempt": 1,
  "max_retries": 2
}
```

#### Error in Orchestrator
```json
{
  "event": "orchestrator_error",
  "trace_id": "abc-123",
  "span_id": "span-1",
  "timestamp": "2026-01-21T10:30:50.000Z",
  "error_type": "SubagentTimeoutError",
  "error_message": "Subagent 'forecast' exceeded timeout of 45s",
  "failed_subagents": ["forecast"],
  "successful_subagents": ["classification", "analysis"],
  "will_synthesize_partial": true
}
```

#### LLM Error
```json
{
  "event": "llm_call_error",
  "trace_id": "abc-123",
  "span_id": "span-8",
  "timestamp": "2026-01-21T10:30:11.500Z",
  "error_type": "RateLimitError",
  "error_message": "Rate limit exceeded for model gpt-4.1",
  "llm_provider": "openai",
  "will_retry": true,
  "retry_delay_seconds": 2
}
```

#### Request Failed
```json
{
  "event": "orchestrator_request_failed",
  "trace_id": "abc-123",
  "span_id": "span-1",
  "timestamp": "2026-01-21T10:30:55.000Z",
  "status": "error",
  "error_type": "MultipleSubagentsFailedError",
  "error_message": "2 of 3 subagents failed, cannot synthesize",
  "total_duration_ms": 12000,
  "partial_results_available": false
}
```

---

## Implementação Técnica

### Estrutura de Arquivos

```
app/core/
├── logging.py                    # Existente (structlog básico)
└── tracing/                      # Novo módulo
    ├── __init__.py              # Exports principais
    ├── context.py               # Trace/Span context management (contextvars)
    ├── middleware.py            # FastAPI middleware para trace_id
    ├── decorators.py            # @log_span decorator
    └── events.py                # Funções de log por evento
```

### 1. Trace Context (context.py)

```python
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

### 2. Middleware FastAPI (middleware.py)

```python
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
            path=request.url.path,
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

### 3. Decorador @log_span (decorators.py)

```python
"""
Decorador para criar spans automáticos em funções.
"""
import time
import traceback
import functools
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

### 4. Event Logging Functions (events.py)

```python
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

### 5. Modificações no Código Existente

#### app/main.py
```python
from app.core.logging import setup_logging
from app.core.tracing.middleware import TraceMiddleware

# Setup logging
log_level = os.getenv("LOG_LEVEL", "INFO")
setup_logging(log_level)

# Add trace middleware (PRIMEIRO middleware, antes de outros)
app.add_middleware(TraceMiddleware)
```

#### app/agent/orchestrator/nodes/parse_request.py
```python
from app.core.tracing import log_span, log_intent_detected, log_llm_call
import time

@log_span("intent_detection", log_args=True, log_result=False)
async def parse_request(state: OrchestratorState) -> OrchestratorState:
    """Detecta a intenção do usuário."""

    # ... código existente para construir prompt ...

    # Chamar LLM
    start = time.time()
    response = await llm.ainvoke(prompt)
    duration = (time.time() - start) * 1000

    # Logar chamada LLM
    log_llm_call(
        prompt=prompt_text,
        response=response.content,
        prompt_tokens=response.usage.prompt_tokens,
        response_tokens=response.usage.completion_tokens,
        duration_ms=duration,
        prompt_type="intent_detection",
        model=llm.model_name
    )

    # Parse resposta
    parsed = json.loads(response.content)

    # Logar intent detectado
    log_intent_detected(
        intent=parsed["intent"],
        confidence=parsed.get("confidence", 1.0),
        reasoning=parsed.get("reasoning", "")
    )

    return state
```

#### app/agent/subagents/base.py
```python
from app.core.tracing import log_span, log_tool_call, log_tool_call_error
import time

class BaseSubagent:
    @log_span("subagent_execution", log_args=True, log_result=True)
    async def run(self, task: SubagentTask) -> AgentResult:
        """Executa o subagente."""

        # Wrappear tools com logging
        wrapped_tools = [self._wrap_tool_with_logging(t) for t in self.tools]

        # ... resto do código existente ...

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

        tool.func = logged_tool
        return tool
```

---

## Configuração

### Variáveis de Ambiente (.env)

```env
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

### Docker Compose (docker-compose.yml)

```yaml
services:
  famachat-ml-api:
    # ... config existente ...

    environment:
      # ... env vars existentes ...
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

---

## Uso e Análise de Logs

### Comandos Básicos

```bash
# Ver logs em tempo real
docker-compose logs -f famachat-ml-api

# Ver logs de um trace_id específico (requisição completa)
docker-compose logs famachat-ml-api | grep '"trace_id":"abc-123"'

# Ver apenas eventos de um subagente
docker-compose logs famachat-ml-api | grep '"subagent":"classification"'

# Ver todos os erros
docker-compose logs famachat-ml-api | grep '"level":"error"'

# Ver chamadas LLM com prompts completos
docker-compose logs famachat-ml-api | grep '"event":"llm_prompt_sent"'

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

# Ver quais subagentes são mais usados
docker-compose logs famachat-ml-api | \
  jq -r 'select(.event=="subagent_execution_start") | .subagent' | \
  sort | uniq -c | sort -rn

# Total de tokens consumidos hoje
docker-compose logs --since 00:00 famachat-ml-api | \
  jq 'select(.event=="orchestrator_response_sent") | .total_tokens' | \
  awk '{sum+=$1} END {print "Total tokens:", sum}'

# Requisições que falharam
docker-compose logs famachat-ml-api | \
  jq 'select(.event=="orchestrator_request_failed") | {trace_id, error_type, error_message}'
```

### Debug de Requisição Específica

```bash
# 1. Obter trace_id da requisição problemática
TRACE_ID="abc-123"

# 2. Extrair todos os logs dessa requisição em ordem
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
  jq 'select(.event | contains("llm")) | {event, prompt_full, response_full}'

# 5. Ver só as chamadas de tools
docker-compose logs famachat-ml-api | \
  grep "$TRACE_ID" | \
  jq 'select(.event | contains("tool_call")) | {tool_name, tool_params, result_full}'
```

### Métricas de Performance

```bash
# P50/P95/P99 de latência (últimas 1000 requests)
docker-compose logs famachat-ml-api | \
  jq -r 'select(.event=="orchestrator_response_sent") | .total_duration_ms' | \
  tail -1000 | sort -n | \
  awk '{arr[NR]=$1} END {
    print "P50:", arr[int(NR*0.5)]
    print "P95:", arr[int(NR*0.95)]
    print "P99:", arr[int(NR*0.99)]
  }'

# Taxa de erro por hora (últimas 24h)
docker-compose logs --since 24h famachat-ml-api | \
  jq -r 'select(.event=="orchestrator_request_failed") | .timestamp' | \
  cut -c1-13 | uniq -c

# Custo estimado de tokens (assumindo $0.01 por 1k tokens)
docker-compose logs --since 24h famachat-ml-api | \
  jq 'select(.event=="orchestrator_response_sent") | .total_tokens' | \
  awk '{sum+=$1} END {print "Tokens:", sum, "| Custo: $" sum/1000*0.01}'

# Subagentes que mais falham
docker-compose logs famachat-ml-api | \
  jq -r 'select(.event=="subagent_execution_error") | .subagent' | \
  sort | uniq -c | sort -rn

# Tools mais lentos
docker-compose logs famachat-ml-api | \
  jq 'select(.event=="tool_call_end") | {tool: .tool_name, duration: .duration_ms}' | \
  jq -s 'group_by(.tool) | map({tool: .[0].tool, avg_ms: (map(.duration) | add / length)}) | sort_by(.avg_ms) | reverse'
```

---

## Próximos Passos

### Implementação
1. Criar estrutura de arquivos em `app/core/tracing/`
2. Implementar módulos: context, middleware, decorators, events
3. Modificar código existente para adicionar decoradores e logs
4. Atualizar configuração (.env, docker-compose.yml)
5. Testar com requisições reais
6. Validar que trace_id correlaciona backend ↔ ML

### Validação
1. Executar request completa e verificar logs
2. Confirmar hierarquia de spans correta
3. Verificar volume de logs (deve estar ~5-10 MB/dia)
4. Testar comandos de análise (grep, jq)
5. Simular erro e verificar logs de erro

### Rollout
1. Deploy em staging com LOG_LEVEL=DEBUG
2. Monitorar volume de logs por 1 semana
3. Coletar feedback da equipe sobre utilidade
4. Ajustar se necessário (adicionar/remover eventos)
5. Deploy em produção com LOG_LEVEL=INFO inicialmente
6. Habilitar DEBUG sob demanda para investigações

### Melhorias Futuras (Opcional)
1. Integração com Grafana + Loki para visualização
2. Dashboards com métricas em tempo real
3. Alertas automáticos (taxa erro, latência)
4. Sampling inteligente (10% de logs completos em produção)
5. Export para OpenTelemetry para APM completo

---

## Riscos e Mitigações

### Risco: Volume de logs muito alto
- **Mitigação**: Docker rotaciona automaticamente (100MB x 10 = 1GB max)
- **Mitigação**: Pode reduzir para LOG_LEVEL=INFO se necessário
- **Mitigação**: Comprimir logs antigos (`compress: true`)

### Risco: Dados sensíveis nos logs
- **Mitigação**: Controlar acesso aos logs (somente equipe autorizada)
- **Mitigação**: Não fazer commit de logs no git
- **Mitigação**: Considerar mascaramento futuro se necessário

### Risco: Performance degradada
- **Mitigação**: Structlog é extremamente rápido (async, non-blocking)
- **Mitigação**: Logs para stdout não bloqueiam
- **Mitigação**: Monitorar latência P95 antes/depois

### Risco: Trace context perdido em execução paralela
- **Mitigação**: `contextvars` é thread-safe e async-safe
- **Mitigação**: Cada subagente herda trace_id automaticamente
- **Mitigação**: Testar bem cenários paralelos

---

## Conclusão

Este design implementa um sistema de logging detalhado que oferece:

✅ **Visibilidade Completa**: Logs de execução técnica + raciocínio + auditoria
✅ **Correlação Total**: Trace ID permite seguir requisição do backend ao ML
✅ **Hierarquia Clara**: Spans mostram orquestrador → subagentes → tools
✅ **Debug Facilitado**: Filtrar por trace_id e ver tudo de uma requisição
✅ **Análise Poderosa**: Logs estruturados JSON + ferramentas (jq, grep)
✅ **Simplicidade**: Docker gerencia rotação, sem complexidade extra
✅ **Flexibilidade**: LOG_LEVEL controla verbosidade sem rebuild

O sistema está pronto para implementação e vai fornecer insights profundos sobre o comportamento do agente de IA.
