"""
Funções de logging para tracing e observabilidade.
"""
from typing import Optional, List, Dict, Any

from shared.infrastructure.logging.structlog_config import get_logger
from shared.infrastructure.tracing.context import get_trace_context

logger = get_logger("tracing.events")

_SENSITIVE_KEYS = {
    "access_token",
    "token",
    "api_key",
    "authorization",
    "password",
    "secret",
    "app_secret",
    "key_hash",
}


def _get_logging_options() -> dict[str, Any]:
    """Retorna opções seguras e estáveis de logging."""
    return {
        "log_full_prompts": False,
        "log_full_responses": False,
        "log_full_tool_data": False,
        "preview_chars": 500,
    }


def _truncate_text(value: str, max_chars: int) -> str:
    """Trunca texto para preview de log."""
    if len(value) <= max_chars:
        return value
    return f"{value[:max_chars]}...[truncated]"


def _is_sensitive_key(key: Any) -> bool:
    """Identifica nomes de chave sensíveis para redaction."""
    key_lower = str(key).lower()
    return key_lower in _SENSITIVE_KEYS or any(marker in key_lower for marker in ("token", "secret", "password"))


def _sanitize_value(value: Any, *, max_chars: int) -> Any:
    """Sanitiza valor arbitrário recursivamente para log seguro."""
    if isinstance(value, dict):
        return {
            key: "***REDACTED***" if _is_sensitive_key(key) else _sanitize_value(val, max_chars=max_chars)
            for key, val in value.items()
        }

    if isinstance(value, list):
        return [_sanitize_value(item, max_chars=max_chars) for item in value]

    if isinstance(value, tuple):
        return tuple(_sanitize_value(item, max_chars=max_chars) for item in value)

    if isinstance(value, str):
        return _truncate_text(value, max_chars=max_chars)

    return value


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
    options = _get_logging_options()
    prompt_preview = _truncate_text(prompt, max_chars=options["preview_chars"])
    response_preview = _truncate_text(response, max_chars=options["preview_chars"])

    prompt_payload = {
        **get_trace_context(),
        "prompt_type": prompt_type,
        "prompt_preview": prompt_preview,
        "prompt_tokens": prompt_tokens,
        "model": model,
    }
    if options["log_full_prompts"]:
        prompt_payload["prompt_full"] = prompt

    logger.debug(
        "llm_prompt_sent",
        **prompt_payload,
    )

    response_payload = {
        **get_trace_context(),
        "response_preview": response_preview,
        "response_tokens": response_tokens,
        "duration_ms": duration_ms,
    }
    if options["log_full_responses"]:
        response_payload["response_full"] = response

    logger.debug(
        "llm_response_received",
        **response_payload,
    )


# ============================================================================
# Eventos de Subagentes
# ============================================================================

def log_subagent_started(
    subagent: str,
    task_description: str
):
    """Loga início da execução de um subagente"""
    logger.info(
        "subagent_started",
        **get_trace_context(),
        subagent=subagent,
        task_description=task_description
    )


def log_subagent_completed(
    subagent: str,
    success: bool,
    duration_ms: float,
    tool_calls: Optional[List[str]] = None
):
    """Loga conclusão de um subagente"""
    logger.info(
        "subagent_completed",
        **get_trace_context(),
        subagent=subagent,
        success=success,
        duration_ms=duration_ms,
        tool_calls=tool_calls or []
    )


def log_subagent_failed(
    subagent: str,
    error_type: str,
    error_message: str,
    duration_ms: float
):
    """Loga falha na execução de um subagente"""
    logger.error(
        "subagent_failed",
        **get_trace_context(),
        subagent=subagent,
        error_type=error_type,
        error_message=error_message,
        duration_ms=duration_ms
    )


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
    options = _get_logging_options()
    sanitized_params = _sanitize_value(params, max_chars=options["preview_chars"])
    sanitized_result = _sanitize_value(result, max_chars=options["preview_chars"])

    payload = {
        **get_trace_context(),
        "tool_name": tool_name,
        "tool_params": sanitized_params,
        "status": status,
        "duration_ms": duration_ms,
    }
    if options["log_full_tool_data"]:
        payload["result_full"] = sanitized_result
    else:
        payload["result_preview"] = _truncate_text(str(sanitized_result), max_chars=options["preview_chars"])

    logger.info(
        "tool_call_end",
        **payload,
    )


def log_tool_call_error(
    tool_name: str,
    params: Dict[str, Any],
    error_type: str,
    error_message: str,
    duration_ms: float
):
    """Loga erro em execução de ferramenta"""
    options = _get_logging_options()
    sanitized_params = _sanitize_value(params, max_chars=options["preview_chars"])

    logger.error(
        "tool_call_error",
        **get_trace_context(),
        tool_name=tool_name,
        tool_params=sanitized_params,
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


def log_synthesis_completed(
    success: bool,
    duration_ms: float,
    response_length: int,
    model: Optional[str] = None,
    tokens_used: Optional[int] = None
):
    """Loga conclusão da síntese"""
    logger.info(
        "synthesis_completed",
        **get_trace_context(),
        success=success,
        duration_ms=duration_ms,
        response_length=response_length,
        model=model,
        tokens_used=tokens_used
    )
