"""
Funções de logging para tracing e observabilidade.
"""
from typing import Dict, Any

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
