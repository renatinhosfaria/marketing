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
