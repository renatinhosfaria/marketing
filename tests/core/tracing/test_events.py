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


def test_log_subagent_dispatched():
    """Deve logar dispatch de subagente"""
    set_trace_context(trace_id="test-trace", span_id="span-6")

    with patch('app.core.tracing.events.logger') as mock_logger:
        log_subagent_dispatched(
            subagent="classification",
            task={"action": "classify", "campaign_ids": [101, 102]}
        )

        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args
        assert call_args[0][0] == "subagent_dispatched"
        assert call_args[1]["subagent"] == "classification"


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
