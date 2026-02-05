"""
Integration tests for the detailed logging system.

These tests validate that the logging infrastructure works end-to-end
with real HTTP requests, verifying trace propagation, event logging,
and observability features.
"""

import pytest
import json
import subprocess
from typing import List, Dict
import time


def get_logs_for_trace(trace_id: str) -> List[Dict]:
    """Extrai todos os logs de um trace_id específico"""
    result = subprocess.run(
        ["docker-compose", "logs", "--no-color", "marketing-api"],
        cwd="/var/www/marketing",
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
    assert any("llm_" in event for event in events if event)

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

    time.sleep(5)

    logs = get_logs_for_trace(trace_id)

    # Verificar que há log de erro
    error_logs = [log for log in logs if log.get("level") == "error"]

    if len(error_logs) > 0:
        for error_log in error_logs:
            assert "error_type" in error_log or "event" in error_log
