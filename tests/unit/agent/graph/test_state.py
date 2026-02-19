"""
Testes dos schemas de estado do grafo principal.

Testa:
  - SupervisorState: campos obrigatorios e reducers
  - AgentReport: contrato de retorno dos agentes
  - UserContext: dados do usuario autenticado
  - AgentInput: input padrao para subgraphs
  - Reducer operator.add funciona para agent_reports
"""

import pytest
import operator

from langchain_core.messages import HumanMessage

from projects.agent.graph.state import (
    AgentReport,
    UserContext,
    AgentInput,
)


@pytest.mark.asyncio
async def test_agent_report_schema():
    """AgentReport deve ter todos os campos obrigatorios."""
    report: AgentReport = {
        "agent_id": "health_monitor",
        "status": "completed",
        "summary": "Nenhuma anomalia detectada.",
        "data": {"anomaly_count": 0},
        "confidence": 0.90,
    }

    assert report["agent_id"] == "health_monitor"
    assert report["status"] == "completed"
    assert report["confidence"] == 0.90
    assert report["data"]["anomaly_count"] == 0


@pytest.mark.asyncio
async def test_agent_report_error_status():
    """AgentReport com status error deve funcionar corretamente."""
    report: AgentReport = {
        "agent_id": "performance_analyst",
        "status": "error",
        "summary": "Falha na ML API: TimeoutError",
        "data": {"error_code": "SUBGRAPH_EXCEPTION", "retryable": False},
        "confidence": 0.0,
    }

    assert report["status"] == "error"
    assert report["confidence"] == 0.0
    assert "SUBGRAPH_EXCEPTION" in str(report["data"])


@pytest.mark.asyncio
async def test_user_context_schema():
    """UserContext deve conter user_id, account_id, account_name, timezone."""
    ctx: UserContext = {
        "user_id": "u123",
        "account_id": "act_987",
        "account_name": "Minha Conta",
        "timezone": "America/Sao_Paulo",
    }

    assert ctx["user_id"] == "u123"
    assert ctx["account_id"] == "act_987"
    assert ctx["timezone"] == "America/Sao_Paulo"


@pytest.mark.asyncio
async def test_agent_input_schema():
    """AgentInput deve conter messages e scope (compativel com Send())."""
    agent_input: AgentInput = {
        "messages": [HumanMessage(content="Analise CPL")],
        "scope": {
            "entity_type": "campaign",
            "entity_ids": ["c1"],
            "lookback_days": 7,
            "top_n": 10,
        },
    }

    assert len(agent_input["messages"]) == 1
    assert agent_input["scope"]["entity_type"] == "campaign"
    assert agent_input["scope"]["entity_ids"] == ["c1"]


@pytest.mark.asyncio
async def test_agent_reports_reducer_add():
    """Reducer operator.add deve acumular agent_reports de multiplos agentes."""
    reports_a = [
        {
            "agent_id": "health_monitor",
            "status": "completed",
            "summary": "OK",
            "data": None,
            "confidence": 0.85,
        },
    ]
    reports_b = [
        {
            "agent_id": "performance_analyst",
            "status": "completed",
            "summary": "CPL estavel",
            "data": None,
            "confidence": 0.80,
        },
    ]

    combined = operator.add(reports_a, reports_b)

    assert len(combined) == 2
    agent_ids = [r["agent_id"] for r in combined]
    assert "health_monitor" in agent_ids
    assert "performance_analyst" in agent_ids


@pytest.mark.asyncio
async def test_agent_input_with_none_scope():
    """AgentInput com scope None deve funcionar (para queries genericas)."""
    agent_input: AgentInput = {
        "messages": [HumanMessage(content="Como estao meus anuncios?")],
        "scope": None,
    }

    assert agent_input["scope"] is None
    assert len(agent_input["messages"]) == 1
