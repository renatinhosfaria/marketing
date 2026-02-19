"""
Testes dos nos individuais do subgraph Health Monitor.

Testa:
  - fetch_metrics_node: busca classificacoes via tool
  - anomaly_detection_node: detecta anomalias via tool
  - diagnose_node: gera diagnostico com LLM
  - _build_diagnosis_prompt: formato do prompt
  - Tratamento de tool results com ok=False
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from langchain_core.messages import HumanMessage, AIMessage


def _make_writer_mock():
    """Cria mock para get_stream_writer."""
    writer = MagicMock()
    return writer


@pytest.mark.asyncio
async def test_fetch_metrics_node_success():
    """fetch_metrics_node retorna classificacoes quando tool retorna ok=True."""
    mock_classifications = {"classifications": [{"entity_id": "c1", "tier": "HIGH_PERFORMER"}]}

    mock_get_classifications = AsyncMock()
    mock_get_classifications.ainvoke.return_value = {
        "ok": True,
        "data": mock_classifications,
        "error": None,
    }

    state = {
        "messages": [HumanMessage(content="Saude das campanhas")],
        "scope": {"entity_type": "campaign"},
        "metrics_ref": None,
        "anomaly_results": None,
        "classifications": None,
        "diagnosis": None,
        "agent_reports": [],
    }
    config = {"configurable": {"thread_id": "t1", "user_id": "u1", "account_id": "a1"}}

    with patch("projects.agent.subgraphs.health_monitor.nodes.get_stream_writer", return_value=_make_writer_mock()), \
         patch("projects.agent.subgraphs.health_monitor.nodes.get_classifications", mock_get_classifications):
        from projects.agent.subgraphs.health_monitor.nodes import fetch_metrics_node
        result = await fetch_metrics_node(state, config)

    assert result["classifications"] == mock_classifications


@pytest.mark.asyncio
async def test_fetch_metrics_node_tool_failure():
    """fetch_metrics_node retorna None quando tool retorna ok=False."""
    mock_get_classifications = AsyncMock()
    mock_get_classifications.ainvoke.return_value = {
        "ok": False,
        "data": None,
        "error": {"code": "TIMEOUT", "message": "ML API timeout"},
    }

    state = {
        "messages": [HumanMessage(content="Saude das campanhas")],
        "scope": {"entity_type": "campaign"},
        "metrics_ref": None,
        "anomaly_results": None,
        "classifications": None,
        "diagnosis": None,
        "agent_reports": [],
    }
    config = {"configurable": {"thread_id": "t1", "user_id": "u1", "account_id": "a1"}}

    with patch("projects.agent.subgraphs.health_monitor.nodes.get_stream_writer", return_value=_make_writer_mock()), \
         patch("projects.agent.subgraphs.health_monitor.nodes.get_classifications", mock_get_classifications):
        from projects.agent.subgraphs.health_monitor.nodes import fetch_metrics_node
        result = await fetch_metrics_node(state, config)

    assert result["classifications"] is None


@pytest.mark.asyncio
async def test_anomaly_detection_node_success():
    """anomaly_detection_node retorna anomalias detectadas."""
    mock_anomaly_data = {
        "detected_count": 2,
        "anomalies": [
            {"entity_id": "c1", "metric_name": "cpl", "severity": "HIGH"},
            {"entity_id": "c2", "metric_name": "ctr", "severity": "MEDIUM"},
        ],
    }

    mock_detect_anomalies = AsyncMock()
    mock_detect_anomalies.ainvoke.return_value = {
        "ok": True,
        "data": mock_anomaly_data,
        "error": None,
    }

    state = {
        "messages": [HumanMessage(content="Anomalias")],
        "scope": {"entity_type": "campaign", "entity_ids": None, "lookback_days": 1},
        "metrics_ref": None,
        "anomaly_results": None,
        "classifications": {},
        "diagnosis": None,
        "agent_reports": [],
    }
    config = {"configurable": {"thread_id": "t1", "user_id": "u1", "account_id": "a1"}}

    with patch("projects.agent.subgraphs.health_monitor.nodes.get_stream_writer", return_value=_make_writer_mock()), \
         patch("projects.agent.subgraphs.health_monitor.nodes.detect_anomalies", mock_detect_anomalies):
        from projects.agent.subgraphs.health_monitor.nodes import anomaly_detection_node
        result = await anomaly_detection_node(state, config)

    assert result["anomaly_results"]["detected_count"] == 2
    assert len(result["anomaly_results"]["anomalies"]) == 2


@pytest.mark.asyncio
async def test_diagnose_node_produces_agent_report():
    """diagnose_node deve produzir AgentReport com status completed."""
    mock_response = AIMessage(content="Diagnostico: CPL anomalo na campanha c1.")
    mock_model = AsyncMock()
    mock_model.ainvoke.return_value = mock_response

    state = {
        "messages": [HumanMessage(content="Diagnostico")],
        "scope": {},
        "metrics_ref": None,
        "anomaly_results": {
            "detected_count": 1,
            "anomalies": [{"entity_id": "c1", "severity": "HIGH"}],
        },
        "classifications": {"c1": "UNDERPERFORMER"},
        "diagnosis": None,
        "agent_reports": [],
    }
    config = {"configurable": {"thread_id": "t1", "user_id": "u1", "account_id": "a1"}}

    with patch("projects.agent.subgraphs.health_monitor.nodes.get_stream_writer", return_value=_make_writer_mock()), \
         patch("projects.agent.subgraphs.health_monitor.nodes.get_model", return_value=mock_model):
        from projects.agent.subgraphs.health_monitor.nodes import diagnose_node
        result = await diagnose_node(state, config)

    assert "agent_reports" in result
    report = result["agent_reports"][0]
    assert report["agent_id"] == "health_monitor"
    assert report["status"] == "completed"
    assert report["confidence"] == 0.85
    assert result["diagnosis"] == mock_response.content


@pytest.mark.asyncio
async def test_build_diagnosis_prompt():
    """_build_diagnosis_prompt inclui anomalias e classificacoes no prompt."""
    from projects.agent.subgraphs.health_monitor.nodes import _build_diagnosis_prompt

    anomalies = {"detected_count": 2, "anomalies": [{"entity_id": "c1"}]}
    classifications = {"c1": "UNDERPERFORMER"}

    prompt = _build_diagnosis_prompt(anomalies, classifications)

    assert "anomalias" in prompt.lower() or "anomalias detectadas" in prompt.lower() or "detected_count" in prompt
    assert "UNDERPERFORMER" in prompt
    assert "portugues" in prompt.lower()


@pytest.mark.asyncio
async def test_diagnose_node_no_anomalies():
    """diagnose_node sem anomalias gera diagnostico positivo."""
    mock_response = AIMessage(content="Tudo normal, sem anomalias.")
    mock_model = AsyncMock()
    mock_model.ainvoke.return_value = mock_response

    state = {
        "messages": [HumanMessage(content="Saude")],
        "scope": {},
        "metrics_ref": None,
        "anomaly_results": None,
        "classifications": None,
        "diagnosis": None,
        "agent_reports": [],
    }
    config = {"configurable": {"thread_id": "t1", "user_id": "u1", "account_id": "a1"}}

    with patch("projects.agent.subgraphs.health_monitor.nodes.get_stream_writer", return_value=_make_writer_mock()), \
         patch("projects.agent.subgraphs.health_monitor.nodes.get_model", return_value=mock_model):
        from projects.agent.subgraphs.health_monitor.nodes import diagnose_node
        result = await diagnose_node(state, config)

    assert result["agent_reports"][0]["data"]["anomaly_count"] == 0
