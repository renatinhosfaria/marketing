"""
Testes dos nos individuais do subgraph Performance Analyst.

Testa:
  - analyze_metrics_node: busca metricas agregadas
  - compare_periods_node: compara periodos
  - generate_report_node: gera relatorio com LLM e produz AgentReport
  - _build_report_prompt: formato do prompt
  - Tratamento quando nao ha entity_ids no scope
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from langchain_core.messages import HumanMessage, AIMessage


def _make_writer_mock():
    writer = MagicMock()
    return writer


@pytest.mark.asyncio
async def test_analyze_metrics_node_success():
    """analyze_metrics_node retorna metricas agregadas."""
    mock_summary = {
        "total_spend": 5000.0,
        "total_leads": 200,
        "avg_cpl": 25.0,
        "avg_ctr": 2.1,
        "total_impressions": 100000,
    }

    mock_get_insights = AsyncMock()
    mock_get_insights.ainvoke.return_value = {
        "ok": True,
        "data": mock_summary,
        "error": None,
    }

    state = {
        "messages": [HumanMessage(content="Performance")],
        "scope": {},
        "metrics_data": None,
        "comparison": None,
        "impact_analysis": None,
        "report": None,
        "agent_reports": [],
    }
    config = {"configurable": {"thread_id": "t1", "user_id": "u1", "account_id": "a1"}}

    with patch("projects.agent.subgraphs.performance_analyst.nodes.get_stream_writer", return_value=_make_writer_mock()), \
         patch("projects.agent.subgraphs.performance_analyst.nodes.get_insights_summary", mock_get_insights):
        from projects.agent.subgraphs.performance_analyst.nodes import analyze_metrics_node
        result = await analyze_metrics_node(state, config)

    assert result["metrics_data"]["total_spend"] == 5000.0
    assert result["metrics_data"]["avg_cpl"] == 25.0


@pytest.mark.asyncio
async def test_analyze_metrics_node_failure():
    """analyze_metrics_node retorna None quando tool falha."""
    mock_get_insights = AsyncMock()
    mock_get_insights.ainvoke.return_value = {
        "ok": False,
        "data": None,
        "error": {"code": "DB_ERROR", "message": "Erro no banco"},
    }

    state = {
        "messages": [HumanMessage(content="Performance")],
        "scope": {},
        "metrics_data": None,
        "comparison": None,
        "impact_analysis": None,
        "report": None,
        "agent_reports": [],
    }
    config = {"configurable": {"thread_id": "t1", "user_id": "u1", "account_id": "a1"}}

    with patch("projects.agent.subgraphs.performance_analyst.nodes.get_stream_writer", return_value=_make_writer_mock()), \
         patch("projects.agent.subgraphs.performance_analyst.nodes.get_insights_summary", mock_get_insights):
        from projects.agent.subgraphs.performance_analyst.nodes import analyze_metrics_node
        result = await analyze_metrics_node(state, config)

    assert result["metrics_data"] is None


@pytest.mark.asyncio
async def test_compare_periods_node_with_entity_ids():
    """compare_periods_node faz comparacao quando ha entity_ids."""
    mock_comparison = {
        "period_a": {"spend": 2500, "leads": 100},
        "period_b": {"spend": 3000, "leads": 120},
        "diffs": {"spend": {"absolute": 500, "pct": 20.0}},
    }

    mock_compare = AsyncMock()
    mock_compare.ainvoke.return_value = {
        "ok": True,
        "data": mock_comparison,
        "error": None,
    }

    state = {
        "messages": [HumanMessage(content="Compare periodos")],
        "scope": {"entity_ids": ["c1"]},
        "metrics_data": {},
        "comparison": None,
        "impact_analysis": None,
        "report": None,
        "agent_reports": [],
    }
    config = {"configurable": {"thread_id": "t1", "user_id": "u1", "account_id": "a1"}}

    with patch("projects.agent.subgraphs.performance_analyst.nodes.get_stream_writer", return_value=_make_writer_mock()), \
         patch("projects.agent.subgraphs.performance_analyst.nodes.compare_periods", mock_compare):
        from projects.agent.subgraphs.performance_analyst.nodes import compare_periods_node
        result = await compare_periods_node(state, config)

    assert result["comparison"] is not None
    mock_compare.ainvoke.assert_awaited_once()


@pytest.mark.asyncio
async def test_compare_periods_node_no_entity_ids():
    """compare_periods_node pula comparacao sem entity_ids."""
    state = {
        "messages": [HumanMessage(content="Performance geral")],
        "scope": {"entity_ids": []},
        "metrics_data": {},
        "comparison": None,
        "impact_analysis": None,
        "report": None,
        "agent_reports": [],
    }
    config = {"configurable": {"thread_id": "t1", "user_id": "u1", "account_id": "a1"}}

    with patch("projects.agent.subgraphs.performance_analyst.nodes.get_stream_writer", return_value=_make_writer_mock()):
        from projects.agent.subgraphs.performance_analyst.nodes import compare_periods_node
        result = await compare_periods_node(state, config)

    assert result["comparison"] is None


@pytest.mark.asyncio
async def test_generate_report_node_produces_agent_report():
    """generate_report_node produz AgentReport com metricas e relatorio."""
    mock_response = AIMessage(content="Relatorio: CPL de R$25, 200 leads, tendencia estavel.")
    mock_model = AsyncMock()
    mock_model.ainvoke.return_value = mock_response

    state = {
        "messages": [HumanMessage(content="Como estao minhas campanhas?")],
        "scope": {},
        "metrics_data": {"total_spend": 5000, "avg_cpl": 25},
        "comparison": {"diffs": {"cpl": {"pct": -5.0}}},
        "impact_analysis": None,
        "report": None,
        "agent_reports": [],
    }
    config = {"configurable": {"thread_id": "t1", "user_id": "u1", "account_id": "a1"}}

    with patch("projects.agent.subgraphs.performance_analyst.nodes.get_stream_writer", return_value=_make_writer_mock()), \
         patch("projects.agent.subgraphs.performance_analyst.nodes.get_model", return_value=mock_model):
        from projects.agent.subgraphs.performance_analyst.nodes import generate_report_node
        result = await generate_report_node(state, config)

    report = result["agent_reports"][0]
    assert report["agent_id"] == "performance_analyst"
    assert report["status"] == "completed"
    assert report["confidence"] == 0.85
    assert result["report"] == mock_response.content
    assert report["data"]["metrics_summary"]["avg_cpl"] == 25
