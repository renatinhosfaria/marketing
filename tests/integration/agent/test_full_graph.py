"""
Testes de integracao do grafo completo com LLM mockado.

Testa:
  - Fan-in: 3 agentes -> synthesizer recebe 3 reports
  - Subgraph failure -> error report no synthesizer
  - Builder cria grafo com todos os nos
  - Safe agent wrapper produz error report em excecao
  - Compile graph com checkpointer e store
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from langchain_core.messages import HumanMessage


@pytest.mark.asyncio
async def test_safe_agent_wrapper_success():
    """_safe_agent_wrapper retorna resultado do subgraph em caso de sucesso."""
    from projects.agent.graph.builder import _safe_agent_wrapper

    mock_subgraph = AsyncMock()
    mock_subgraph.ainvoke.return_value = {
        "agent_reports": [{
            "agent_id": "health_monitor",
            "status": "completed",
            "summary": "OK",
            "data": {},
            "confidence": 0.85,
        }],
    }

    wrapped = _safe_agent_wrapper("health_monitor", mock_subgraph)
    state = {"messages": [HumanMessage(content="Teste")], "scope": {}}
    config = {"configurable": {"thread_id": "t1"}}

    result = await wrapped(state, config)

    assert result["agent_reports"][0]["status"] == "completed"
    mock_subgraph.ainvoke.assert_awaited_once()


@pytest.mark.asyncio
async def test_safe_agent_wrapper_exception_produces_error_report():
    """_safe_agent_wrapper produz error report quando subgraph falha."""
    from projects.agent.graph.builder import _safe_agent_wrapper

    mock_subgraph = AsyncMock()
    mock_subgraph.ainvoke.side_effect = RuntimeError("Timeout na ML API")

    wrapped = _safe_agent_wrapper("performance_analyst", mock_subgraph)
    state = {"messages": [HumanMessage(content="Teste")], "scope": {}}
    config = {"configurable": {"thread_id": "t1"}}

    with patch("projects.agent.graph.builder.asyncio.sleep", new_callable=AsyncMock):
        result = await wrapped(state, config)

    report = result["agent_reports"][0]
    assert report["agent_id"] == "performance_analyst"
    assert report["status"] == "error"
    assert "RuntimeError" in report["summary"]
    assert report["confidence"] == 0.0
    assert report["data"]["error_code"] == "SUBGRAPH_EXCEPTION"


@pytest.mark.asyncio
async def test_safe_agent_wrapper_retries_before_failing():
    """_safe_agent_wrapper tenta 3 vezes antes de retornar error report."""
    from projects.agent.graph.builder import _safe_agent_wrapper

    mock_subgraph = AsyncMock()
    mock_subgraph.ainvoke.side_effect = ConnectionError("Connection refused")

    wrapped = _safe_agent_wrapper("forecast_scientist", mock_subgraph)
    state = {"messages": [HumanMessage(content="Teste")], "scope": {}}
    config = {"configurable": {"thread_id": "t1"}}

    with patch("projects.agent.graph.builder.asyncio.sleep", new_callable=AsyncMock):
        result = await wrapped(state, config)

    # 3 tentativas (HTTP_MAX_ATTEMPTS = 3)
    assert mock_subgraph.ainvoke.call_count == 3
    assert result["agent_reports"][0]["status"] == "error"


@pytest.mark.asyncio
async def test_safe_agent_wrapper_succeeds_on_retry():
    """_safe_agent_wrapper retorna sucesso se retry funciona."""
    from projects.agent.graph.builder import _safe_agent_wrapper

    success_result = {
        "agent_reports": [{
            "agent_id": "audience_specialist",
            "status": "completed",
            "summary": "Analise concluida",
            "data": {},
            "confidence": 0.80,
        }],
    }

    mock_subgraph = AsyncMock()
    mock_subgraph.ainvoke.side_effect = [
        ConnectionError("Retry 1"),
        success_result,
    ]

    wrapped = _safe_agent_wrapper("audience_specialist", mock_subgraph)
    state = {"messages": [HumanMessage(content="Teste")], "scope": {}}
    config = {"configurable": {"thread_id": "t1"}}

    with patch("projects.agent.graph.builder.asyncio.sleep", new_callable=AsyncMock):
        result = await wrapped(state, config)

    assert result["agent_reports"][0]["status"] == "completed"
    assert mock_subgraph.ainvoke.call_count == 2


@pytest.mark.asyncio
async def test_safe_agent_wrapper_callable_node():
    """_safe_agent_wrapper funciona com no simples (callable, sem ainvoke)."""
    from projects.agent.graph.builder import _safe_agent_wrapper

    async def simple_node(state, config):
        return {
            "agent_reports": [{
                "agent_id": "forecast_scientist",
                "status": "completed",
                "summary": "Previsao concluida",
                "data": {},
                "confidence": 0.75,
            }],
        }

    wrapped = _safe_agent_wrapper("forecast_scientist", simple_node)
    state = {"messages": [HumanMessage(content="Previsao")], "scope": {}}
    config = {"configurable": {"thread_id": "t1"}}

    result = await wrapped(state, config)

    assert result["agent_reports"][0]["status"] == "completed"


@pytest.mark.asyncio
async def test_build_supervisor_graph_has_all_nodes():
    """build_supervisor_graph cria grafo com todos os nos esperados."""
    with patch("projects.agent.graph.builder.build_health_graph") as mock_health, \
         patch("projects.agent.graph.builder.build_performance_graph") as mock_perf, \
         patch("projects.agent.graph.builder.build_creative_graph") as mock_creative, \
         patch("projects.agent.graph.builder.build_operations_graph") as mock_ops:
        mock_health.return_value = MagicMock()
        mock_perf.return_value = MagicMock()
        mock_creative.return_value = MagicMock()
        mock_ops.return_value = MagicMock()

        from projects.agent.graph.builder import build_supervisor_graph
        builder = build_supervisor_graph()

    # Verificar que todos os nos existem no builder
    node_names = set(builder.nodes.keys())
    expected = {
        "supervisor",
        "health_monitor",
        "performance_analyst",
        "creative_specialist",
        "audience_specialist",
        "forecast_scientist",
        "operations_manager",
        "synthesizer",
    }
    assert expected.issubset(node_names)
