"""
Testes do no simples Forecast Scientist.

Testa:
  - forecast_node: fluxo completo com previsoes
  - Multiplas entidades com sucesso e erro parcial
  - Sem entity_ids no scope
  - Multiplas metricas (cpl, leads)
  - AgentReport com dados corretos
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from langchain_core.messages import HumanMessage, AIMessage


def _make_writer_mock():
    writer = MagicMock()
    return writer


@pytest.mark.asyncio
async def test_forecast_node_with_entities():
    """forecast_node gera previsoes para entidades no scope."""
    mock_forecast_data = {"predicted_values": [25, 23, 22], "horizon": 7}

    mock_generate = AsyncMock()
    mock_generate.ainvoke.return_value = {
        "ok": True,
        "data": mock_forecast_data,
        "error": None,
    }

    mock_response = AIMessage(content="Previsao: CPL deve cair para R$22 nos proximos 7 dias.")
    mock_model = AsyncMock()
    mock_model.ainvoke.return_value = mock_response

    state = {
        "messages": [HumanMessage(content="Previsao de CPL para campanha c1")],
        "scope": {"entity_ids": ["c1"], "lookback_days": 7},
    }
    config = {"configurable": {"thread_id": "t1", "user_id": "u1", "account_id": "a1"}}

    with patch("projects.agent.subgraphs.forecast_scientist.node.get_stream_writer", return_value=_make_writer_mock()), \
         patch("projects.agent.subgraphs.forecast_scientist.node.generate_forecast", mock_generate), \
         patch("projects.agent.subgraphs.forecast_scientist.node.get_model", return_value=mock_model):
        from projects.agent.subgraphs.forecast_scientist.node import forecast_node
        result = await forecast_node(state, config)

    report = result["agent_reports"][0]
    assert report["agent_id"] == "forecast_scientist"
    assert report["status"] == "completed"
    assert report["confidence"] == 0.75
    # 2 metricas (cpl, leads) x 1 entidade = 2 chamadas
    assert mock_generate.ainvoke.call_count == 2
    assert report["data"]["forecasts_generated"] == 2


@pytest.mark.asyncio
async def test_forecast_node_no_entities():
    """forecast_node sem entities gera relatorio sugerindo especificar."""
    mock_response = AIMessage(content="Especifique uma campanha para previsao.")
    mock_model = AsyncMock()
    mock_model.ainvoke.return_value = mock_response

    state = {
        "messages": [HumanMessage(content="Previsao")],
        "scope": {"entity_ids": []},
    }
    config = {"configurable": {"thread_id": "t1", "user_id": "u1", "account_id": "a1"}}

    with patch("projects.agent.subgraphs.forecast_scientist.node.get_stream_writer", return_value=_make_writer_mock()), \
         patch("projects.agent.subgraphs.forecast_scientist.node.get_model", return_value=mock_model):
        from projects.agent.subgraphs.forecast_scientist.node import forecast_node
        result = await forecast_node(state, config)

    report = result["agent_reports"][0]
    assert report["data"]["forecasts_generated"] == 0
    assert report["data"]["forecast_errors"] == 0


@pytest.mark.asyncio
async def test_forecast_node_partial_errors():
    """forecast_node com erros parciais registra forecasts e errors."""
    call_count = 0

    async def mock_ainvoke(args, config=None):
        nonlocal call_count
        call_count += 1
        if args.get("metric") == "cpl":
            return {"ok": True, "data": {"predicted": [25]}, "error": None}
        else:
            return {"ok": False, "data": None, "error": {"code": "TIMEOUT", "message": "Timeout"}}

    mock_generate = AsyncMock()
    mock_generate.ainvoke.side_effect = mock_ainvoke

    mock_response = AIMessage(content="CPL previsto, mas leads falhou.")
    mock_model = AsyncMock()
    mock_model.ainvoke.return_value = mock_response

    state = {
        "messages": [HumanMessage(content="Previsao de c1")],
        "scope": {"entity_ids": ["c1"], "lookback_days": 7},
    }
    config = {"configurable": {"thread_id": "t1", "user_id": "u1", "account_id": "a1"}}

    with patch("projects.agent.subgraphs.forecast_scientist.node.get_stream_writer", return_value=_make_writer_mock()), \
         patch("projects.agent.subgraphs.forecast_scientist.node.generate_forecast", mock_generate), \
         patch("projects.agent.subgraphs.forecast_scientist.node.get_model", return_value=mock_model):
        from projects.agent.subgraphs.forecast_scientist.node import forecast_node
        result = await forecast_node(state, config)

    report = result["agent_reports"][0]
    assert report["data"]["forecasts_generated"] == 1
    assert report["data"]["forecast_errors"] == 1


@pytest.mark.asyncio
async def test_forecast_node_limits_entities_to_five():
    """forecast_node processa no maximo 5 entidades."""
    mock_generate = AsyncMock()
    mock_generate.ainvoke.return_value = {
        "ok": True,
        "data": {"predicted": [10]},
        "error": None,
    }

    mock_response = AIMessage(content="Previsoes geradas.")
    mock_model = AsyncMock()
    mock_model.ainvoke.return_value = mock_response

    state = {
        "messages": [HumanMessage(content="Previsao")],
        "scope": {"entity_ids": [f"c{i}" for i in range(10)], "lookback_days": 7},
    }
    config = {"configurable": {"thread_id": "t1", "user_id": "u1", "account_id": "a1"}}

    with patch("projects.agent.subgraphs.forecast_scientist.node.get_stream_writer", return_value=_make_writer_mock()), \
         patch("projects.agent.subgraphs.forecast_scientist.node.generate_forecast", mock_generate), \
         patch("projects.agent.subgraphs.forecast_scientist.node.get_model", return_value=mock_model):
        from projects.agent.subgraphs.forecast_scientist.node import forecast_node
        await forecast_node(state, config)

    # 5 entidades x 2 metricas = 10 chamadas max
    assert mock_generate.ainvoke.call_count == 10
