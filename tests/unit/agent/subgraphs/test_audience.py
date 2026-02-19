"""
Testes do no simples Audience Specialist.

Testa:
  - audience_node: fluxo completo (busca, saturacao, performance, relatorio)
  - Tratamento de audiencias vazias
  - Saturacao detectada
  - AgentReport com dados corretos
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from langchain_core.messages import HumanMessage, AIMessage


def _make_writer_mock():
    writer = MagicMock()
    return writer


@pytest.mark.asyncio
async def test_audience_node_full_flow():
    """audience_node executa fluxo completo com audiencias."""
    mock_audiences = [
        {"adset_id": "as1", "name": "Adset 1", "targeting": {}},
        {"adset_id": "as2", "name": "Adset 2", "targeting": {}},
    ]
    mock_saturation = {
        "saturated": [{"adset_id": "as1", "saturation_level": "high"}],
        "healthy": [{"adset_id": "as2", "saturation_level": "low"}],
    }
    mock_perf = [
        {"adset_id": "as1", "avg_cpl": 30, "total_leads": 50},
        {"adset_id": "as2", "avg_cpl": 20, "total_leads": 80},
    ]

    mock_get_audiences = AsyncMock()
    mock_get_audiences.ainvoke.return_value = {"ok": True, "data": mock_audiences, "error": None}

    mock_detect_saturation = AsyncMock()
    mock_detect_saturation.ainvoke.return_value = {"ok": True, "data": mock_saturation, "error": None}

    mock_get_performance = AsyncMock()
    mock_get_performance.ainvoke.return_value = {"ok": True, "data": mock_perf, "error": None}

    mock_response = AIMessage(content="Adset as1 esta saturado, considere expandir audiencia.")
    mock_model = AsyncMock()
    mock_model.ainvoke.return_value = mock_response

    state = {
        "messages": [HumanMessage(content="Como estao minhas audiencias?")],
        "scope": {"entity_type": "campaign", "entity_ids": ["c1"]},
    }
    config = {"configurable": {"thread_id": "t1", "user_id": "u1", "account_id": "a1"}}

    with patch("projects.agent.subgraphs.audience_specialist.node.get_stream_writer", return_value=_make_writer_mock()), \
         patch("projects.agent.subgraphs.audience_specialist.node.get_adset_audiences", mock_get_audiences), \
         patch("projects.agent.subgraphs.audience_specialist.node.detect_audience_saturation", mock_detect_saturation), \
         patch("projects.agent.subgraphs.audience_specialist.node.get_audience_performance", mock_get_performance), \
         patch("projects.agent.subgraphs.audience_specialist.node.get_model", return_value=mock_model):
        from projects.agent.subgraphs.audience_specialist.node import audience_node
        result = await audience_node(state, config)

    report = result["agent_reports"][0]
    assert report["agent_id"] == "audience_specialist"
    assert report["status"] == "completed"
    assert report["data"]["total_audiences"] == 2
    assert report["data"]["saturated_count"] == 1


@pytest.mark.asyncio
async def test_audience_node_no_audiences():
    """audience_node com audiencias vazias gera relatorio informativo."""
    mock_get_audiences = AsyncMock()
    mock_get_audiences.ainvoke.return_value = {"ok": True, "data": [], "error": None}

    mock_response = AIMessage(content="Nenhuma audiencia encontrada para esta campanha.")
    mock_model = AsyncMock()
    mock_model.ainvoke.return_value = mock_response

    state = {
        "messages": [HumanMessage(content="Audiencias")],
        "scope": {"entity_type": "campaign", "entity_ids": ["c1"]},
    }
    config = {"configurable": {"thread_id": "t1", "user_id": "u1", "account_id": "a1"}}

    with patch("projects.agent.subgraphs.audience_specialist.node.get_stream_writer", return_value=_make_writer_mock()), \
         patch("projects.agent.subgraphs.audience_specialist.node.get_adset_audiences", mock_get_audiences), \
         patch("projects.agent.subgraphs.audience_specialist.node.get_model", return_value=mock_model):
        from projects.agent.subgraphs.audience_specialist.node import audience_node
        result = await audience_node(state, config)

    report = result["agent_reports"][0]
    assert report["data"]["total_audiences"] == 0
    assert report["data"]["saturated_count"] == 0


@pytest.mark.asyncio
async def test_audience_node_saturation_detection_failure():
    """audience_node continua mesmo se deteccao de saturacao falha."""
    mock_audiences = [{"adset_id": "as1", "name": "Adset 1", "targeting": {}}]

    mock_get_audiences = AsyncMock()
    mock_get_audiences.ainvoke.return_value = {"ok": True, "data": mock_audiences, "error": None}

    mock_detect_saturation = AsyncMock()
    mock_detect_saturation.ainvoke.return_value = {"ok": False, "data": None, "error": {"code": "DB_ERROR"}}

    mock_get_performance = AsyncMock()
    mock_get_performance.ainvoke.return_value = {"ok": False, "data": None, "error": {"code": "DB_ERROR"}}

    mock_response = AIMessage(content="Analise parcial disponivel.")
    mock_model = AsyncMock()
    mock_model.ainvoke.return_value = mock_response

    state = {
        "messages": [HumanMessage(content="Audiencias")],
        "scope": {"entity_type": "campaign", "entity_ids": ["c1"]},
    }
    config = {"configurable": {"thread_id": "t1", "user_id": "u1", "account_id": "a1"}}

    with patch("projects.agent.subgraphs.audience_specialist.node.get_stream_writer", return_value=_make_writer_mock()), \
         patch("projects.agent.subgraphs.audience_specialist.node.get_adset_audiences", mock_get_audiences), \
         patch("projects.agent.subgraphs.audience_specialist.node.detect_audience_saturation", mock_detect_saturation), \
         patch("projects.agent.subgraphs.audience_specialist.node.get_audience_performance", mock_get_performance), \
         patch("projects.agent.subgraphs.audience_specialist.node.get_model", return_value=mock_model):
        from projects.agent.subgraphs.audience_specialist.node import audience_node
        result = await audience_node(state, config)

    report = result["agent_reports"][0]
    assert report["status"] == "completed"
    assert report["data"]["saturated_count"] == 0


@pytest.mark.asyncio
async def test_audience_node_without_campaign_id():
    """audience_node funciona sem campaign_id no scope."""
    mock_get_audiences = AsyncMock()
    mock_get_audiences.ainvoke.return_value = {"ok": True, "data": [], "error": None}

    mock_response = AIMessage(content="Sem dados disponiveis.")
    mock_model = AsyncMock()
    mock_model.ainvoke.return_value = mock_response

    state = {
        "messages": [HumanMessage(content="Audiencias")],
        "scope": {"entity_type": "adset"},
    }
    config = {"configurable": {"thread_id": "t1", "user_id": "u1", "account_id": "a1"}}

    with patch("projects.agent.subgraphs.audience_specialist.node.get_stream_writer", return_value=_make_writer_mock()), \
         patch("projects.agent.subgraphs.audience_specialist.node.get_adset_audiences", mock_get_audiences), \
         patch("projects.agent.subgraphs.audience_specialist.node.get_model", return_value=mock_model):
        from projects.agent.subgraphs.audience_specialist.node import audience_node
        await audience_node(state, config)

    # campaign_id deve ser None quando entity_type nao e campaign
    call_args = mock_get_audiences.ainvoke.call_args
    assert call_args[0][0]["campaign_id"] is None
