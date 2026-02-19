"""
Testes dos nos individuais do subgraph Creative Specialist.

Testa:
  - fetch_ads_node: busca criativos via tool
  - analyze_fatigue_node: detecta fadiga criativa
  - recommend_node: gera recomendacoes e AgentReport
  - Tratamento de lista vazia de ads
  - Propagacao de scope (campaign_id)
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from langchain_core.messages import HumanMessage, AIMessage


def _make_writer_mock():
    writer = MagicMock()
    return writer


@pytest.mark.asyncio
async def test_fetch_ads_node_success():
    """fetch_ads_node retorna criativos quando tool retorna ok=True."""
    mock_ads = [
        {"ad_id": "ad1", "name": "Ad 1", "status": "ACTIVE", "creative_id": "cr1"},
        {"ad_id": "ad2", "name": "Ad 2", "status": "ACTIVE", "creative_id": "cr2"},
    ]

    mock_get_creatives = AsyncMock()
    mock_get_creatives.ainvoke.return_value = {
        "ok": True,
        "data": mock_ads,
        "error": None,
    }

    state = {
        "messages": [HumanMessage(content="Analise meus criativos")],
        "scope": {"entity_type": "campaign", "entity_ids": ["c1"]},
        "ad_creatives": [],
        "fatigue_analysis": None,
        "preview_urls": [],
        "recommendation": None,
        "agent_reports": [],
    }
    config = {"configurable": {"thread_id": "t1", "user_id": "u1", "account_id": "a1"}}

    with patch("projects.agent.subgraphs.creative_specialist.nodes.get_stream_writer", return_value=_make_writer_mock()), \
         patch("projects.agent.subgraphs.creative_specialist.nodes.get_ad_creatives", mock_get_creatives):
        from projects.agent.subgraphs.creative_specialist.nodes import fetch_ads_node
        result = await fetch_ads_node(state, config)

    assert len(result["ad_creatives"]) == 2
    assert result["ad_creatives"][0]["ad_id"] == "ad1"


@pytest.mark.asyncio
async def test_fetch_ads_node_empty():
    """fetch_ads_node retorna lista vazia quando nao ha criativos."""
    mock_get_creatives = AsyncMock()
    mock_get_creatives.ainvoke.return_value = {
        "ok": True,
        "data": [],
        "error": None,
    }

    state = {
        "messages": [HumanMessage(content="Criativos")],
        "scope": {},
        "ad_creatives": [],
        "fatigue_analysis": None,
        "preview_urls": [],
        "recommendation": None,
        "agent_reports": [],
    }
    config = {"configurable": {"thread_id": "t1", "user_id": "u1", "account_id": "a1"}}

    with patch("projects.agent.subgraphs.creative_specialist.nodes.get_stream_writer", return_value=_make_writer_mock()), \
         patch("projects.agent.subgraphs.creative_specialist.nodes.get_ad_creatives", mock_get_creatives):
        from projects.agent.subgraphs.creative_specialist.nodes import fetch_ads_node
        result = await fetch_ads_node(state, config)

    assert result["ad_creatives"] == []


@pytest.mark.asyncio
async def test_analyze_fatigue_node_with_ads():
    """analyze_fatigue_node detecta fadiga quando ha criativos."""
    mock_fatigue = {
        "fatigued_ads": [{"ad_id": "ad1", "fatigue_score": "high"}],
        "healthy_ads": [{"ad_id": "ad2", "fatigue_score": "none"}],
        "total_analyzed": 2,
    }

    mock_detect_fatigue = AsyncMock()
    mock_detect_fatigue.ainvoke.return_value = {
        "ok": True,
        "data": mock_fatigue,
        "error": None,
    }

    state = {
        "messages": [HumanMessage(content="Fadiga")],
        "scope": {"lookback_days": 14},
        "ad_creatives": [
            {"ad_id": "ad1", "name": "Ad 1"},
            {"ad_id": "ad2", "name": "Ad 2"},
        ],
        "fatigue_analysis": None,
        "preview_urls": [],
        "recommendation": None,
        "agent_reports": [],
    }
    config = {"configurable": {"thread_id": "t1", "user_id": "u1", "account_id": "a1"}}

    with patch("projects.agent.subgraphs.creative_specialist.nodes.get_stream_writer", return_value=_make_writer_mock()), \
         patch("projects.agent.subgraphs.creative_specialist.nodes.detect_creative_fatigue", mock_detect_fatigue):
        from projects.agent.subgraphs.creative_specialist.nodes import analyze_fatigue_node
        result = await analyze_fatigue_node(state, config)

    assert result["fatigue_analysis"]["total_analyzed"] == 2
    assert len(result["fatigue_analysis"]["fatigued_ads"]) == 1


@pytest.mark.asyncio
async def test_analyze_fatigue_node_no_ads():
    """analyze_fatigue_node pula analise sem criativos."""
    state = {
        "messages": [HumanMessage(content="Fadiga")],
        "scope": {},
        "ad_creatives": [],
        "fatigue_analysis": None,
        "preview_urls": [],
        "recommendation": None,
        "agent_reports": [],
    }
    config = {"configurable": {"thread_id": "t1", "user_id": "u1", "account_id": "a1"}}

    with patch("projects.agent.subgraphs.creative_specialist.nodes.get_stream_writer", return_value=_make_writer_mock()):
        from projects.agent.subgraphs.creative_specialist.nodes import analyze_fatigue_node
        result = await analyze_fatigue_node(state, config)

    assert result["fatigue_analysis"] is None


@pytest.mark.asyncio
async def test_recommend_node_produces_agent_report():
    """recommend_node gera AgentReport com dados de fadiga."""
    mock_response = AIMessage(content="Recomendo pausar ad1 por fadiga criativa.")
    mock_model = AsyncMock()
    mock_model.ainvoke.return_value = mock_response

    state = {
        "messages": [HumanMessage(content="Recomendacoes de criativos")],
        "scope": {},
        "ad_creatives": [{"ad_id": "ad1"}, {"ad_id": "ad2"}, {"ad_id": "ad3"}],
        "fatigue_analysis": {
            "fatigued_ads": [{"ad_id": "ad1", "fatigue_score": "high"}],
            "healthy_ads": [{"ad_id": "ad2"}, {"ad_id": "ad3"}],
        },
        "preview_urls": [],
        "recommendation": None,
        "agent_reports": [],
    }
    config = {"configurable": {"thread_id": "t1", "user_id": "u1", "account_id": "a1"}}

    with patch("projects.agent.subgraphs.creative_specialist.nodes.get_stream_writer", return_value=_make_writer_mock()), \
         patch("projects.agent.subgraphs.creative_specialist.nodes.get_model", return_value=mock_model):
        from projects.agent.subgraphs.creative_specialist.nodes import recommend_node
        result = await recommend_node(state, config)

    report = result["agent_reports"][0]
    assert report["agent_id"] == "creative_specialist"
    assert report["status"] == "completed"
    assert report["confidence"] == 0.80
    assert report["data"]["total_ads"] == 3
    assert report["data"]["fatigued_count"] == 1
    assert result["recommendation"] == mock_response.content
