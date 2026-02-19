"""
Testes das tools do Especialista em Audiencias.

Testa:
  - get_adset_audiences: mock DB query
  - detect_audience_saturation: analise de tendencia
  - get_audience_performance: KPIs por adset
  - ToolResult contract (ok/data/error)
  - Ownership validation
"""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock


@pytest.mark.asyncio
async def test_get_adset_audiences_success():
    """get_adset_audiences retorna dados de segmentacao."""
    mock_row = MagicMock()
    mock_row._mapping = {
        "adset_id": "as1",
        "name": "Adset Teste",
        "status": "ACTIVE",
        "targeting": {"age_min": 25, "age_max": 45},
        "daily_budget": 50.0,
        "campaign_id": "c1",
    }
    mock_result = MagicMock()
    mock_result.all.return_value = [mock_row]

    mock_session = AsyncMock()
    mock_session.execute.return_value = mock_result
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    with patch("projects.agent.tools.audience_tools.resolve_config_id", new_callable=AsyncMock, return_value=1), \
         patch("projects.agent.tools.audience_tools.async_session_maker", return_value=mock_session):
        from projects.agent.tools.audience_tools import get_adset_audiences
        result = await get_adset_audiences.ainvoke(
            {},
            config={"configurable": {"account_id": "a1"}},
        )

    assert result["ok"] is True
    assert len(result["data"]) == 1
    assert result["data"][0]["adset_id"] == "as1"


@pytest.mark.asyncio
async def test_get_adset_audiences_with_campaign_filter():
    """get_adset_audiences filtra por campaign_id quando fornecido."""
    mock_row = MagicMock()
    mock_row._mapping = {
        "adset_id": "as1",
        "name": "Adset Filtrado",
        "status": "ACTIVE",
        "targeting": {},
        "daily_budget": 30.0,
        "campaign_id": "c1",
    }
    mock_result = MagicMock()
    mock_result.all.return_value = [mock_row]

    mock_session = AsyncMock()
    mock_session.execute.return_value = mock_result
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    with patch("projects.agent.tools.audience_tools.resolve_config_id", new_callable=AsyncMock, return_value=1), \
         patch("projects.agent.tools.audience_tools._validate_entity_ownership", new_callable=AsyncMock, return_value=True), \
         patch("projects.agent.tools.audience_tools.async_session_maker", return_value=mock_session):
        from projects.agent.tools.audience_tools import get_adset_audiences
        result = await get_adset_audiences.ainvoke(
            {"campaign_id": "c1"},
            config={"configurable": {"account_id": "a1"}},
        )

    assert result["ok"] is True


@pytest.mark.asyncio
async def test_get_adset_audiences_campaign_ownership_error():
    """get_adset_audiences retorna erro para campanha de outro tenant."""
    with patch("projects.agent.tools.audience_tools.resolve_config_id", new_callable=AsyncMock, return_value=1), \
         patch("projects.agent.tools.audience_tools._validate_entity_ownership", new_callable=AsyncMock, return_value=False), \
         patch("projects.agent.tools.audience_tools.async_session_maker", return_value=AsyncMock()):
        from projects.agent.tools.audience_tools import get_adset_audiences
        result = await get_adset_audiences.ainvoke(
            {"campaign_id": "c_outro"},
            config={"configurable": {"account_id": "a1"}},
        )

    assert result["ok"] is False
    assert result["error"]["code"] == "OWNERSHIP_ERROR"


@pytest.mark.asyncio
async def test_detect_audience_saturation_high():
    """detect_audience_saturation detecta saturacao alta."""
    rows = []
    for i in range(6):
        row = MagicMock()
        if i < 3:
            row.date = f"2026-02-0{i+1}"
            row.ctr = 3.0
            row.frequency = 1.0
            row.impressions = 5000
        else:
            row.date = f"2026-02-0{i+1}"
            row.ctr = 2.0  # CTR caindo
            row.frequency = 1.5  # Frequency subindo
            row.impressions = 5000
        rows.append(row)

    mock_result = MagicMock()
    mock_result.all.return_value = rows

    mock_session = AsyncMock()
    mock_session.execute.return_value = mock_result
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    with patch("projects.agent.tools.audience_tools.resolve_config_id", new_callable=AsyncMock, return_value=1), \
         patch("projects.agent.tools.audience_tools._validate_entity_ownership", new_callable=AsyncMock, return_value=True), \
         patch("projects.agent.tools.audience_tools.async_session_maker", return_value=mock_session):
        from projects.agent.tools.audience_tools import detect_audience_saturation
        result = await detect_audience_saturation.ainvoke(
            {"adset_ids": ["as1"]},
            config={"configurable": {"account_id": "a1"}},
        )

    assert result["ok"] is True
    assert result["data"]["total_analyzed"] == 1
    assert len(result["data"]["saturated"]) == 1


@pytest.mark.asyncio
async def test_get_audience_performance_success():
    """get_audience_performance retorna metricas por adset."""
    mock_row = MagicMock()
    mock_row.avg_cpl = 25.0
    mock_row.avg_ctr = 2.0
    mock_row.total_leads = 50
    mock_row.total_spend = 1250.0
    mock_row.total_impressions = 50000

    mock_result = MagicMock()
    mock_result.one.return_value = mock_row

    mock_session = AsyncMock()
    mock_session.execute.return_value = mock_result
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    with patch("projects.agent.tools.audience_tools.resolve_config_id", new_callable=AsyncMock, return_value=1), \
         patch("projects.agent.tools.audience_tools._validate_entity_ownership", new_callable=AsyncMock, return_value=True), \
         patch("projects.agent.tools.audience_tools.async_session_maker", return_value=mock_session):
        from projects.agent.tools.audience_tools import get_audience_performance
        result = await get_audience_performance.ainvoke(
            {"adset_ids": ["as1"]},
            config={"configurable": {"account_id": "a1"}},
        )

    assert result["ok"] is True
    assert len(result["data"]) == 1
    assert result["data"][0]["avg_cpl"] == 25.0
