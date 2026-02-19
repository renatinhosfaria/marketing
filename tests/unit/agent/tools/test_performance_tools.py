"""
Testes das tools do Analista de Performance.

Testa:
  - get_campaign_insights: mock DB queries
  - compare_periods: comparacao entre periodos
  - get_insights_summary: KPIs agregados
  - analyze_causal_impact: chamada ML API
  - ToolResult contract (ok/data/error)
  - Ownership validation
"""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from projects.agent.tools.result import tool_success


def _mock_session_with_rows(rows):
    """Cria mock de async_session_maker que retorna rows."""
    mock_result = MagicMock()
    mock_result.all.return_value = rows
    mock_result.one.return_value = rows[0] if rows else MagicMock()

    mock_session = AsyncMock()
    mock_session.execute.return_value = mock_result
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    mock_maker = MagicMock(return_value=mock_session)
    return mock_maker


@pytest.mark.asyncio
async def test_get_campaign_insights_success():
    """get_campaign_insights retorna metricas do banco."""
    # Mock row com _mapping
    mock_row = MagicMock()
    mock_row._mapping = {
        "date": "2026-02-01",
        "spend": 100.0,
        "leads": 5,
        "cpl": 20.0,
        "ctr": 2.5,
        "cpc": 1.0,
        "impressions": 5000,
    }
    mock_result = MagicMock()
    mock_result.all.return_value = [mock_row]

    mock_session = AsyncMock()
    mock_session.execute.return_value = mock_result
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    with patch("projects.agent.tools.performance_tools.resolve_config_id", new_callable=AsyncMock, return_value=1), \
         patch("projects.agent.tools.performance_tools._validate_entity_ownership", new_callable=AsyncMock, return_value=True), \
         patch("projects.agent.tools.performance_tools.async_session_maker", return_value=mock_session):
        from projects.agent.tools.performance_tools import get_campaign_insights
        result = await get_campaign_insights.ainvoke(
            {
                "entity_type": "campaign",
                "entity_id": "c1",
                "date_start": "2026-02-01",
                "date_end": "2026-02-07",
            },
            config={"configurable": {"account_id": "a1"}},
        )

    assert result["ok"] is True
    assert len(result["data"]) == 1


@pytest.mark.asyncio
async def test_get_campaign_insights_ownership_error():
    """get_campaign_insights retorna erro quando entidade nao pertence a conta."""
    with patch("projects.agent.tools.performance_tools.resolve_config_id", new_callable=AsyncMock, return_value=1), \
         patch("projects.agent.tools.performance_tools._validate_entity_ownership", new_callable=AsyncMock, return_value=False):
        from projects.agent.tools.performance_tools import get_campaign_insights
        result = await get_campaign_insights.ainvoke(
            {
                "entity_type": "campaign",
                "entity_id": "c_outro_tenant",
                "date_start": "2026-02-01",
                "date_end": "2026-02-07",
            },
            config={"configurable": {"account_id": "a1"}},
        )

    assert result["ok"] is False
    assert result["error"]["code"] == "OWNERSHIP_ERROR"


@pytest.mark.asyncio
async def test_get_insights_summary_success():
    """get_insights_summary retorna KPIs agregados."""
    mock_row = MagicMock()
    mock_row.total_spend = 5000.0
    mock_row.total_leads = 200
    mock_row.avg_cpl = 25.0
    mock_row.avg_ctr = 2.1
    mock_row.total_impressions = 100000

    mock_result = MagicMock()
    mock_result.one.return_value = mock_row

    mock_session = AsyncMock()
    mock_session.execute.return_value = mock_result
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    with patch("projects.agent.tools.performance_tools.resolve_config_id", new_callable=AsyncMock, return_value=1), \
         patch("projects.agent.tools.performance_tools.async_session_maker", return_value=mock_session):
        from projects.agent.tools.performance_tools import get_insights_summary
        result = await get_insights_summary.ainvoke(
            {},
            config={"configurable": {"account_id": "a1"}},
        )

    assert result["ok"] is True
    assert result["data"]["total_spend"] == 5000.0
    assert result["data"]["total_leads"] == 200
    assert result["data"]["avg_cpl"] == 25.0


@pytest.mark.asyncio
async def test_compare_periods_success():
    """compare_periods retorna diferencas entre periodos."""
    mock_row = MagicMock()
    mock_row.spend = 2500.0
    mock_row.leads = 100
    mock_row.cpl = 25.0
    mock_row.ctr = 2.0

    mock_result = MagicMock()
    mock_result.one.return_value = mock_row

    mock_session = AsyncMock()
    mock_session.execute.return_value = mock_result
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    with patch("projects.agent.tools.performance_tools.resolve_config_id", new_callable=AsyncMock, return_value=1), \
         patch("projects.agent.tools.performance_tools._validate_entity_ownership", new_callable=AsyncMock, return_value=True), \
         patch("projects.agent.tools.performance_tools.async_session_maker", return_value=mock_session):
        from projects.agent.tools.performance_tools import compare_periods
        result = await compare_periods.ainvoke(
            {
                "entity_id": "c1",
                "period_a_start": "2026-01-25",
                "period_a_end": "2026-02-01",
                "period_b_start": "2026-02-01",
                "period_b_end": "2026-02-08",
            },
            config={"configurable": {"account_id": "a1"}},
        )

    assert result["ok"] is True
    assert "period_a" in result["data"]
    assert "period_b" in result["data"]
    assert "diffs" in result["data"]


@pytest.mark.asyncio
async def test_analyze_causal_impact_success():
    """analyze_causal_impact chama ML API corretamente."""
    mock_impact = {"impact_pct": -15.0, "significant": True}
    ml_call = AsyncMock(return_value=tool_success(mock_impact))

    with patch("projects.agent.tools.performance_tools.resolve_config_id", new_callable=AsyncMock, return_value=1), \
         patch("projects.agent.tools.performance_tools._ml_api_call", ml_call):
        from projects.agent.tools.performance_tools import analyze_causal_impact
        result = await analyze_causal_impact.ainvoke(
            {
                "entity_type": "campaign",
                "entity_id": "c1",
                "change_date": "2026-02-01T00:00:00Z",
                "change_type": "budget_change",
            },
            config={"configurable": {"account_id": "a1"}},
        )

    assert result["ok"] is True
    assert result["data"]["impact_pct"] == -15.0
    ml_call.assert_called_once_with(
        "post",
        "/api/v1/impact/analyze",
        json={
            "config_id": 1,
            "entity_type": "campaign",
            "entity_id": "c1",
            "change_date": "2026-02-01T00:00:00Z",
            "change_type": "budget_change",
            "window_before": 7,
            "window_after": 7,
        },
        account_id="a1",
        timeout=30,
    )
