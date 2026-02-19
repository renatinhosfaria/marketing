"""
Testes das tools do Especialista em Criativos.

Testa:
  - get_ad_creatives: mock DB query
  - detect_creative_fatigue: analise de tendencia
  - compare_creatives: ranking por metrica
  - get_ad_preview_url: URL de preview
  - ToolResult contract (ok/data/error)
"""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock


@pytest.mark.asyncio
async def test_get_ad_creatives_success():
    """get_ad_creatives retorna lista de anuncios."""
    mock_row = MagicMock()
    mock_row._mapping = {
        "ad_id": "ad1",
        "name": "Ad Teste",
        "status": "ACTIVE",
        "creative_id": "cr1",
        "adset_id": "as1",
        "campaign_id": "c1",
    }
    mock_result = MagicMock()
    mock_result.all.return_value = [mock_row]

    mock_session = AsyncMock()
    mock_session.execute.return_value = mock_result
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    with patch("projects.agent.tools.creative_tools.resolve_config_id", new_callable=AsyncMock, return_value=1), \
         patch("projects.agent.tools.creative_tools.async_session_maker", return_value=mock_session):
        from projects.agent.tools.creative_tools import get_ad_creatives
        result = await get_ad_creatives.ainvoke(
            {"campaign_id": "c1"},
            config={"configurable": {"account_id": "a1"}},
        )

    assert result["ok"] is True
    assert len(result["data"]) == 1
    assert result["data"][0]["ad_id"] == "ad1"


@pytest.mark.asyncio
async def test_get_ad_creatives_db_error():
    """get_ad_creatives retorna DB_ERROR quando query falha."""
    mock_session = AsyncMock()
    mock_session.execute.side_effect = Exception("Connection refused")
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    with patch("projects.agent.tools.creative_tools.resolve_config_id", new_callable=AsyncMock, return_value=1), \
         patch("projects.agent.tools.creative_tools.async_session_maker", return_value=mock_session):
        from projects.agent.tools.creative_tools import get_ad_creatives
        result = await get_ad_creatives.ainvoke(
            {},
            config={"configurable": {"account_id": "a1"}},
        )

    assert result["ok"] is False
    assert result["error"]["code"] == "DB_ERROR"


@pytest.mark.asyncio
async def test_detect_creative_fatigue_high():
    """detect_creative_fatigue detecta fadiga alta (CTR caindo + frequency subindo)."""
    # Simular 6 linhas: metade boa, metade ruim
    rows = []
    for i in range(6):
        row = MagicMock()
        if i < 3:  # Primeira metade: CTR alto, frequency baixo
            row.date = f"2026-02-0{i+1}"
            row.ctr = 3.0
            row.frequency = 1.0
            row.impressions = 5000
        else:  # Segunda metade: CTR baixo, frequency alto
            row.date = f"2026-02-0{i+1}"
            row.ctr = 2.0  # < 3.0 * 0.85 = 2.55
            row.frequency = 1.5  # > 1.0 * 1.2 = 1.2
            row.impressions = 5000
        rows.append(row)

    mock_result = MagicMock()
    mock_result.all.return_value = rows

    mock_session = AsyncMock()
    mock_session.execute.return_value = mock_result
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    with patch("projects.agent.tools.creative_tools.resolve_config_id", new_callable=AsyncMock, return_value=1), \
         patch("projects.agent.tools.creative_tools._validate_entity_ownership", new_callable=AsyncMock, return_value=True), \
         patch("projects.agent.tools.creative_tools.async_session_maker", return_value=mock_session):
        from projects.agent.tools.creative_tools import detect_creative_fatigue
        result = await detect_creative_fatigue.ainvoke(
            {"ad_ids": ["ad1"]},
            config={"configurable": {"account_id": "a1"}},
        )

    assert result["ok"] is True
    assert result["data"]["total_analyzed"] == 1
    assert len(result["data"]["fatigued_ads"]) == 1
    assert result["data"]["fatigued_ads"][0]["fatigue_score"] == "high"


@pytest.mark.asyncio
async def test_detect_creative_fatigue_healthy():
    """detect_creative_fatigue classifica como healthy (CTR estavel, frequency estavel)."""
    rows = []
    for i in range(6):
        row = MagicMock()
        row.date = f"2026-02-0{i+1}"
        row.ctr = 2.5
        row.frequency = 1.1
        row.impressions = 5000
        rows.append(row)

    mock_result = MagicMock()
    mock_result.all.return_value = rows

    mock_session = AsyncMock()
    mock_session.execute.return_value = mock_result
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    with patch("projects.agent.tools.creative_tools.resolve_config_id", new_callable=AsyncMock, return_value=1), \
         patch("projects.agent.tools.creative_tools._validate_entity_ownership", new_callable=AsyncMock, return_value=True), \
         patch("projects.agent.tools.creative_tools.async_session_maker", return_value=mock_session):
        from projects.agent.tools.creative_tools import detect_creative_fatigue
        result = await detect_creative_fatigue.ainvoke(
            {"ad_ids": ["ad1"]},
            config={"configurable": {"account_id": "a1"}},
        )

    assert result["ok"] is True
    assert len(result["data"]["healthy_ads"]) == 1
    assert result["data"]["healthy_ads"][0]["fatigue_score"] == "none"


@pytest.mark.asyncio
async def test_get_ad_preview_url_success():
    """get_ad_preview_url retorna URL de preview."""
    with patch("projects.agent.tools.creative_tools.resolve_config_id", new_callable=AsyncMock, return_value=1), \
         patch("projects.agent.tools.creative_tools._validate_entity_ownership", new_callable=AsyncMock, return_value=True):
        from projects.agent.tools.creative_tools import get_ad_preview_url
        result = await get_ad_preview_url.ainvoke(
            {"ad_id": "ad123"},
            config={"configurable": {"account_id": "a1"}},
        )

    assert result["ok"] is True
    assert "ad123" in result["data"]["preview_url"]


@pytest.mark.asyncio
async def test_get_ad_preview_url_ownership_error():
    """get_ad_preview_url retorna erro para anuncio de outro tenant."""
    with patch("projects.agent.tools.creative_tools.resolve_config_id", new_callable=AsyncMock, return_value=1), \
         patch("projects.agent.tools.creative_tools._validate_entity_ownership", new_callable=AsyncMock, return_value=False):
        from projects.agent.tools.creative_tools import get_ad_preview_url
        result = await get_ad_preview_url.ainvoke(
            {"ad_id": "ad_outro"},
            config={"configurable": {"account_id": "a1"}},
        )

    assert result["ok"] is False
    assert result["error"]["code"] == "OWNERSHIP_ERROR"
