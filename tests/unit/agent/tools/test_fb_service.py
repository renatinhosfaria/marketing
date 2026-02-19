"""
Testes do adapter fb_service.

Testa:
  - update_budget: conversao reais → centavos
  - update_budget: round() evita perda de precisao float
  - update_status: envia status correto
  - update_status: rejeita status invalido
  - update_budget/update_status: cleanup do client em caso de erro
  - _get_graph_client: campanha nao encontrada
"""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from projects.agent.tools.fb_service import (
    update_budget,
    update_status,
    _get_graph_client,
)

# Prefixo para patches no namespace do modulo
_FB = "projects.agent.tools.fb_service"


@pytest.mark.asyncio
async def test_update_budget_converts_to_centavos():
    """update_budget converte R$50.00 para 5000 centavos."""
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value={"success": True})
    mock_client.close = AsyncMock()

    with patch(f"{_FB}._get_graph_client", new_callable=AsyncMock, return_value=(mock_client, "act_123")):
        await update_budget("c1", 50.0)

    mock_client.post.assert_called_once_with(
        endpoint="c1",
        data={"daily_budget": 5000},
    )
    mock_client.close.assert_called_once()


@pytest.mark.asyncio
async def test_update_budget_fractional_centavos():
    """update_budget converte R$75.50 para 7550 centavos."""
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value={"success": True})
    mock_client.close = AsyncMock()

    with patch(f"{_FB}._get_graph_client", new_callable=AsyncMock, return_value=(mock_client, "act_123")):
        await update_budget("c1", 75.50)

    mock_client.post.assert_called_once_with(
        endpoint="c1",
        data={"daily_budget": 7550},
    )


@pytest.mark.asyncio
async def test_update_status_sends_correct_status():
    """update_status envia PAUSED corretamente via Graph API."""
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value={"success": True})
    mock_client.close = AsyncMock()

    with patch(f"{_FB}._get_graph_client", new_callable=AsyncMock, return_value=(mock_client, "act_123")):
        await update_status("c1", "PAUSED")

    mock_client.post.assert_called_once_with(
        endpoint="c1",
        data={"status": "PAUSED"},
    )
    mock_client.close.assert_called_once()


@pytest.mark.asyncio
async def test_update_status_active():
    """update_status envia ACTIVE corretamente."""
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value={"success": True})
    mock_client.close = AsyncMock()

    with patch(f"{_FB}._get_graph_client", new_callable=AsyncMock, return_value=(mock_client, "act_123")):
        await update_status("c1", "ACTIVE")

    mock_client.post.assert_called_once_with(
        endpoint="c1",
        data={"status": "ACTIVE"},
    )


@pytest.mark.asyncio
async def test_update_budget_closes_client_on_error():
    """update_budget fecha client mesmo quando Graph API falha."""
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(side_effect=Exception("API Error"))
    mock_client.close = AsyncMock()

    with patch(f"{_FB}._get_graph_client", new_callable=AsyncMock, return_value=(mock_client, "act_123")):
        with pytest.raises(Exception, match="API Error"):
            await update_budget("c1", 50.0)

    # Client deve ser fechado MESMO com erro (finally)
    mock_client.close.assert_called_once()


@pytest.mark.asyncio
async def test_update_status_closes_client_on_error():
    """update_status fecha client mesmo quando Graph API falha."""
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(side_effect=Exception("Connection refused"))
    mock_client.close = AsyncMock()

    with patch(f"{_FB}._get_graph_client", new_callable=AsyncMock, return_value=(mock_client, "act_123")):
        with pytest.raises(Exception, match="Connection refused"):
            await update_status("c1", "PAUSED")

    mock_client.close.assert_called_once()


@pytest.mark.asyncio
async def test_update_budget_round_avoids_float_precision_loss():
    """update_budget usa round() para evitar perda de precisao float.

    75.55 * 100 = 7554.999... → int() daria 7554, round() da 7555.
    """
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value={"success": True})
    mock_client.close = AsyncMock()

    with patch(f"{_FB}._get_graph_client", new_callable=AsyncMock, return_value=(mock_client, "act_123")):
        await update_budget("c1", 75.55)

    mock_client.post.assert_called_once_with(
        endpoint="c1",
        data={"daily_budget": 7555},  # round(75.55 * 100) = 7555, nao 7554
    )


@pytest.mark.asyncio
async def test_update_status_invalid_status_raises():
    """update_status rejeita status invalido com ValueError."""
    with pytest.raises(ValueError, match="Status invalido"):
        await update_status("c1", "DELETED")


@pytest.mark.asyncio
async def test_get_graph_client_campaign_not_found():
    """_get_graph_client levanta ValueError para campanha inexistente."""
    mock_session = AsyncMock()
    mock_result = MagicMock()
    mock_result.one_or_none.return_value = None
    mock_session.execute = AsyncMock(return_value=mock_result)

    mock_ctx = AsyncMock()
    mock_ctx.__aenter__ = AsyncMock(return_value=mock_session)
    mock_ctx.__aexit__ = AsyncMock(return_value=False)

    with patch(f"{_FB}.async_session_maker", return_value=mock_ctx):
        with pytest.raises(ValueError, match="nao encontrada ou config inativa"):
            await _get_graph_client("inexistente")
