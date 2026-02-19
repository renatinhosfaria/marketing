"""
Testes das tools do Cientista de Previsao.

Testa:
  - generate_forecast: chamada ML API com parametros corretos
  - get_forecast_history: historico de previsoes
  - validate_forecast: validacao previsto vs realizado
  - ToolResult contract (ok/data/error)
  - Conta nao encontrada
"""

import pytest
from unittest.mock import patch, AsyncMock

from projects.agent.tools.result import tool_success, tool_error


@pytest.mark.asyncio
async def test_generate_forecast_success():
    """generate_forecast chama ML API e retorna previsao."""
    mock_forecast = {
        "entity_id": "c1",
        "metric": "cpl",
        "predicted_values": [25.0, 24.5, 24.0, 23.5, 23.0, 22.5, 22.0],
        "horizon_days": 7,
    }

    with patch("projects.agent.tools.forecast_tools.resolve_config_id", new_callable=AsyncMock, return_value=1), \
         patch("projects.agent.tools.forecast_tools._validate_entity_ownership", new_callable=AsyncMock, return_value=True), \
         patch("projects.agent.tools.forecast_tools._ml_api_call", new_callable=AsyncMock, return_value=tool_success(mock_forecast)):
        from projects.agent.tools.forecast_tools import generate_forecast
        result = await generate_forecast.ainvoke(
            {
                "entity_id": "c1",
                "metric": "cpl",
                "horizon_days": 7,
            },
            config={"configurable": {"account_id": "a1"}},
        )

    assert result["ok"] is True
    assert result["data"]["metric"] == "cpl"
    assert len(result["data"]["predicted_values"]) == 7


@pytest.mark.asyncio
async def test_generate_forecast_account_not_found():
    """generate_forecast retorna erro quando conta nao encontrada."""
    with patch("projects.agent.tools.forecast_tools.resolve_config_id", new_callable=AsyncMock, return_value=None):
        from projects.agent.tools.forecast_tools import generate_forecast
        result = await generate_forecast.ainvoke(
            {"entity_id": "c1", "metric": "cpl"},
            config={"configurable": {"account_id": "inexistente"}},
        )

    assert result["ok"] is False
    assert result["error"]["code"] == "NOT_FOUND"


@pytest.mark.asyncio
async def test_generate_forecast_ml_api_timeout():
    """generate_forecast propaga timeout da ML API."""
    ml_error = tool_error("TIMEOUT", "ML API timeout", retryable=True)

    with patch("projects.agent.tools.forecast_tools.resolve_config_id", new_callable=AsyncMock, return_value=1), \
         patch("projects.agent.tools.forecast_tools._validate_entity_ownership", new_callable=AsyncMock, return_value=True), \
         patch("projects.agent.tools.forecast_tools._ml_api_call", new_callable=AsyncMock, return_value=ml_error):
        from projects.agent.tools.forecast_tools import generate_forecast
        result = await generate_forecast.ainvoke(
            {"entity_id": "c1", "metric": "leads", "horizon_days": 14},
            config={"configurable": {"account_id": "a1"}},
        )

    assert result["ok"] is False
    assert result["error"]["retryable"] is True


@pytest.mark.asyncio
async def test_generate_forecast_rejects_unsupported_metric():
    """generate_forecast rejeita metrica fora da capacidade atual do backend."""
    with patch("projects.agent.tools.forecast_tools.resolve_config_id", new_callable=AsyncMock, return_value=1), \
         patch("projects.agent.tools.forecast_tools._validate_entity_ownership", new_callable=AsyncMock, return_value=True):
        from projects.agent.tools.forecast_tools import generate_forecast
        result = await generate_forecast.ainvoke(
            {"entity_id": "c1", "metric": "spend"},
            config={"configurable": {"account_id": "a1"}},
        )

    assert result["ok"] is False
    assert result["error"]["code"] == "NOT_SUPPORTED"


@pytest.mark.asyncio
async def test_get_forecast_history_success():
    """get_forecast_history retorna historico de previsoes."""
    mock_history = [
        {"forecast_id": 1, "metric": "cpl", "predicted_value": 25.0, "actual_value": 26.0},
    ]
    ml_call = AsyncMock(return_value=tool_success(mock_history))

    with patch("projects.agent.tools.forecast_tools.resolve_config_id", new_callable=AsyncMock, return_value=1), \
         patch("projects.agent.tools.forecast_tools._ml_api_call", ml_call):
        from projects.agent.tools.forecast_tools import get_forecast_history
        result = await get_forecast_history.ainvoke(
            {"entity_type": "campaign", "entity_id": "c1"},
            config={"configurable": {"account_id": "a1"}},
        )

    assert result["ok"] is True
    assert len(result["data"]) == 1
    ml_call.assert_called_once()
    assert ml_call.call_args[0][0] == "get"
    assert ml_call.call_args[0][1] == "/api/v1/forecasts"


@pytest.mark.asyncio
async def test_validate_forecast_not_supported():
    """validate_forecast retorna NOT_SUPPORTED ate endpoint real existir."""
    with patch("projects.agent.tools.forecast_tools.resolve_config_id", new_callable=AsyncMock, return_value=1):
        from projects.agent.tools.forecast_tools import validate_forecast
        result = await validate_forecast.ainvoke(
            {"forecast_id": "f1"},
            config={"configurable": {"account_id": "a1"}},
        )

    assert result["ok"] is False
    assert result["error"]["code"] == "NOT_SUPPORTED"
