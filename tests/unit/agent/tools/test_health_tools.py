"""
Testes das tools do Monitor de Saude & Anomalias.

Testa:
  - detect_anomalies: chamada ML API + validacao de schema
  - get_classifications: chamada ML API
  - classify_entity: chamada ML API
  - get_anomaly_history: chamada ML API
  - ToolResult contract (ok/data/error)
  - Tratamento de erros (timeout, schema mismatch)
"""

import pytest
from unittest.mock import patch, AsyncMock

from projects.agent.tools.result import tool_success, tool_error


@pytest.mark.asyncio
async def test_detect_anomalies_success():
    """detect_anomalies retorna ToolResult com ok=True quando ML API responde."""
    mock_ml_response = {
        "config_id": 1,
        "detected_count": 2,
        "anomalies": [
            {
                "entity_type": "campaign",
                "entity_id": "c1",
                "anomaly_type": "isolation_forest",
                "metric_name": "cpl",
                "observed_value": 40.0,
                "expected_value": 25.0,
                "deviation_score": 3.5,
                "severity": "HIGH",
                "is_acknowledged": False,
            },
            {
                "entity_type": "campaign",
                "entity_id": "c2",
                "anomaly_type": "z_score",
                "metric_name": "ctr",
                "observed_value": 0.5,
                "expected_value": 2.0,
                "deviation_score": -2.8,
                "severity": "MEDIUM",
                "is_acknowledged": False,
            },
        ],
    }

    with patch("projects.agent.tools.health_tools.resolve_config_id", new_callable=AsyncMock, return_value=1), \
         patch("projects.agent.tools.health_tools._ml_api_call", new_callable=AsyncMock, return_value=tool_success(mock_ml_response)):
        from projects.agent.tools.health_tools import detect_anomalies
        result = await detect_anomalies.ainvoke(
            {"entity_type": "campaign", "days": 1},
            config={"configurable": {"account_id": "a1"}},
        )

    assert result["ok"] is True
    assert result["data"]["detected_count"] == 2
    assert len(result["data"]["anomalies"]) == 2


@pytest.mark.asyncio
async def test_detect_anomalies_account_not_found():
    """detect_anomalies retorna erro quando conta nao e encontrada."""
    with patch("projects.agent.tools.health_tools.resolve_config_id", new_callable=AsyncMock, return_value=None):
        from projects.agent.tools.health_tools import detect_anomalies
        result = await detect_anomalies.ainvoke(
            {"entity_type": "campaign"},
            config={"configurable": {"account_id": "inexistente"}},
        )

    assert result["ok"] is False
    assert result["error"]["code"] == "NOT_FOUND"


@pytest.mark.asyncio
async def test_detect_anomalies_schema_mismatch():
    """detect_anomalies retorna SCHEMA_MISMATCH quando ML API retorna dados invalidos."""
    # Resposta sem campos obrigatorios
    bad_response = {"invalid": "data"}

    with patch("projects.agent.tools.health_tools.resolve_config_id", new_callable=AsyncMock, return_value=1), \
         patch("projects.agent.tools.health_tools._ml_api_call", new_callable=AsyncMock, return_value=tool_success(bad_response)):
        from projects.agent.tools.health_tools import detect_anomalies
        result = await detect_anomalies.ainvoke(
            {"entity_type": "campaign"},
            config={"configurable": {"account_id": "a1"}},
        )

    assert result["ok"] is False
    assert result["error"]["code"] == "SCHEMA_MISMATCH"


@pytest.mark.asyncio
async def test_get_classifications_success():
    """get_classifications retorna classificacoes da ML API."""
    mock_data = [
        {"entity_id": "c1", "tier": "HIGH_PERFORMER", "confidence_score": 0.95},
    ]

    with patch("projects.agent.tools.health_tools.resolve_config_id", new_callable=AsyncMock, return_value=1), \
         patch("projects.agent.tools.health_tools._ml_api_call", new_callable=AsyncMock, return_value=tool_success(mock_data)):
        from projects.agent.tools.health_tools import get_classifications
        result = await get_classifications.ainvoke(
            {"entity_type": "campaign"},
            config={"configurable": {"account_id": "a1"}},
        )

    assert result["ok"] is True
    assert result["data"][0]["tier"] == "HIGH_PERFORMER"


@pytest.mark.asyncio
async def test_get_anomaly_history_success():
    """get_anomaly_history retorna historico de anomalias."""
    mock_history = [{"entity_id": "c1", "metric_name": "cpl", "severity": "HIGH"}]

    with patch("projects.agent.tools.health_tools.resolve_config_id", new_callable=AsyncMock, return_value=1), \
         patch("projects.agent.tools.health_tools._ml_api_call", new_callable=AsyncMock, return_value=tool_success(mock_history)):
        from projects.agent.tools.health_tools import get_anomaly_history
        result = await get_anomaly_history.ainvoke(
            {"days": 7},
            config={"configurable": {"account_id": "a1"}},
        )

    assert result["ok"] is True
    assert len(result["data"]) == 1


@pytest.mark.asyncio
async def test_detect_anomalies_ml_api_error():
    """detect_anomalies propaga erro da ML API."""
    ml_error = tool_error("TIMEOUT", "ML API timeout", retryable=True)

    with patch("projects.agent.tools.health_tools.resolve_config_id", new_callable=AsyncMock, return_value=1), \
         patch("projects.agent.tools.health_tools._ml_api_call", new_callable=AsyncMock, return_value=ml_error):
        from projects.agent.tools.health_tools import detect_anomalies
        result = await detect_anomalies.ainvoke(
            {"entity_type": "campaign"},
            config={"configurable": {"account_id": "a1"}},
        )

    assert result["ok"] is False
    assert result["error"]["code"] == "TIMEOUT"
    assert result["error"]["retryable"] is True
