"""
Testes de retry com jitter do _ml_api_call.

Verifica:
  - Retenta em TIMEOUT (ate _MAX_RETRY_ATTEMPTS vezes)
  - Retenta em ConnectError
  - Retenta em 5xx
  - NAO retenta em 4xx (falha imediata)
  - NAO retenta quando circuit breaker esta aberto
  - Retorna sucesso se uma tentativa subsequente funcionar
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch, call

import httpx
import pytest

from projects.agent.tools.result import tool_success, tool_error


@pytest.fixture(autouse=True)
def reset_http_client():
    """Garante que _ml_http_client esta configurado para cada teste."""
    import projects.agent.tools.http_client as hc
    mock_client = MagicMock()
    original = hc._ml_http_client
    hc._ml_http_client = mock_client
    hc._sem_factory = None  # Usar semaphores in-memory

    yield mock_client

    hc._ml_http_client = original
    hc._sem_factory = None


@pytest.fixture(autouse=True)
def mock_circuit_breaker():
    """Circuit breaker que sempre passa (CLOSED) por padrao.

    Patcha no modulo de origem porque get_registry e importado lazy
    (dentro da funcao _ml_api_call).
    """
    with patch("projects.agent.resilience.circuit_breaker.get_registry") as mock_registry:
        cb = MagicMock()
        cb.is_open = False
        cb.retry_after.return_value = 60.0  # Evita TypeError no f-string :.0f

        async def passthrough(func):
            return await func()

        cb.call = passthrough
        mock_registry.return_value.get.return_value = cb
        yield cb


@pytest.fixture(autouse=True)
def fast_retry(monkeypatch):
    """Remove delays de retry nos testes."""
    monkeypatch.setattr(
        "projects.agent.tools.http_client._retry_delay",
        lambda attempt: 0.0,
    )


@pytest.mark.asyncio
async def test_success_on_first_attempt(reset_http_client):
    """Chamada bem-sucedida na primeira tentativa retorna tool_success."""
    resp = MagicMock()
    resp.status_code = 200
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {"ok": True}
    reset_http_client.get = AsyncMock(return_value=resp)

    from projects.agent.tools.http_client import _ml_api_call
    result = await _ml_api_call("get", "/api/v1/test")

    assert result["ok"] is True
    assert reset_http_client.get.call_count == 1


@pytest.mark.asyncio
async def test_retries_on_timeout(reset_http_client):
    """TIMEOUT retenta ate _MAX_RETRY_ATTEMPTS e retorna TIMEOUT."""
    reset_http_client.get = AsyncMock(side_effect=httpx.TimeoutException("timeout"))

    from projects.agent.tools.http_client import _ml_api_call, _MAX_RETRY_ATTEMPTS
    result = await _ml_api_call("get", "/api/v1/test")

    assert result["ok"] is False
    assert result["error"]["code"] == "TIMEOUT"
    assert result["error"]["retryable"] is True
    assert reset_http_client.get.call_count == _MAX_RETRY_ATTEMPTS


@pytest.mark.asyncio
async def test_retries_on_connect_error(reset_http_client):
    """ConnectError retenta ate _MAX_RETRY_ATTEMPTS e retorna UNAVAILABLE."""
    reset_http_client.get = AsyncMock(
        side_effect=httpx.ConnectError("connection refused")
    )

    from projects.agent.tools.http_client import _ml_api_call, _MAX_RETRY_ATTEMPTS
    result = await _ml_api_call("get", "/api/v1/test")

    assert result["ok"] is False
    assert result["error"]["code"] == "UNAVAILABLE"
    assert reset_http_client.get.call_count == _MAX_RETRY_ATTEMPTS


@pytest.mark.asyncio
async def test_retries_on_5xx(reset_http_client):
    """5xx retenta ate _MAX_RETRY_ATTEMPTS e retorna HTTP_ERROR retryable."""
    resp_5xx = MagicMock()
    resp_5xx.status_code = 503
    http_err = httpx.HTTPStatusError(
        "503", request=MagicMock(), response=resp_5xx
    )
    resp_5xx.raise_for_status = MagicMock(side_effect=http_err)
    reset_http_client.get = AsyncMock(return_value=resp_5xx)

    from projects.agent.tools.http_client import _ml_api_call, _MAX_RETRY_ATTEMPTS
    result = await _ml_api_call("get", "/api/v1/test")

    assert result["ok"] is False
    assert result["error"]["code"] == "HTTP_ERROR"
    assert result["error"]["retryable"] is True
    assert reset_http_client.get.call_count == _MAX_RETRY_ATTEMPTS


@pytest.mark.asyncio
async def test_no_retry_on_4xx(reset_http_client):
    """4xx NAO retenta — falha imediata."""
    resp_4xx = MagicMock()
    resp_4xx.status_code = 404
    http_err = httpx.HTTPStatusError(
        "404", request=MagicMock(), response=resp_4xx
    )
    resp_4xx.raise_for_status = MagicMock(side_effect=http_err)
    reset_http_client.get = AsyncMock(return_value=resp_4xx)

    from projects.agent.tools.http_client import _ml_api_call
    result = await _ml_api_call("get", "/api/v1/test")

    assert result["ok"] is False
    assert result["error"]["code"] == "HTTP_ERROR"
    assert result["error"]["retryable"] is False
    assert reset_http_client.get.call_count == 1  # Sem retry


@pytest.mark.asyncio
async def test_success_on_second_attempt(reset_http_client):
    """Falha na 1a tentativa + sucesso na 2a retorna tool_success."""
    resp_ok = MagicMock()
    resp_ok.status_code = 200
    resp_ok.raise_for_status = MagicMock()
    resp_ok.json.return_value = {"data": "ok"}

    reset_http_client.get = AsyncMock(
        side_effect=[httpx.TimeoutException("timeout"), resp_ok]
    )

    from projects.agent.tools.http_client import _ml_api_call
    result = await _ml_api_call("get", "/api/v1/test")

    assert result["ok"] is True
    assert reset_http_client.get.call_count == 2


@pytest.mark.asyncio
async def test_no_retry_when_circuit_open(reset_http_client, mock_circuit_breaker):
    """Quando circuit breaker esta aberto, nao tenta a chamada HTTP."""
    from projects.agent.resilience.circuit_breaker import CircuitOpenError

    mock_circuit_breaker.is_open = True

    from projects.agent.tools.http_client import _ml_api_call
    result = await _ml_api_call("get", "/api/v1/test")

    assert result["ok"] is False
    assert result["error"]["code"] == "UNAVAILABLE"
    # Client HTTP nao deve ter sido chamado
    assert reset_http_client.get.call_count == 0
