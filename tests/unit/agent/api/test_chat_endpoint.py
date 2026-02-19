"""
Testes do endpoint /chat e funcoes auxiliares do router.

Testa:
  - _build_thread_id: prefixo tenant isolamento
  - _build_thread_id: validacao de thread_id cross-tenant
  - _get_semaphore: criacao e reuso de semaforos
  - health endpoint: retorna status correto
  - SSE stream format (sse_event, _safe_json)
"""

import pytest
import json
import asyncio
from unittest.mock import patch, MagicMock

from fastapi import HTTPException


@pytest.mark.asyncio
async def test_build_thread_id_prefixes():
    """_build_thread_id adiciona prefixo user:account ao thread_id."""
    from projects.agent.api.router import _build_thread_id

    result = _build_thread_id("abc-123", "u1", "a1")
    assert result == "u1:a1:abc-123"


@pytest.mark.asyncio
async def test_build_thread_id_already_prefixed():
    """_build_thread_id nao duplica prefixo quando ja esta presente."""
    from projects.agent.api.router import _build_thread_id

    result = _build_thread_id("u1:a1:abc-123", "u1", "a1")
    assert result == "u1:a1:abc-123"


@pytest.mark.asyncio
async def test_build_thread_id_cross_tenant_rejection():
    """_build_thread_id rejeita thread_id de outro tenant."""
    from projects.agent.api.router import _build_thread_id

    with pytest.raises(HTTPException) as exc_info:
        _build_thread_id("u2:a2:abc-123", "u1", "a1")

    assert exc_info.value.status_code == 400
    assert "outro tenant" in exc_info.value.detail.lower()


@pytest.mark.asyncio
async def test_get_semaphore_creates_and_reuses():
    """_get_semaphore cria semaforo e reutiliza para mesmo usuario."""
    from projects.agent.api.router import _get_semaphore, _stream_semaphores

    # Limpar estado
    _stream_semaphores.clear()

    sem1 = _get_semaphore("user_test_1")
    sem2 = _get_semaphore("user_test_1")
    sem3 = _get_semaphore("user_test_2")

    assert sem1 is sem2  # Mesmo usuario, mesmo semaforo
    assert sem1 is not sem3  # Usuarios diferentes, semaforos diferentes

    # Cleanup
    _stream_semaphores.clear()


@pytest.mark.asyncio
async def test_validate_account_id_accepts_act_prefix():
    """_validate_account_id aceita formato act_<digitos>."""
    from projects.agent.api.router import _validate_account_id

    assert _validate_account_id("act_123456") == "act_123456"


@pytest.mark.asyncio
async def test_validate_account_id_accepts_digits_only():
    """_validate_account_id aceita formato somente digitos."""
    from projects.agent.api.router import _validate_account_id

    assert _validate_account_id("123456") == "123456"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "invalid_account_id",
    ["abc", "act_%", "12_34", "12 34", '12"34'],
)
async def test_validate_account_id_rejects_invalid_values(invalid_account_id: str):
    """_validate_account_id rejeita account_id com caracteres invalidos."""
    from projects.agent.api.router import _validate_account_id

    with pytest.raises(HTTPException) as exc_info:
        _validate_account_id(invalid_account_id)

    assert exc_info.value.status_code == 400
    assert "account_id invalido" in exc_info.value.detail.lower()


@pytest.mark.asyncio
async def test_health_endpoint():
    """GET /health retorna status saudavel."""
    mock_settings = MagicMock()
    mock_settings.agent_version = "1.0.0"

    with patch("projects.agent.api.router.agent_settings", mock_settings):
        from projects.agent.api.router import health
        result = await health()

    assert result["status"] == "healthy"
    assert result["service"] == "famachat-agent-api"
    assert result["version"] == "1.0.0"


@pytest.mark.asyncio
async def test_sse_event_format():
    """sse_event formata evento SSE corretamente."""
    from projects.agent.api.stream import sse_event

    event = sse_event("message", {"content": "Ola!", "agent": "supervisor"})

    assert event.startswith("event: message\n")
    assert "data: " in event
    assert event.endswith("\n\n")

    # Parse data JSON
    data_line = event.split("\n")[1]
    data_json = json.loads(data_line.replace("data: ", ""))
    assert data_json["content"] == "Ola!"
    assert data_json["agent"] == "supervisor"


@pytest.mark.asyncio
async def test_sse_event_preserves_accents():
    """sse_event preserva acentos em portugues (ensure_ascii=False)."""
    from projects.agent.api.stream import sse_event

    event = sse_event("message", {"content": "Previsao de orcamento"})
    assert "Previsao" in event
    assert "orcamento" in event
    # Nao deve ter unicode escapes
    assert "\\u" not in event


@pytest.mark.asyncio
async def test_safe_json_valid_json():
    """_safe_json parseia JSON valido."""
    from projects.agent.api.stream import _safe_json

    result = _safe_json('{"key": "value"}')
    assert isinstance(result, dict)
    assert result["key"] == "value"


@pytest.mark.asyncio
async def test_safe_json_invalid_json():
    """_safe_json retorna string original quando JSON invalido."""
    from projects.agent.api.stream import _safe_json

    result = _safe_json("texto simples")
    assert isinstance(result, str)
    assert result == "texto simples"


@pytest.mark.asyncio
async def test_safe_json_none_input():
    """_safe_json trata None como input."""
    from projects.agent.api.stream import _safe_json

    result = _safe_json(None)
    assert result is None


@pytest.mark.asyncio
async def test_resolve_stream_status_cancelled_task():
    """_resolve_stream_status trata task cancelada como cancelled sem exception."""
    from projects.agent.api.router import _resolve_stream_status

    async def _never():
        await asyncio.sleep(0.1)

    task = asyncio.create_task(_never())
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert _resolve_stream_status(task) == "cancelled"
