"""
Testes de autenticacao do Agent API.

Testa:
  - verify_api_key: rejeita request sem header X-API-Key (401)
  - verify_api_key: rejeita API key invalida (403)
  - verify_api_key: aceita API key valida
  - verify_api_key: retorna dev user quando auth desabilitada
  - verify_api_key: retorna 500 quando hash nao configurado
  - _validate_secrets: fail-fast em producao com defaults
"""

import hashlib
from unittest.mock import MagicMock, patch, AsyncMock

import pytest
from fastapi import HTTPException

from projects.agent.api.dependencies import verify_api_key, AuthUser


VALID_KEY = "minha-chave-secreta-123"
VALID_HASH = hashlib.sha256(VALID_KEY.encode()).hexdigest()


def _mock_request(api_key: str | None = None) -> MagicMock:
    """Cria mock de Request com header X-API-Key opcional."""
    request = MagicMock()
    headers = {}
    if api_key is not None:
        headers["X-API-Key"] = api_key
    request.headers = headers
    return request


@pytest.mark.asyncio
async def test_rejects_missing_api_key():
    """Request sem header X-API-Key retorna 401."""
    request = _mock_request(api_key=None)

    with patch(
        "projects.agent.api.dependencies.agent_settings"
    ) as mock_settings:
        mock_settings.require_auth = True
        mock_settings.api_key_hash = VALID_HASH
        mock_settings.runtime_user_id = "system"
        mock_settings.runtime_user_name = "System User"

        with pytest.raises(HTTPException) as exc_info:
            await verify_api_key(request)

    assert exc_info.value.status_code == 401
    assert "X-API-Key" in exc_info.value.detail


@pytest.mark.asyncio
async def test_rejects_invalid_api_key():
    """Request com API key invalida retorna 403."""
    request = _mock_request(api_key="chave-errada")

    with patch(
        "projects.agent.api.dependencies.agent_settings"
    ) as mock_settings:
        mock_settings.require_auth = True
        mock_settings.api_key_hash = VALID_HASH
        mock_settings.runtime_user_id = "system"
        mock_settings.runtime_user_name = "System User"

        with pytest.raises(HTTPException) as exc_info:
            await verify_api_key(request)

    assert exc_info.value.status_code == 403
    assert "invalida" in exc_info.value.detail.lower()


@pytest.mark.asyncio
async def test_accepts_valid_api_key():
    """Request com API key valida retorna AuthUser."""
    request = _mock_request(api_key=VALID_KEY)

    with patch(
        "projects.agent.api.dependencies.agent_settings"
    ) as mock_settings:
        mock_settings.require_auth = True
        mock_settings.api_key_hash = VALID_HASH
        mock_settings.runtime_user_id = "system"
        mock_settings.runtime_user_name = "System User"

        user = await verify_api_key(request)

    assert isinstance(user, AuthUser)
    assert user.user_id == "system"


@pytest.mark.asyncio
async def test_auth_disabled_allows_any_request():
    """Com require_auth=False, qualquer request e aceita."""
    request = _mock_request(api_key=None)

    with patch(
        "projects.agent.api.dependencies.agent_settings"
    ) as mock_settings:
        mock_settings.require_auth = False
        mock_settings.runtime_user_id = "system"
        mock_settings.runtime_user_name = "System User"

        user = await verify_api_key(request)

    assert isinstance(user, AuthUser)
    assert user.user_id == "system"


@pytest.mark.asyncio
async def test_returns_500_when_hash_not_configured():
    """Se api_key_hash esta vazio e auth habilitada, retorna 500."""
    request = _mock_request(api_key="qualquer-key")

    with patch(
        "projects.agent.api.dependencies.agent_settings"
    ) as mock_settings:
        mock_settings.require_auth = True
        mock_settings.api_key_hash = ""
        mock_settings.runtime_user_id = "system"
        mock_settings.runtime_user_name = "System User"

        with pytest.raises(HTTPException) as exc_info:
            await verify_api_key(request)

    assert exc_info.value.status_code == 500
    assert "nao configurada" in exc_info.value.detail.lower()


# --- Testes de _validate_secrets ---


def test_validate_secrets_allows_default_approval_token_when_auth_disabled():
    """_validate_secrets nao bloqueia em modo endpoint publico sem auth."""
    from app.agent_main import _validate_secrets

    with (
        patch("app.agent_main.settings") as mock_settings,
        patch("app.agent_main.agent_settings") as mock_agent,
    ):
        mock_settings.environment = "production"
        mock_agent.approval_token_secret = "change-me-in-env"
        mock_agent.require_auth = False
        mock_agent.api_key_hash = ""

        _validate_secrets()  # nao deve levantar erro neste modo operacional


def test_validate_secrets_fails_without_api_key_hash_in_prod():
    """_validate_secrets bloqueia startup sem api_key_hash em prod com auth ativa."""
    from app.agent_main import _validate_secrets

    with (
        patch("app.agent_main.settings") as mock_settings,
        patch("app.agent_main.agent_settings") as mock_agent,
    ):
        mock_settings.environment = "production"
        mock_agent.approval_token_secret = "um-secret-real-aqui"
        mock_agent.require_auth = True
        mock_agent.api_key_hash = ""

        with pytest.raises(RuntimeError, match="AGENT_API_KEY_HASH"):
            _validate_secrets()


def test_validate_secrets_passes_with_valid_config():
    """_validate_secrets nao levanta erro com config valida em prod."""
    from app.agent_main import _validate_secrets

    with (
        patch("app.agent_main.settings") as mock_settings,
        patch("app.agent_main.agent_settings") as mock_agent,
    ):
        mock_settings.environment = "production"
        mock_agent.approval_token_secret = "um-secret-real"
        mock_agent.require_auth = True
        mock_agent.api_key_hash = VALID_HASH

        _validate_secrets()  # Nao deve levantar erro


def test_validate_secrets_skips_in_development():
    """_validate_secrets nao valida em ambiente de desenvolvimento."""
    from app.agent_main import _validate_secrets

    with (
        patch("app.agent_main.settings") as mock_settings,
        patch("app.agent_main.agent_settings") as mock_agent,
    ):
        mock_settings.environment = "development"
        mock_agent.approval_token_secret = "change-me-in-env"
        mock_agent.require_auth = True
        mock_agent.api_key_hash = ""

        _validate_secrets()  # Nao deve levantar erro em dev
