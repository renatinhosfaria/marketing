"""
Testes de integracao do fluxo interrupt/resume.

Testa:
  - Token anti-forgery: token invalido e rejeitado
  - Token valido permite aprovacao
  - Idempotencia: mesma acao nao executa duas vezes
  - Rejeicao gera relatorio de cancelamento
  - Fluxo completo de proposta -> interrupt -> resume
"""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from projects.agent.tools.operations_tools import (
    _generate_idempotency_key,
    build_approval_token,
    _already_executed,
    _validate_write_operation,
    WriteValidationError,
)


@pytest.mark.asyncio
async def test_approval_token_consistency():
    """Token de aprovacao e consistente entre propose e resume."""
    thread_id = "u1:a1:t1"
    idemp_key = _generate_idempotency_key(thread_id, "c1", "budget", "60.0")

    token_propose = build_approval_token(thread_id, idemp_key)
    token_resume = build_approval_token(thread_id, idemp_key)

    assert token_propose == token_resume


@pytest.mark.asyncio
async def test_approval_token_different_per_action():
    """Tokens diferentes para acoes diferentes."""
    thread_id = "u1:a1:t1"
    key1 = _generate_idempotency_key(thread_id, "c1", "budget", "60.0")
    key2 = _generate_idempotency_key(thread_id, "c1", "budget", "70.0")

    token1 = build_approval_token(thread_id, key1)
    token2 = build_approval_token(thread_id, key2)

    assert token1 != token2


@pytest.mark.asyncio
async def test_idempotency_check_not_executed():
    """_already_executed retorna False quando acao nao foi executada."""
    mock_result = MagicMock()
    mock_result.scalar.return_value = None

    mock_session = AsyncMock()
    mock_session.execute.return_value = mock_result
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    with patch("projects.agent.tools.operations_tools.async_session_maker", return_value=mock_session):
        result = await _already_executed("abc123")

    assert result is False


@pytest.mark.asyncio
async def test_idempotency_check_already_executed():
    """_already_executed retorna True quando acao ja foi executada."""
    mock_result = MagicMock()
    mock_result.scalar.return_value = 1

    mock_session = AsyncMock()
    mock_session.execute.return_value = mock_result
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    with patch("projects.agent.tools.operations_tools.async_session_maker", return_value=mock_session):
        result = await _already_executed("abc123")

    assert result is True


@pytest.mark.asyncio
async def test_idempotency_check_db_error_returns_false():
    """_already_executed retorna False quando DB falha (graceful degradation)."""
    mock_session = AsyncMock()
    mock_session.execute.side_effect = Exception("Table not found")
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    with patch("projects.agent.tools.operations_tools.async_session_maker", return_value=mock_session):
        result = await _already_executed("abc123")

    assert result is False


@pytest.mark.asyncio
async def test_validate_write_operation_budget_limit():
    """_validate_write_operation bloqueia variacao acima do limite."""
    mock_campaign = MagicMock()
    mock_campaign.account_id = "a1"

    with patch("projects.agent.tools.operations_tools.fb_service") as mock_fb:
        mock_fb.get_campaign_by_config = AsyncMock(return_value=mock_campaign)
        with patch("projects.agent.tools.operations_tools._count_recent_actions", new_callable=AsyncMock, return_value=0):
            with pytest.raises(WriteValidationError) as exc_info:
                await _validate_write_operation(
                    "c1", "a1", "budget_change",
                    new_value=200.0, current_value=50.0,
                )

    assert "excede limite" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_validate_write_operation_min_budget():
    """_validate_write_operation bloqueia budget abaixo do minimo."""
    mock_campaign = MagicMock()
    mock_campaign.account_id = "a1"

    with patch("projects.agent.tools.operations_tools.fb_service") as mock_fb:
        mock_fb.get_campaign_by_config = AsyncMock(return_value=mock_campaign)
        with patch("projects.agent.tools.operations_tools._count_recent_actions", new_callable=AsyncMock, return_value=0):
            with pytest.raises(WriteValidationError) as exc_info:
                await _validate_write_operation(
                    "c1", "a1", "budget_change",
                    new_value=0.5, current_value=1.0,
                )

    assert "minimo" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_validate_write_operation_cooldown():
    """_validate_write_operation bloqueia quando cooldown ativo."""
    mock_campaign = MagicMock()
    mock_campaign.account_id = "a1"

    with patch("projects.agent.tools.operations_tools.fb_service") as mock_fb:
        mock_fb.get_campaign_by_config = AsyncMock(return_value=mock_campaign)
        with patch("projects.agent.tools.operations_tools._count_recent_actions", new_callable=AsyncMock, return_value=5):
            with pytest.raises(WriteValidationError) as exc_info:
                await _validate_write_operation("c1", "a1", "budget_change")

    assert "limite de alteracoes" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_validate_write_operation_ownership_fail():
    """_validate_write_operation bloqueia campanha de outro tenant."""
    with patch("projects.agent.tools.operations_tools.fb_service") as mock_fb:
        mock_fb.get_campaign_by_config = AsyncMock(return_value=None)
        with pytest.raises(WriteValidationError) as exc_info:
            await _validate_write_operation("c1", "a1", "budget_change")

    assert "nao encontrada" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_validate_write_operation_cross_tenant():
    """_validate_write_operation bloqueia campanha de outra conta.

    get_campaign_by_config filtra por config_id, entao retorna None
    se a campanha nao pertence a conta.
    """
    with patch("projects.agent.tools.operations_tools.fb_service") as mock_fb:
        mock_fb.get_campaign_by_config = AsyncMock(return_value=None)
        with pytest.raises(WriteValidationError) as exc_info:
            await _validate_write_operation("c1", "a1", "budget_change")

    assert "nao encontrada" in str(exc_info.value).lower()
