"""
Testes das tools puras do Gerente de Operacoes (prepare_/execute_).

Testa:
  - prepare_budget_change: validacao, idempotency key
  - prepare_status_change: validacao
  - get_recommendations: chamada ML API
  - _generate_idempotency_key: deterministico
  - build_approval_token: deterministico
  - _validate_write_operation: limites de budget, ownership, cooldown
  - execute_budget_change: erro FB API
  - execute_status_change: erro FB API
"""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from projects.agent.tools.result import tool_success
from projects.agent.tools.operations_tools import (
    prepare_budget_change,
    prepare_status_change,
    get_recommendations,
    execute_budget_change,
    execute_status_change,
    _count_recent_actions,
    _generate_idempotency_key,
    build_approval_token,
    _claim_action_log,
    _mark_action_log_status,
)

# Prefixo para patches no namespace do modulo
_TOOLS = "projects.agent.tools.operations_tools"


@pytest.mark.asyncio
async def test_prepare_budget_change_success():
    """prepare_budget_change retorna proposta valida."""
    mock_campaign = MagicMock()
    mock_campaign.account_id = "a1"
    mock_campaign.daily_budget = 50.0

    with patch(f"{_TOOLS}.resolve_config_id", new_callable=AsyncMock, return_value=1), \
         patch(f"{_TOOLS}.fb_service") as mock_fb, \
         patch(f"{_TOOLS}._count_recent_actions", new_callable=AsyncMock, return_value=0):
        mock_fb.get_campaign_budget = AsyncMock(return_value=50.0)
        mock_fb.get_campaign_by_config = AsyncMock(return_value=mock_campaign)
        result = await prepare_budget_change.ainvoke(
            {
                "campaign_id": "c1",
                "new_daily_budget": 60.0,
                "reason": "CPL alto",
            },
            config={"configurable": {"account_id": "a1", "thread_id": "t1"}},
        )

    assert result["ok"] is True
    assert result["data"]["action_type"] == "budget_change"
    assert result["data"]["current_value"] == 50.0
    assert result["data"]["new_value"] == 60.0
    assert "idempotency_key" in result["data"]


@pytest.mark.asyncio
async def test_prepare_budget_change_exceeds_limit():
    """prepare_budget_change rejeita variacao acima do limite."""
    mock_campaign = MagicMock()
    mock_campaign.account_id = "a1"
    mock_campaign.daily_budget = 50.0

    with patch(f"{_TOOLS}.resolve_config_id", new_callable=AsyncMock, return_value=1), \
         patch(f"{_TOOLS}.fb_service") as mock_fb, \
         patch(f"{_TOOLS}._count_recent_actions", new_callable=AsyncMock, return_value=0):
        mock_fb.get_campaign_budget = AsyncMock(return_value=50.0)
        mock_fb.get_campaign_by_config = AsyncMock(return_value=mock_campaign)
        result = await prepare_budget_change.ainvoke(
            {
                "campaign_id": "c1",
                "new_daily_budget": 200.0,  # 300% aumento (> 50% default)
                "reason": "Teste",
            },
            config={"configurable": {"account_id": "a1", "thread_id": "t1"}},
        )

    assert result["ok"] is False
    assert result["error"]["code"] == "VALIDATION_ERROR"
    assert "excede limite" in result["error"]["message"].lower()


@pytest.mark.asyncio
async def test_prepare_budget_change_negative_value():
    """prepare_budget_change rejeita orcamento negativo com mensagem explicita."""
    mock_campaign = MagicMock()
    mock_campaign.account_id = "a1"
    mock_campaign.daily_budget = 50.0

    with patch(f"{_TOOLS}.resolve_config_id", new_callable=AsyncMock, return_value=1), \
         patch(f"{_TOOLS}.fb_service") as mock_fb, \
         patch(f"{_TOOLS}._count_recent_actions", new_callable=AsyncMock, return_value=0):
        mock_fb.get_campaign_budget = AsyncMock(return_value=50.0)
        mock_fb.get_campaign_by_config = AsyncMock(return_value=mock_campaign)
        result = await prepare_budget_change.ainvoke(
            {
                "campaign_id": "c1",
                "new_daily_budget": -10.0,
                "reason": "Teste",
            },
            config={"configurable": {"account_id": "a1", "thread_id": "t1"}},
        )

    assert result["ok"] is False
    assert result["error"]["code"] == "VALIDATION_ERROR"
    assert "positivo" in result["error"]["message"].lower()


@pytest.mark.asyncio
async def test_prepare_status_change_success():
    """prepare_status_change retorna proposta de mudanca de status."""
    mock_campaign = MagicMock()
    mock_campaign.account_id = "a1"

    with patch(f"{_TOOLS}.resolve_config_id", new_callable=AsyncMock, return_value=1), \
         patch(f"{_TOOLS}.fb_service") as mock_fb, \
         patch(f"{_TOOLS}._count_recent_actions", new_callable=AsyncMock, return_value=0):
        mock_fb.get_campaign_by_config = AsyncMock(return_value=mock_campaign)
        result = await prepare_status_change.ainvoke(
            {
                "campaign_id": "c1",
                "new_status": "PAUSED",
                "reason": "CPL acima do limite",
            },
            config={"configurable": {"account_id": "a1", "thread_id": "t1"}},
        )

    assert result["ok"] is True
    assert result["data"]["action_type"] == "status_change"
    assert result["data"]["new_status"] == "PAUSED"


@pytest.mark.asyncio
async def test_get_recommendations_success():
    """get_recommendations retorna recomendacoes da ML API."""
    mock_recs = [
        {"title": "Aumentar budget da campanha c1", "priority": 1, "confidence_score": 0.90},
    ]

    with patch(f"{_TOOLS}.resolve_config_id", new_callable=AsyncMock, return_value=1), \
         patch(f"{_TOOLS}._ml_api_call", new_callable=AsyncMock, return_value=tool_success(mock_recs)):
        result = await get_recommendations.ainvoke(
            {"entity_type": "campaign"},
            config={"configurable": {"account_id": "a1"}},
        )

    assert result["ok"] is True
    assert len(result["data"]) == 1


@pytest.mark.asyncio
async def test_idempotency_key_deterministic():
    """_generate_idempotency_key produz mesmo resultado para mesmos inputs."""
    key1 = _generate_idempotency_key("t1", "c1", "budget", "60.0")
    key2 = _generate_idempotency_key("t1", "c1", "budget", "60.0")
    key3 = _generate_idempotency_key("t1", "c1", "budget", "70.0")

    assert key1 == key2  # Mesmos inputs = mesma key
    assert key1 != key3  # Inputs diferentes = keys diferentes
    assert len(key1) == 16  # SHA-256[:16]


@pytest.mark.asyncio
async def testbuild_approval_token_deterministic():
    """build_approval_token e deterministico para mesma proposta."""
    token1 = build_approval_token("t1", "abc123")
    token2 = build_approval_token("t1", "abc123")
    token3 = build_approval_token("t2", "abc123")

    assert token1 == token2
    assert token1 != token3
    assert len(token1) == 32


@pytest.mark.asyncio
async def test_execute_budget_change_fb_api_error():
    """execute_budget_change retorna tool_error quando FB API falha."""
    mock_campaign = MagicMock()
    mock_campaign.account_id = "a1"
    mock_campaign.daily_budget = 50.0

    with patch(f"{_TOOLS}.resolve_config_id", new_callable=AsyncMock, return_value=1), \
         patch(f"{_TOOLS}.fb_service") as mock_fb, \
         patch(f"{_TOOLS}._count_recent_actions", new_callable=AsyncMock, return_value=0), \
         patch(f"{_TOOLS}._already_executed", new_callable=AsyncMock, return_value=False):
        mock_fb.get_campaign_budget = AsyncMock(return_value=50.0)
        mock_fb.get_campaign_by_config = AsyncMock(return_value=mock_campaign)
        mock_fb.update_budget = AsyncMock(side_effect=Exception("Graph API timeout"))
        result = await execute_budget_change.ainvoke(
            {
                "campaign_id": "c1",
                "new_daily_budget": 60.0,
                "idempotency_key": "key123",
            },
            config={"configurable": {"account_id": "a1", "thread_id": "t1"}},
        )

    assert result["ok"] is False
    assert result["error"]["code"] == "FB_API_ERROR"
    assert "Graph API timeout" in result["error"]["message"]
    assert result["error"]["retryable"] is True


@pytest.mark.asyncio
async def test_execute_status_change_fb_api_error():
    """execute_status_change retorna tool_error quando FB API falha."""
    mock_campaign = MagicMock()
    mock_campaign.account_id = "a1"

    with patch(f"{_TOOLS}.resolve_config_id", new_callable=AsyncMock, return_value=1), \
         patch(f"{_TOOLS}.fb_service") as mock_fb, \
         patch(f"{_TOOLS}._count_recent_actions", new_callable=AsyncMock, return_value=0), \
         patch(f"{_TOOLS}._already_executed", new_callable=AsyncMock, return_value=False):
        mock_fb.get_campaign_by_config = AsyncMock(return_value=mock_campaign)
        mock_fb.update_status = AsyncMock(side_effect=Exception("Rate limit exceeded"))
        result = await execute_status_change.ainvoke(
            {
                "campaign_id": "c1",
                "new_status": "PAUSED",
                "idempotency_key": "key456",
            },
            config={"configurable": {"account_id": "a1", "thread_id": "t1"}},
        )

    assert result["ok"] is False
    assert result["error"]["code"] == "FB_API_ERROR"
    assert "Rate limit" in result["error"]["message"]


@pytest.mark.asyncio
async def test_execute_budget_change_uses_action_log_claim_for_idempotency():
    """execute_budget_change nao chama FB API se claim de idempotencia falhar."""
    mock_campaign = MagicMock()
    mock_campaign.account_id = "a1"

    with patch(f"{_TOOLS}.resolve_config_id", new_callable=AsyncMock, return_value=1), \
         patch(f"{_TOOLS}.fb_service") as mock_fb, \
         patch(f"{_TOOLS}._count_recent_actions", new_callable=AsyncMock, return_value=0), \
         patch(f"{_TOOLS}._already_executed", new_callable=AsyncMock, return_value=False), \
         patch(f"{_TOOLS}._claim_action_log", new_callable=AsyncMock, return_value=False):
        mock_fb.get_campaign_budget = AsyncMock(return_value=50.0)
        mock_fb.get_campaign_by_config = AsyncMock(return_value=mock_campaign)
        mock_fb.update_budget = AsyncMock()

        result = await execute_budget_change.ainvoke(
            {
                "campaign_id": "c1",
                "new_daily_budget": 60.0,
                "idempotency_key": "key123",
            },
            config={"configurable": {"account_id": "a1", "thread_id": "t1"}},
        )

    assert result["ok"] is True
    assert "idempotencia" in result["data"]["message"].lower()
    mock_fb.update_budget.assert_not_called()


@pytest.mark.asyncio
async def test_execute_budget_change_passes_config_id_to_claim():
    """execute_budget_change passa config_id resolvido para _claim_action_log."""
    mock_campaign = MagicMock()
    mock_campaign.account_id = "a1"

    claim_mock = AsyncMock(return_value=False)

    with patch(f"{_TOOLS}.resolve_config_id", new_callable=AsyncMock, return_value=77), \
         patch(f"{_TOOLS}.fb_service") as mock_fb, \
         patch(f"{_TOOLS}._count_recent_actions", new_callable=AsyncMock, return_value=0), \
         patch(f"{_TOOLS}._already_executed", new_callable=AsyncMock, return_value=False), \
         patch(f"{_TOOLS}._claim_action_log", claim_mock):
        mock_fb.get_campaign_budget = AsyncMock(return_value=50.0)
        mock_fb.get_campaign_by_config = AsyncMock(return_value=mock_campaign)

        await execute_budget_change.ainvoke(
            {
                "campaign_id": "c1",
                "new_daily_budget": 60.0,
                "idempotency_key": "key123",
            },
            config={"configurable": {"account_id": "act_999", "thread_id": "t1", "user_id": "system"}},
        )

    assert claim_mock.await_count == 1
    kwargs = claim_mock.await_args.kwargs
    assert kwargs["config_id"] == 77


@pytest.mark.asyncio
async def test_claim_action_log_persists_trace_fields_and_config_id():
    """_claim_action_log persiste config_id e detalhes de rastreio bruto."""
    mock_result = MagicMock()
    mock_result.scalar.return_value = 1

    mock_session = AsyncMock()
    mock_session.execute.return_value = mock_result
    mock_session.commit = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    with patch(f"{_TOOLS}.async_session_maker", return_value=mock_session):
        claimed = await _claim_action_log(
            idempotency_key="idem-1",
            campaign_id="c1",
            operation_type="budget_change",
            config={"account_id": "act_123", "user_id": "system"},
            config_id=42,
            details={"phase": "claimed"},
        )

    assert claimed is True
    params = mock_session.execute.call_args[0][1]
    assert params["account_id"] == 42
    assert params["details"]["account_id_raw"] == "act_123"
    assert params["details"]["user_id_raw"] == "system"
    assert params["details"]["config_id"] == 42


@pytest.mark.asyncio
async def test_mark_action_log_status_merges_details_without_overwrite():
    """_mark_action_log_status usa merge de JSON para preservar details existentes."""
    mock_session = AsyncMock()
    mock_session.execute.return_value = MagicMock()
    mock_session.commit = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    with patch(f"{_TOOLS}.async_session_maker", return_value=mock_session):
        await _mark_action_log_status(
            idempotency_key="idem-2",
            status="executed",
            details={"phase": "executed"},
        )

    sql = str(mock_session.execute.call_args[0][0])
    assert "COALESCE(details::jsonb" in sql
    assert "|| (:details)::jsonb" in sql


@pytest.mark.asyncio
async def test_count_recent_actions_filters_to_success_statuses():
    """_count_recent_actions considera apenas status executado/sucesso."""
    mock_result = MagicMock()
    mock_result.scalar.return_value = 2

    mock_session = AsyncMock()
    mock_session.execute.return_value = mock_result
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    with patch(f"{_TOOLS}.async_session_maker", return_value=mock_session):
        count = await _count_recent_actions("c1", "budget_change", minutes=10)

    assert count == 2
    sql = str(mock_session.execute.call_args[0][0])
    assert "status IN ('executed', 'success')" in sql
