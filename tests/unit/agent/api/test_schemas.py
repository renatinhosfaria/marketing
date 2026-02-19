"""
Testes dos schemas Pydantic da API do Agent.

Testa:
  - ChatRequest: validacao de campos obrigatorios e opcionais
  - ResumePayload: validacao de aprovacao/rejeicao
  - Serializacao/desserializacao
  - Campos default
"""

import pytest
from pydantic import ValidationError

from projects.agent.api.schemas import ChatRequest, ResumePayload


@pytest.mark.asyncio
async def test_chat_request_with_message():
    """ChatRequest com message e valido."""
    req = ChatRequest(
        message="Como esta meu CPL?",
        thread_id="abc-123",
        account_id="act_987",
    )

    assert req.message == "Como esta meu CPL?"
    assert req.thread_id == "abc-123"
    assert req.account_id == "act_987"
    assert req.resume_payload is None


@pytest.mark.asyncio
async def test_chat_request_with_resume():
    """ChatRequest com resume_payload e valido."""
    req = ChatRequest(
        thread_id="abc-123",
        account_id="act_987",
        resume_payload=ResumePayload(
            approved=True,
            approval_token="token123",
        ),
    )

    assert req.message is None
    assert req.resume_payload is not None
    assert req.resume_payload.approved is True
    assert req.resume_payload.approval_token == "token123"


@pytest.mark.asyncio
async def test_chat_request_accepts_numeric_account_id():
    """ChatRequest aceita account_id apenas numerico."""
    req = ChatRequest(
        message="Como esta meu CPL?",
        thread_id="abc-123",
        account_id="123456",
    )

    assert req.account_id == "123456"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "invalid_account_id",
    ["abc", "act_12_34", "act_%", "12 34", '12"34'],
)
async def test_chat_request_rejects_invalid_account_id_pattern(invalid_account_id: str):
    """ChatRequest rejeita account_id fora do padrao permitido."""
    with pytest.raises(ValidationError):
        ChatRequest(
            message="Oi",
            thread_id="abc-123",
            account_id=invalid_account_id,
        )


@pytest.mark.asyncio
async def test_chat_request_missing_thread_id():
    """ChatRequest sem thread_id deve falhar."""
    with pytest.raises(ValidationError):
        ChatRequest(
            message="Oi",
            account_id="act_987",
        )


@pytest.mark.asyncio
async def test_chat_request_missing_account_id():
    """ChatRequest sem account_id deve falhar."""
    with pytest.raises(ValidationError):
        ChatRequest(
            message="Oi",
            thread_id="abc-123",
        )


@pytest.mark.asyncio
async def test_resume_payload_approved():
    """ResumePayload com aprovacao."""
    payload = ResumePayload(
        approved=True,
        approval_token="abc123def456",
    )

    assert payload.approved is True
    assert payload.new_budget_override is None


@pytest.mark.asyncio
async def test_resume_payload_with_budget_override():
    """ResumePayload com edit on resume (budget override)."""
    payload = ResumePayload(
        approved=True,
        approval_token="abc123def456",
        new_budget_override=75.0,
    )

    assert payload.approved is True
    assert payload.new_budget_override == 75.0


@pytest.mark.asyncio
async def test_resume_payload_rejected():
    """ResumePayload com rejeicao."""
    payload = ResumePayload(
        approved=False,
        approval_token="abc123def456",
    )

    assert payload.approved is False


@pytest.mark.asyncio
async def test_resume_payload_model_dump():
    """ResumePayload.model_dump() retorna dict serializavel."""
    payload = ResumePayload(
        approved=True,
        approval_token="token",
        new_budget_override=60.0,
    )

    dump = payload.model_dump()
    assert isinstance(dump, dict)
    assert dump["approved"] is True
    assert dump["approval_token"] == "token"
    assert dump["new_budget_override"] == 60.0


@pytest.mark.asyncio
async def test_resume_payload_accepts_legacy_override_alias():
    """ResumePayload aceita override_value legado e normaliza para campo canonico."""
    payload = ResumePayload.model_validate(
        {
            "approved": True,
            "approval_token": "token",
            "override_value": 55.5,
        }
    )

    assert payload.new_budget_override == 55.5
    dump = payload.model_dump()
    assert dump["new_budget_override"] == 55.5
    assert "override_value" not in dump


@pytest.mark.asyncio
async def test_resume_payload_missing_token():
    """ResumePayload sem approval_token deve falhar."""
    with pytest.raises(ValidationError):
        ResumePayload(approved=True)
