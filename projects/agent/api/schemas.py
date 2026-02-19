"""
Schemas Pydantic da API do Agent.

ChatRequest: payload do endpoint /chat.
ResumePayload: payload para retomar apos interrupt.
"""

from datetime import datetime

from pydantic import BaseModel, Field, AliasChoices, ConfigDict
from typing import Optional

ACCOUNT_ID_PATTERN = r"^(act_)?\d+$"


class ResumePayload(BaseModel):
    """Payload para retomar apos aprovacao (interrupt)."""
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    approved: bool = Field(description="Se o usuario aprovou a acao")
    approval_token: str = Field(description="Token anti-forgery")
    new_budget_override: Optional[float] = Field(
        default=None,
        validation_alias=AliasChoices("new_budget_override", "override_value"),
        description="Novo valor de budget editado pelo usuario (edit on resume)",
    )


class ChatRequest(BaseModel):
    """Payload do endpoint POST /chat."""
    message: Optional[str] = Field(
        default=None,
        description="Mensagem do usuario (obrigatorio se nao for resume)",
    )
    thread_id: str = Field(
        description="ID da conversa (UUID gerado pelo frontend)",
    )
    account_id: str = Field(
        pattern=ACCOUNT_ID_PATTERN,
        description="ID da conta de Facebook Ads",
    )
    resume_payload: Optional[ResumePayload] = Field(
        default=None,
        description="Payload de retomada apos interrupt (aprovacao/rejeicao)",
    )


class ConversationPreview(BaseModel):
    """Preview de uma conversa para listagem."""
    thread_id: str = Field(description="ID original da thread (sem prefixo tenant)")
    title: str = Field(description="Titulo gerado por IA")
    created_at: datetime = Field(description="Data de criacao")
    last_message_at: datetime = Field(description="Data da ultima mensagem")


class ConversationMessages(BaseModel):
    """Mensagens de uma conversa."""
    thread_id: str
    messages: list[dict] = Field(description="Lista de mensagens {role, content, timestamp}")
