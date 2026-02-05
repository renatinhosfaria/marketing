"""Schemas Pydantic para configuração de contas Facebook Ads."""

from datetime import datetime
from typing import Optional
from pydantic import Field

from projects.facebook_ads.schemas.base import CamelCaseModel


class ConfigResponse(CamelCaseModel):
    """Resposta com dados de configuração (sem tokens)."""
    id: int
    account_id: str
    account_name: str
    is_active: bool
    sync_enabled: bool
    sync_frequency_minutes: int
    last_sync_at: Optional[datetime] = None
    token_expires_at: Optional[datetime] = None
    token_status: str = "unknown"
    created_at: datetime
    updated_at: datetime


class ConfigCreateRequest(CamelCaseModel):
    """Request para criar configuração."""
    account_id: str = Field(..., description="ID da ad account (sem prefixo act_)")
    account_name: str = Field(..., description="Nome da conta")
    access_token: str = Field(..., description="Token de acesso (será criptografado)")
    app_id: Optional[str] = Field(None, description="App ID (usa padrão se vazio)")
    app_secret: Optional[str] = Field(None, description="App Secret (usa padrão se vazio)")
    sync_enabled: bool = True
    sync_frequency_minutes: int = 60


class ConfigUpdateRequest(CamelCaseModel):
    """Request para atualizar configuração."""
    account_name: Optional[str] = None
    is_active: Optional[bool] = None
    sync_enabled: Optional[bool] = None
    sync_frequency_minutes: Optional[int] = None


class ConfigTestResponse(CamelCaseModel):
    """Resposta de teste de conexão."""
    success: bool
    account_name: Optional[str] = None
    currency: Optional[str] = None
    timezone: Optional[str] = None
    error: Optional[str] = None


class AdAccountInfo(CamelCaseModel):
    """Info de uma ad account disponível."""
    id: str
    name: str
    account_id: str
    currency: str
    timezone_name: str
    account_status: int
