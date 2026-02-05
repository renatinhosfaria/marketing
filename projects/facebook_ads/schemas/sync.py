"""Schemas Pydantic para sincronização do Facebook Ads."""

from datetime import datetime
from typing import Optional
from pydantic import Field

from projects.facebook_ads.schemas.base import CamelCaseModel


class SyncStartRequest(CamelCaseModel):
    """Request para iniciar sincronização."""
    sync_type: str = Field("full", description="Tipo: full, incremental, today_only, historical")
    days_back: Optional[int] = Field(None, description="Dias para backfill (se historical)")
    date_range_start: Optional[datetime] = Field(None, description="Data inicial (se aplicável)")
    date_range_end: Optional[datetime] = Field(None, description="Data final (se aplicável)")


class SyncStartResponse(CamelCaseModel):
    """Resposta ao iniciar sync."""
    success: bool = True
    sync_id: int
    status: str = "pending"
    message: str = "Sincronização iniciada"


class SyncStatusResponse(CamelCaseModel):
    """Status de uma sincronização."""
    id: int
    config_id: int
    sync_type: str
    status: str
    entities_synced: Optional[int] = 0
    campaigns_synced: Optional[int] = 0
    adsets_synced: Optional[int] = 0
    ads_synced: Optional[int] = 0
    insights_synced: Optional[int] = 0
    date_range_start: Optional[datetime] = None
    date_range_end: Optional[datetime] = None
    error_message: Optional[str] = None
    error_details: Optional[dict] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_ms: Optional[int] = None


class SyncProgressResponse(CamelCaseModel):
    """Progresso de sincronização em tempo real."""
    stage: str
    campaigns_synced: int = 0
    adsets_synced: int = 0
    ads_synced: int = 0
    insights_synced: int = 0


class SyncHistoryResponse(CamelCaseModel):
    """Lista de sincronizações."""
    success: bool = True
    data: list[SyncStatusResponse]
    total: int


class SyncSummaryResponse(CamelCaseModel):
    """Resumo de status de sincronização para a UI."""
    config_id: int
    account_name: Optional[str] = None
    is_running: bool
    progress: Optional[SyncProgressResponse] = None
    last_sync: Optional[SyncStatusResponse] = None
    last_sync_at: Optional[datetime] = None
    next_sync_at: Optional[datetime] = None
    sync_enabled: bool
    sync_frequency_minutes: int
