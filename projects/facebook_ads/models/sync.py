"""Modelo para histórico de sincronizações do Facebook Ads."""

import enum
from datetime import datetime
from typing import Any, Optional

from sqlalchemy import Integer, String, Text, DateTime, JSON, Enum
from sqlalchemy.orm import Mapped, mapped_column

from shared.db.session import Base


class SyncStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PARTIAL = "partial"


class SyncType(str, enum.Enum):
    FULL = "full"
    INCREMENTAL = "incremental"
    TODAY_ONLY = "today_only"
    HISTORICAL = "historical"
    CAMPAIGNS_ONLY = "campaigns_only"


class SistemaFacebookAdsSyncHistory(Base):
    """Histórico de sincronizações do Facebook Ads."""
    __tablename__ = "sistema_facebook_ads_sync_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    config_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    status: Mapped[str] = mapped_column(
        String(50), nullable=False, default=SyncStatus.PENDING.value
    )
    sync_type: Mapped[str] = mapped_column(String(50), nullable=False)

    # Contadores
    entities_synced: Mapped[Optional[int]] = mapped_column(Integer, default=0)
    campaigns_synced: Mapped[Optional[int]] = mapped_column(Integer, default=0)
    adsets_synced: Mapped[Optional[int]] = mapped_column(Integer, default=0)
    ads_synced: Mapped[Optional[int]] = mapped_column(Integer, default=0)
    insights_synced: Mapped[Optional[int]] = mapped_column(Integer, default=0)

    # Range de datas (quando aplicável)
    date_range_start: Mapped[Optional[datetime]] = mapped_column(DateTime)
    date_range_end: Mapped[Optional[datetime]] = mapped_column(DateTime)

    # Erros/detalhes
    error_message: Mapped[Optional[str]] = mapped_column(Text)
    error_details: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON)

    # Timestamps
    started_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    duration_ms: Mapped[Optional[int]] = mapped_column(Integer)
