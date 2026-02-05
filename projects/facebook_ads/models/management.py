"""Modelos para audit trail e rate limit logging do Facebook Ads."""

from datetime import datetime
from typing import Any, Optional

from sqlalchemy import Integer, String, Text, DateTime, JSON, Numeric
from sqlalchemy.orm import Mapped, mapped_column

from shared.db.session import Base


class MLFacebookAdsManagementLog(Base):
    """Audit trail de ações de gerenciamento de campanhas."""
    __tablename__ = "ml_facebook_ads_management_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    config_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    user_id: Mapped[int] = mapped_column(Integer, nullable=False)

    # Ação
    action: Mapped[str] = mapped_column(String(100), nullable=False)
    entity_type: Mapped[str] = mapped_column(String(50), nullable=False)  # campaign, adset, ad
    entity_id: Mapped[str] = mapped_column(String(100), nullable=False)

    # Dados
    before_state: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON)
    after_state: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON)
    request_data: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON)
    response_data: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON)

    # Status
    success: Mapped[bool] = mapped_column(default=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text)

    # Metadata
    source: Mapped[str] = mapped_column(String(50), default="manual")  # manual, ml_recommendation, scheduled
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class MLFacebookAdsRateLimitLog(Base):
    """Log de eventos de rate limit da Facebook API."""
    __tablename__ = "ml_facebook_ads_rate_limit_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    config_id: Mapped[Optional[int]] = mapped_column(Integer)
    account_id: Mapped[Optional[str]] = mapped_column(String(100))

    # Métricas
    call_count_pct: Mapped[Optional[float]] = mapped_column(Numeric(5, 2))
    total_cputime_pct: Mapped[Optional[float]] = mapped_column(Numeric(5, 2))
    total_time_pct: Mapped[Optional[float]] = mapped_column(Numeric(5, 2))
    max_usage_pct: Mapped[Optional[float]] = mapped_column(Numeric(5, 2))

    # Ação tomada
    action_taken: Mapped[str] = mapped_column(String(50), nullable=False)  # delay, pause, none
    wait_seconds: Mapped[Optional[float]] = mapped_column(Numeric(10, 2))

    # Metadata
    endpoint: Mapped[Optional[str]] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
