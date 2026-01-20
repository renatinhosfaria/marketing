"""
Modelos SQLAlchemy READ-ONLY para tabelas do Facebook Ads do FamaChat.
Estes modelos espelham as tabelas existentes e são usados apenas para leitura.
"""

from datetime import datetime
from decimal import Decimal
from typing import Any, Optional

from sqlalchemy import (
    Integer,
    String,
    Text,
    Boolean,
    Numeric,
    DateTime,
    JSON,
    ForeignKey,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.session import Base


class SistemaFacebookAdsConfig(Base):
    """
    Configuração de conta do Facebook Ads.
    READ-ONLY - gerenciada pelo FamaChat Node.js.
    """
    __tablename__ = "sistema_facebook_ads_config"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    account_id: Mapped[str] = mapped_column(Text, nullable=False)
    account_name: Mapped[str] = mapped_column(Text, nullable=False)
    access_token: Mapped[str] = mapped_column(Text, nullable=False)
    token_expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    app_id: Mapped[str] = mapped_column(Text, nullable=False)
    app_secret: Mapped[str] = mapped_column(Text, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    sync_enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    sync_frequency_minutes: Mapped[int] = mapped_column(Integer, default=60)
    last_sync_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    created_by: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Relacionamentos
    campaigns: Mapped[list["SistemaFacebookAdsCampaigns"]] = relationship(
        back_populates="config", lazy="selectin"
    )
    adsets: Mapped[list["SistemaFacebookAdsAdsets"]] = relationship(
        back_populates="config", lazy="selectin"
    )
    ads: Mapped[list["SistemaFacebookAdsAds"]] = relationship(
        back_populates="config", lazy="selectin"
    )

    # Alias para compatibilidade
    @property
    def name(self) -> str:
        return self.account_name

    @property
    def ad_account_id(self) -> str:
        return self.account_id


class SistemaFacebookAdsCampaigns(Base):
    """
    Campanhas do Facebook Ads.
    READ-ONLY - gerenciada pelo FamaChat Node.js.
    """
    __tablename__ = "sistema_facebook_ads_campaigns"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    config_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("sistema_facebook_ads_config.id", ondelete="CASCADE")
    )
    campaign_id: Mapped[str] = mapped_column(String, nullable=False)
    name: Mapped[str] = mapped_column(String, nullable=False)
    objective: Mapped[Optional[str]] = mapped_column(String)
    status: Mapped[str] = mapped_column(String, nullable=False)
    effective_status: Mapped[Optional[str]] = mapped_column(String)
    daily_budget: Mapped[Optional[Decimal]] = mapped_column(Numeric(15, 2))
    lifetime_budget: Mapped[Optional[Decimal]] = mapped_column(Numeric(15, 2))
    budget_remaining: Mapped[Optional[Decimal]] = mapped_column(Numeric(15, 2))
    start_time: Mapped[Optional[datetime]] = mapped_column(DateTime)
    stop_time: Mapped[Optional[datetime]] = mapped_column(DateTime)
    created_time: Mapped[Optional[datetime]] = mapped_column(DateTime)
    updated_time: Mapped[Optional[datetime]] = mapped_column(DateTime)
    synced_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Relacionamentos
    config: Mapped["SistemaFacebookAdsConfig"] = relationship(
        back_populates="campaigns"
    )


class SistemaFacebookAdsAdsets(Base):
    """
    Conjuntos de anúncios (Ad Sets) do Facebook Ads.
    READ-ONLY - gerenciada pelo FamaChat Node.js.
    """
    __tablename__ = "sistema_facebook_ads_adsets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    config_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("sistema_facebook_ads_config.id", ondelete="CASCADE")
    )
    campaign_id: Mapped[str] = mapped_column(String, nullable=False)
    adset_id: Mapped[str] = mapped_column(String, nullable=False)
    name: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False)
    effective_status: Mapped[Optional[str]] = mapped_column(String)
    daily_budget: Mapped[Optional[Decimal]] = mapped_column(Numeric(15, 2))
    lifetime_budget: Mapped[Optional[Decimal]] = mapped_column(Numeric(15, 2))
    budget_remaining: Mapped[Optional[Decimal]] = mapped_column(Numeric(15, 2))
    bid_amount: Mapped[Optional[Decimal]] = mapped_column(Numeric(15, 2))
    bid_strategy: Mapped[Optional[str]] = mapped_column(String)
    optimization_goal: Mapped[Optional[str]] = mapped_column(String)
    billing_event: Mapped[Optional[str]] = mapped_column(String)
    targeting: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON)
    start_time: Mapped[Optional[datetime]] = mapped_column(DateTime)
    end_time: Mapped[Optional[datetime]] = mapped_column(DateTime)
    created_time: Mapped[Optional[datetime]] = mapped_column(DateTime)
    updated_time: Mapped[Optional[datetime]] = mapped_column(DateTime)
    synced_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Relacionamentos
    config: Mapped["SistemaFacebookAdsConfig"] = relationship(
        back_populates="adsets"
    )


class SistemaFacebookAdsAds(Base):
    """
    Anúncios individuais do Facebook Ads.
    READ-ONLY - gerenciada pelo FamaChat Node.js.
    """
    __tablename__ = "sistema_facebook_ads_ads"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    config_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("sistema_facebook_ads_config.id", ondelete="CASCADE")
    )
    campaign_id: Mapped[str] = mapped_column(String, nullable=False)
    adset_id: Mapped[str] = mapped_column(String, nullable=False)
    ad_id: Mapped[str] = mapped_column(String, nullable=False)
    name: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False)
    effective_status: Mapped[Optional[str]] = mapped_column(String)
    creative_id: Mapped[Optional[str]] = mapped_column(String)
    preview_shareable_link: Mapped[Optional[str]] = mapped_column(Text)
    tracking_specs: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON)
    created_time: Mapped[Optional[datetime]] = mapped_column(DateTime)
    updated_time: Mapped[Optional[datetime]] = mapped_column(DateTime)
    synced_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Relacionamentos
    config: Mapped["SistemaFacebookAdsConfig"] = relationship(
        back_populates="ads"
    )


class SistemaFacebookAdsInsightsHistory(Base):
    """
    Insights históricos do Facebook Ads (um registro por anúncio por dia).
    READ-ONLY - gerenciada pelo FamaChat Node.js.
    """
    __tablename__ = "sistema_facebook_ads_insights_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    config_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("sistema_facebook_ads_config.id", ondelete="CASCADE")
    )

    # Identificação
    ad_id: Mapped[str] = mapped_column(String, nullable=False)
    adset_id: Mapped[str] = mapped_column(String, nullable=False)
    campaign_id: Mapped[str] = mapped_column(String, nullable=False)

    # Data do registro
    date: Mapped[datetime] = mapped_column(DateTime, nullable=False)

    # Métricas de alcance e impressões
    impressions: Mapped[int] = mapped_column(Integer, default=0)
    reach: Mapped[int] = mapped_column(Integer, default=0)
    frequency: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 4))

    # Métricas de cliques
    clicks: Mapped[int] = mapped_column(Integer, default=0)
    unique_clicks: Mapped[Optional[int]] = mapped_column(Integer, default=0)
    inline_link_clicks: Mapped[Optional[int]] = mapped_column(Integer, default=0)
    outbound_clicks: Mapped[Optional[int]] = mapped_column(Integer, default=0)

    # Métricas de custo
    spend: Mapped[Decimal] = mapped_column(Numeric(15, 2), default=0)
    cpc: Mapped[Optional[Decimal]] = mapped_column(Numeric(15, 4))
    cpm: Mapped[Optional[Decimal]] = mapped_column(Numeric(15, 4))

    # Métricas de taxa
    ctr: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 4))

    # Métricas de conversão e leads
    conversions: Mapped[int] = mapped_column(Integer, default=0)
    conversion_values: Mapped[Optional[Decimal]] = mapped_column(Numeric(15, 2))
    leads: Mapped[int] = mapped_column(Integer, default=0)
    cost_per_lead: Mapped[Optional[Decimal]] = mapped_column(Numeric(15, 4))

    # Métricas de vídeo
    video_views: Mapped[int] = mapped_column(Integer, default=0)
    video_p100_watched: Mapped[int] = mapped_column(Integer, default=0)

    # Engajamento
    post_engagement: Mapped[int] = mapped_column(Integer, default=0)
    post_reactions: Mapped[int] = mapped_column(Integer, default=0)
    post_comments: Mapped[int] = mapped_column(Integer, default=0)
    post_shares: Mapped[int] = mapped_column(Integer, default=0)

    # Ações detalhadas
    actions: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON)

    # Metadados
    consolidated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow
    )


class SistemaFacebookAdsInsightsToday(Base):
    """
    Insights do dia atual do Facebook Ads.
    READ-ONLY - gerenciada pelo FamaChat Node.js.
    """
    __tablename__ = "sistema_facebook_ads_insights_today"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    config_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("sistema_facebook_ads_config.id", ondelete="CASCADE")
    )

    # Identificação
    ad_id: Mapped[str] = mapped_column(String, nullable=False)
    adset_id: Mapped[str] = mapped_column(String, nullable=False)
    campaign_id: Mapped[str] = mapped_column(String, nullable=False)

    # Data
    date: Mapped[datetime] = mapped_column(DateTime, nullable=False)

    # Métricas (mesmas do histórico)
    impressions: Mapped[int] = mapped_column(Integer, default=0)
    reach: Mapped[int] = mapped_column(Integer, default=0)
    frequency: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 4))
    clicks: Mapped[int] = mapped_column(Integer, default=0)
    unique_clicks: Mapped[Optional[int]] = mapped_column(Integer, default=0)
    inline_link_clicks: Mapped[Optional[int]] = mapped_column(Integer, default=0)
    outbound_clicks: Mapped[Optional[int]] = mapped_column(Integer, default=0)
    spend: Mapped[Decimal] = mapped_column(Numeric(15, 2), default=0)
    cpc: Mapped[Optional[Decimal]] = mapped_column(Numeric(15, 4))
    cpm: Mapped[Optional[Decimal]] = mapped_column(Numeric(15, 4))
    ctr: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 4))
    conversions: Mapped[int] = mapped_column(Integer, default=0)
    conversion_values: Mapped[Optional[Decimal]] = mapped_column(Numeric(15, 2))
    leads: Mapped[int] = mapped_column(Integer, default=0)
    cost_per_lead: Mapped[Optional[Decimal]] = mapped_column(Numeric(15, 4))
    video_views: Mapped[int] = mapped_column(Integer, default=0)
    video_p100_watched: Mapped[int] = mapped_column(Integer, default=0)
    post_engagement: Mapped[int] = mapped_column(Integer, default=0)
    post_reactions: Mapped[int] = mapped_column(Integer, default=0)
    post_comments: Mapped[int] = mapped_column(Integer, default=0)
    post_shares: Mapped[int] = mapped_column(Integer, default=0)
    actions: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON)
    synced_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


# Aliases para compatibilidade com codigo legado
FacebookAdsConfig = SistemaFacebookAdsConfig
FacebookAdsCampaign = SistemaFacebookAdsCampaigns
FacebookAdsAdset = SistemaFacebookAdsAdsets
FacebookAdsAd = SistemaFacebookAdsAds
FacebookAdsInsight = SistemaFacebookAdsInsightsHistory
FacebookAdsInsightToday = SistemaFacebookAdsInsightsToday
