"""
Modelos SQLAlchemy READ-ONLY para tabelas do FamaChat.
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

from shared.db.session import Base


class SistemaUsers(Base):
    """
    Usuários do sistema FamaChat.
    READ-ONLY - gerenciada pelo FamaChat Node.js.
    Usada para autenticação e consulta de dados do usuário.
    """
    __tablename__ = "sistema_users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    password_hash: Mapped[str] = mapped_column(Text, nullable=False)
    full_name: Mapped[str] = mapped_column(Text, nullable=False)
    email: Mapped[Optional[str]] = mapped_column(Text)
    phone: Mapped[Optional[str]] = mapped_column(Text)
    role: Mapped[str] = mapped_column(Text, nullable=False)
    department: Mapped[str] = mapped_column(Text, nullable=False)
    is_active: Mapped[Optional[bool]] = mapped_column(Boolean, default=True)
    whatsapp_instance: Mapped[Optional[str]] = mapped_column(Text)
    whatsapp_connected: Mapped[Optional[bool]] = mapped_column(Boolean, default=False)
    token_version: Mapped[Optional[int]] = mapped_column(Integer, default=0)
    last_login_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    failed_login_attempts: Mapped[Optional[int]] = mapped_column(Integer, default=0)
    locked_until: Mapped[Optional[datetime]] = mapped_column(DateTime)


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

    # Action values raw
    action_values: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON)

    # Quality diagnostics (available with 500+ impressions)
    quality_ranking: Mapped[Optional[str]] = mapped_column(String(50))
    engagement_rate_ranking: Mapped[Optional[str]] = mapped_column(String(50))
    conversion_rate_ranking: Mapped[Optional[str]] = mapped_column(String(50))

    # ROAS
    purchase_roas: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 4))
    website_purchase_roas: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 4))

    # Granular costs
    cpp: Mapped[Optional[Decimal]] = mapped_column(Numeric(15, 4))
    cost_per_unique_click: Mapped[Optional[Decimal]] = mapped_column(Numeric(15, 4))
    cost_per_inline_link_click: Mapped[Optional[Decimal]] = mapped_column(Numeric(15, 4))
    cost_per_outbound_click: Mapped[Optional[Decimal]] = mapped_column(Numeric(15, 4))
    cost_per_thruplay: Mapped[Optional[Decimal]] = mapped_column(Numeric(15, 4))

    # Specific CTRs
    unique_ctr: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 4))
    inline_link_click_ctr: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 4))
    outbound_clicks_ctr: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 4))

    # Video funnel
    video_plays: Mapped[Optional[int]] = mapped_column(Integer)
    video_15s_watched: Mapped[Optional[int]] = mapped_column(Integer)
    video_p25_watched: Mapped[Optional[int]] = mapped_column(Integer)
    video_p50_watched: Mapped[Optional[int]] = mapped_column(Integer)
    video_p75_watched: Mapped[Optional[int]] = mapped_column(Integer)
    video_p95_watched: Mapped[Optional[int]] = mapped_column(Integer)
    video_thruplay: Mapped[Optional[int]] = mapped_column(Integer)
    video_avg_time: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 2))

    # Unique metrics
    unique_inline_link_clicks: Mapped[Optional[int]] = mapped_column(Integer)
    unique_outbound_clicks: Mapped[Optional[int]] = mapped_column(Integer)
    unique_conversions: Mapped[Optional[int]] = mapped_column(Integer)

    # Unique costs
    cost_per_unique_conversion: Mapped[Optional[Decimal]] = mapped_column(Numeric(15, 4))
    cost_per_unique_outbound_click: Mapped[Optional[Decimal]] = mapped_column(Numeric(15, 4))
    cost_per_inline_post_engagement: Mapped[Optional[Decimal]] = mapped_column(Numeric(15, 4))

    # Unique CTRs
    unique_link_clicks_ctr: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 4))
    unique_inline_link_click_ctr: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 4))
    unique_outbound_clicks_ctr: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 4))

    # Brand awareness
    estimated_ad_recallers: Mapped[Optional[int]] = mapped_column(Integer)
    estimated_ad_recall_rate: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 4))
    cost_per_estimated_ad_recallers: Mapped[Optional[Decimal]] = mapped_column(Numeric(15, 4))

    # Landing page
    landing_page_views: Mapped[Optional[int]] = mapped_column(Integer)

    # Catalog
    converted_product_quantity: Mapped[Optional[int]] = mapped_column(Integer)
    converted_product_value: Mapped[Optional[Decimal]] = mapped_column(Numeric(15, 2))

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

    # Action values raw
    action_values: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON)

    # Quality diagnostics (available with 500+ impressions)
    quality_ranking: Mapped[Optional[str]] = mapped_column(String(50))
    engagement_rate_ranking: Mapped[Optional[str]] = mapped_column(String(50))
    conversion_rate_ranking: Mapped[Optional[str]] = mapped_column(String(50))

    # ROAS
    purchase_roas: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 4))
    website_purchase_roas: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 4))

    # Granular costs
    cpp: Mapped[Optional[Decimal]] = mapped_column(Numeric(15, 4))
    cost_per_unique_click: Mapped[Optional[Decimal]] = mapped_column(Numeric(15, 4))
    cost_per_inline_link_click: Mapped[Optional[Decimal]] = mapped_column(Numeric(15, 4))
    cost_per_outbound_click: Mapped[Optional[Decimal]] = mapped_column(Numeric(15, 4))
    cost_per_thruplay: Mapped[Optional[Decimal]] = mapped_column(Numeric(15, 4))

    # Specific CTRs
    unique_ctr: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 4))
    inline_link_click_ctr: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 4))
    outbound_clicks_ctr: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 4))

    # Video funnel
    video_plays: Mapped[Optional[int]] = mapped_column(Integer)
    video_15s_watched: Mapped[Optional[int]] = mapped_column(Integer)
    video_p25_watched: Mapped[Optional[int]] = mapped_column(Integer)
    video_p50_watched: Mapped[Optional[int]] = mapped_column(Integer)
    video_p75_watched: Mapped[Optional[int]] = mapped_column(Integer)
    video_p95_watched: Mapped[Optional[int]] = mapped_column(Integer)
    video_thruplay: Mapped[Optional[int]] = mapped_column(Integer)
    video_avg_time: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 2))

    # Unique metrics
    unique_inline_link_clicks: Mapped[Optional[int]] = mapped_column(Integer)
    unique_outbound_clicks: Mapped[Optional[int]] = mapped_column(Integer)
    unique_conversions: Mapped[Optional[int]] = mapped_column(Integer)

    # Unique costs
    cost_per_unique_conversion: Mapped[Optional[Decimal]] = mapped_column(Numeric(15, 4))
    cost_per_unique_outbound_click: Mapped[Optional[Decimal]] = mapped_column(Numeric(15, 4))
    cost_per_inline_post_engagement: Mapped[Optional[Decimal]] = mapped_column(Numeric(15, 4))

    # Unique CTRs
    unique_link_clicks_ctr: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 4))
    unique_inline_link_click_ctr: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 4))
    unique_outbound_clicks_ctr: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 4))

    # Brand awareness
    estimated_ad_recallers: Mapped[Optional[int]] = mapped_column(Integer)
    estimated_ad_recall_rate: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 4))
    cost_per_estimated_ad_recallers: Mapped[Optional[Decimal]] = mapped_column(Numeric(15, 4))

    # Landing page
    landing_page_views: Mapped[Optional[int]] = mapped_column(Integer)

    # Catalog
    converted_product_quantity: Mapped[Optional[int]] = mapped_column(Integer)
    converted_product_value: Mapped[Optional[Decimal]] = mapped_column(Numeric(15, 2))

    synced_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class SistemaFacebookAdsInsightsBreakdowns(Base):
    """Insights com breakdowns avancados (platform_position, region, etc.)."""
    __tablename__ = "sistema_facebook_ads_insights_breakdowns"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    config_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("sistema_facebook_ads_config.id", ondelete="CASCADE")
    )
    ad_id: Mapped[str] = mapped_column(String, nullable=False)
    adset_id: Mapped[str] = mapped_column(String, nullable=False)
    campaign_id: Mapped[str] = mapped_column(String, nullable=False)
    date: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    breakdown_type: Mapped[str] = mapped_column(String(100), nullable=False)
    breakdown_value: Mapped[str] = mapped_column(String(255), nullable=False)
    impressions: Mapped[int] = mapped_column(Integer, default=0)
    reach: Mapped[int] = mapped_column(Integer, default=0)
    clicks: Mapped[int] = mapped_column(Integer, default=0)
    spend: Mapped[Decimal] = mapped_column(Numeric(15, 2), default=0)
    leads: Mapped[int] = mapped_column(Integer, default=0)
    conversions: Mapped[int] = mapped_column(Integer, default=0)
    conversion_values: Mapped[Optional[Decimal]] = mapped_column(Numeric(15, 2))
    ctr: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 4))
    cpc: Mapped[Optional[Decimal]] = mapped_column(Numeric(15, 4))
    cpl: Mapped[Optional[Decimal]] = mapped_column(Numeric(15, 4))
    actions: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON)
    synced_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


# Aliases para compatibilidade com codigo legado
FacebookAdsConfig = SistemaFacebookAdsConfig
FacebookAdsCampaign = SistemaFacebookAdsCampaigns
FacebookAdsAdset = SistemaFacebookAdsAdsets
FacebookAdsAd = SistemaFacebookAdsAds
FacebookAdsInsight = SistemaFacebookAdsInsightsHistory
FacebookAdsInsightToday = SistemaFacebookAdsInsightsToday
FacebookAdsInsightBreakdown = SistemaFacebookAdsInsightsBreakdowns
