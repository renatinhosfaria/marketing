"""Schemas Pydantic para insights do Facebook Ads."""

from decimal import Decimal
from typing import Any, Optional
from pydantic import Field

from projects.facebook_ads.schemas.base import CamelCaseModel


class KPIResponse(CamelCaseModel):
    """KPIs agregados."""
    spend: Decimal = Decimal("0")
    impressions: int = 0
    reach: int = 0
    clicks: int = 0
    leads: int = 0
    conversions: int = 0
    ctr: Optional[Decimal] = None
    cpc: Optional[Decimal] = None
    cpm: Optional[Decimal] = None
    cpl: Optional[Decimal] = None
    frequency: Optional[Decimal] = None
    cpp: Optional[Decimal] = None
    cost_per_unique_click: Optional[Decimal] = None
    cost_per_thruplay: Optional[Decimal] = None
    unique_ctr: Optional[Decimal] = None
    inline_link_click_ctr: Optional[Decimal] = None
    purchase_roas: Optional[Decimal] = None
    period: str = "last_30d"


class DailyInsightResponse(CamelCaseModel):
    """Insight diário."""
    date: str
    spend: Decimal = Decimal("0")
    impressions: int = 0
    reach: int = 0
    clicks: int = 0
    leads: int = 0
    conversions: int = 0
    ctr: Optional[Decimal] = None
    cpc: Optional[Decimal] = None
    cpl: Optional[Decimal] = None
    unique_clicks: int = 0
    unique_ctr: Optional[Decimal] = None
    video_plays: Optional[int] = None
    video_thruplay: Optional[int] = None


class CampaignInsightResponse(CamelCaseModel):
    """Insight por campanha."""
    campaign_id: str
    campaign_name: str
    objective: Optional[str] = None
    status: Optional[str] = None
    spend: Decimal = Decimal("0")
    impressions: int = 0
    reach: int = 0
    clicks: int = 0
    leads: int = 0
    conversions: int = 0
    ctr: Optional[Decimal] = None
    cpc: Optional[Decimal] = None
    cpl: Optional[Decimal] = None
    cpp: Optional[Decimal] = None
    unique_ctr: Optional[Decimal] = None
    cost_per_unique_click: Optional[Decimal] = None


class BreakdownInsightResponse(CamelCaseModel):
    """Insight com breakdown."""
    breakdown_key: str
    breakdown_value: str
    spend: Decimal = Decimal("0")
    impressions: int = 0
    clicks: int = 0
    leads: int = 0
    ctr: Optional[Decimal] = None
    cpl: Optional[Decimal] = None


class CompareResponse(CamelCaseModel):
    """Comparação entre dois períodos."""
    current: KPIResponse
    previous: KPIResponse
    changes: dict[str, Optional[float]] = Field(default_factory=dict)


class BreakdownDetailResponse(CamelCaseModel):
    """Insight detalhado com breakdown avançado."""
    breakdown_type: str
    breakdown_value: str
    campaign_id: Optional[str] = None
    campaign_name: Optional[str] = None
    spend: Decimal = Decimal("0")
    impressions: int = 0
    reach: int = 0
    clicks: int = 0
    leads: int = 0
    conversions: int = 0
    ctr: Optional[Decimal] = None
    cpc: Optional[Decimal] = None
    cpl: Optional[Decimal] = None


class VideoFunnelResponse(CamelCaseModel):
    """Funil completo de vídeo."""
    video_plays: int = 0
    video_15s_watched: int = 0
    video_p25_watched: int = 0
    video_p50_watched: int = 0
    video_p75_watched: int = 0
    video_p95_watched: int = 0
    video_30s_watched: int = 0
    video_p100_watched: int = 0
    video_thruplay: int = 0
    video_avg_time: Optional[Decimal] = None


class QualityDiagnosticsResponse(CamelCaseModel):
    """Diagnósticos de qualidade de anúncios."""
    ad_id: str
    ad_name: Optional[str] = None
    quality_ranking: Optional[str] = None
    engagement_rate_ranking: Optional[str] = None
    conversion_rate_ranking: Optional[str] = None
    impressions: int = 0
    spend: Decimal = Decimal("0")


class InsightsListResponse(CamelCaseModel):
    """Resposta padrão com insights."""
    success: bool = True
    data: list[dict[str, Any]]
    meta: dict[str, Any] = Field(default_factory=dict)
