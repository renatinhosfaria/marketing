"""Schemas Pydantic para campanhas, adsets e ads."""

from datetime import datetime
from decimal import Decimal
from typing import Any, Optional

from projects.facebook_ads.schemas.base import CamelCaseModel


class CampaignResponse(CamelCaseModel):
    """Resposta de campanha."""
    id: int
    config_id: int
    campaign_id: str
    name: str
    objective: Optional[str] = None
    status: str
    effective_status: Optional[str] = None
    daily_budget: Optional[Decimal] = None
    lifetime_budget: Optional[Decimal] = None
    budget_remaining: Optional[Decimal] = None
    start_time: Optional[datetime] = None
    stop_time: Optional[datetime] = None
    created_time: Optional[datetime] = None
    updated_time: Optional[datetime] = None
    synced_at: datetime


class AdSetResponse(CamelCaseModel):
    """Resposta de ad set."""
    id: int
    config_id: int
    campaign_id: str
    adset_id: str
    name: str
    status: str
    effective_status: Optional[str] = None
    daily_budget: Optional[Decimal] = None
    lifetime_budget: Optional[Decimal] = None
    budget_remaining: Optional[Decimal] = None
    bid_amount: Optional[Decimal] = None
    bid_strategy: Optional[str] = None
    optimization_goal: Optional[str] = None
    billing_event: Optional[str] = None
    targeting: Optional[dict[str, Any]] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    synced_at: datetime


class AdResponse(CamelCaseModel):
    """Resposta de anúncio."""
    id: int
    config_id: int
    campaign_id: str
    adset_id: str
    ad_id: str
    name: str
    status: str
    effective_status: Optional[str] = None
    creative_id: Optional[str] = None
    preview_shareable_link: Optional[str] = None
    tracking_specs: Optional[dict[str, Any] | list[dict[str, Any]]] = None
    synced_at: datetime


class CampaignListResponse(CamelCaseModel):
    """Lista de campanhas com paginação."""
    success: bool = True
    data: list[CampaignResponse]
    total: int

class AdSetListResponse(CamelCaseModel):
    """Lista de ad sets com paginação."""
    success: bool = True
    data: list[AdSetResponse]
    total: int

class AdListResponse(CamelCaseModel):
    """Lista de anúncios com paginação."""
    success: bool = True
    data: list[AdResponse]
    total: int
