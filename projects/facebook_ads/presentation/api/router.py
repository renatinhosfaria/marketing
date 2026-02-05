"""Router principal do módulo Facebook Ads (Clean Architecture)."""

from fastapi import APIRouter

from projects.facebook_ads.presentation.api.routes import (
    health,
    oauth,
    config,
    sync,
    campaigns,
    insights,
)

# Router público (health checks)
public_router = APIRouter()
public_router.include_router(health.router, prefix="/health", tags=["Facebook Ads - Health"])

# Router autenticado
authenticated_router = APIRouter()
authenticated_router.include_router(config.router, prefix="/config", tags=["Facebook Ads - Config"])
authenticated_router.include_router(sync.router, prefix="/sync", tags=["Facebook Ads - Sync"])
authenticated_router.include_router(campaigns.router, tags=["Facebook Ads - Campaigns"])
authenticated_router.include_router(insights.router, prefix="/insights", tags=["Facebook Ads - Insights"])

# Router principal que combina público + autenticado
facebook_ads_router = APIRouter()
public_router.include_router(oauth.router, prefix="/oauth", tags=["Facebook Ads - OAuth"])
facebook_ads_router.include_router(public_router)
facebook_ads_router.include_router(authenticated_router)
