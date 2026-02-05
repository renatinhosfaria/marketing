"""
Router da API v1 exclusivo para o modulo Facebook Ads.
Roda em processo FastAPI separado (porta 8002).
"""

from fastapi import APIRouter

from projects.facebook_ads.api.router import facebook_ads_router

# Router principal
fb_ads_api_router = APIRouter()


# Facebook Ads (ja inclui health publico + rotas autenticadas)
fb_ads_api_router.include_router(
    facebook_ads_router,
    prefix="/facebook-ads",
)
