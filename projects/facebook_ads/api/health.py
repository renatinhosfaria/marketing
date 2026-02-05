"""Endpoints de health check do módulo Facebook Ads."""

from fastapi import APIRouter, Depends
from sqlalchemy import select, func, text
from sqlalchemy.ext.asyncio import AsyncSession

from shared.db.session import get_db
from shared.db.models.famachat_readonly import SistemaFacebookAdsConfig
from projects.facebook_ads.config import fb_settings
from projects.facebook_ads.utils.rate_limiter import rate_limiter
from shared.core.logging import get_logger

logger = get_logger(__name__)

MODULE_VERSION = "2.0.0"

router = APIRouter()


@router.get("/simple")
async def health_simple():
    """Health check simples para load balancers (sem auth)."""
    return {"status": "ok", "module": "facebook-ads"}


@router.get("")
async def health_detailed(db: AsyncSession = Depends(get_db)):
    """Health check detalhado com status do módulo."""
    # Verificar conexão com banco
    db_ok = False
    try:
        await db.execute(text("SELECT 1"))
        db_ok = True
    except Exception as e:
        logger.error("Health check: erro ao conectar ao banco", error=str(e))

    # Contar configs ativas
    active_configs = 0
    if db_ok:
        try:
            result = await db.execute(
                select(func.count(SistemaFacebookAdsConfig.id)).where(
                    SistemaFacebookAdsConfig.is_active == True  # noqa: E712
                )
            )
            active_configs = result.scalar_one()
        except Exception:
            pass

    # Verificar configuração
    config_ok = bool(fb_settings.facebook_app_id and fb_settings.facebook_app_secret)

    # Rate limit status
    rate_limit_usage = rate_limiter.get_usage()
    rate_limit_status = "ok"
    if rate_limit_usage >= fb_settings.facebook_rate_limit_pause_threshold:
        rate_limit_status = "critical"
    elif rate_limit_usage >= fb_settings.facebook_rate_limit_threshold:
        rate_limit_status = "warning"

    overall_status = "ok"
    if not db_ok:
        overall_status = "unhealthy"
    elif not config_ok:
        overall_status = "degraded"

    return {
        "status": overall_status,
        "module": "facebook-ads",
        "version": MODULE_VERSION,
        "api_version": fb_settings.facebook_api_version,
        "database": {"connected": db_ok},
        "active_configs": active_configs,
        "config_present": config_ok,
        "rate_limit": {
            "status": rate_limit_status,
            "usage_pct": rate_limit_usage,
            "threshold": fb_settings.facebook_rate_limit_threshold,
            "pause_threshold": fb_settings.facebook_rate_limit_pause_threshold,
        },
        "features": {
            "oauth": config_ok,
            "sync": config_ok,
            "insights": config_ok,
            "async_reports": True,
            "breakdowns": True,
            "app_secret_proof": bool(fb_settings.facebook_app_secret),
        },
    }
