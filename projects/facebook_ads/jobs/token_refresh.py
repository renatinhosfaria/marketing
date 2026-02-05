"""
Job Celery para renovação automática de tokens do Facebook.
Executado diariamente às 06:00 UTC.
"""

import asyncio

from app.celery import celery_app
from shared.db.session import isolated_async_session
from shared.core.logging import get_logger
from projects.facebook_ads.config import fb_settings

logger = get_logger(__name__)


@celery_app.task(
    bind=True,
    name="projects.facebook_ads.jobs.token_refresh.facebook_ads_token_refresh",
    max_retries=2,
    default_retry_delay=1800,
    acks_late=True,
)
def facebook_ads_token_refresh(self):
    """
    Verifica e renova tokens que expiram nos próximos N dias.
    Configurado via FACEBOOK_TOKEN_REFRESH_DAYS_BEFORE (default: 14).
    """
    async def _run():
        from datetime import datetime, timedelta
        from sqlalchemy import select, and_
        from shared.db.models.famachat_readonly import SistemaFacebookAdsConfig
        from projects.facebook_ads.services.oauth_service import OAuthService

        threshold = datetime.utcnow() + timedelta(days=fb_settings.facebook_token_refresh_days_before)

        async with isolated_async_session() as session:
            result = await session.execute(
                select(SistemaFacebookAdsConfig).where(
                    and_(
                        SistemaFacebookAdsConfig.is_active == True,
                        SistemaFacebookAdsConfig.token_expires_at != None,
                        SistemaFacebookAdsConfig.token_expires_at <= threshold,
                    )
                )
            )
            configs = result.scalars().all()

            if not configs:
                logger.info("Nenhum token precisa de renovação")
                return

            logger.info("Tokens a renovar", count=len(configs))

            for config in configs:
                try:
                    service = OAuthService(session)
                    result = await service.refresh_token(config.id)
                    await session.commit()

                    logger.info(
                        "Token renovado automaticamente",
                        config_id=config.id,
                        new_expires_at=str(result.get("expires_at")),
                    )
                except Exception as e:
                    await session.rollback()
                    logger.error(
                        "Falha ao renovar token",
                        config_id=config.id,
                        error=str(e),
                    )

            logger.info("Renovação de tokens finalizada")

    try:
        asyncio.run(_run())
    except Exception as e:
        logger.error("Erro fatal na renovação de tokens", error=str(e))
        raise self.retry(exc=e)
