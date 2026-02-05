"""
Jobs Celery para sincronização periódica do Facebook Ads.
"""

import asyncio
from typing import Optional

from app.celery import celery_app
from shared.db.session import isolated_async_session
from shared.core.logging import get_logger

logger = get_logger(__name__)


@celery_app.task(
    bind=True,
    name="projects.facebook_ads.jobs.sync_job.facebook_ads_sync_incremental",
    max_retries=2,
    default_retry_delay=300,
    acks_late=True,
)
def facebook_ads_sync_incremental(self):
    """
    Sync incremental (insights de hoje) para todas as contas ativas.
    Executado a cada hora.
    """
    async def _run():
        from sqlalchemy import select
        from shared.db.models.famachat_readonly import SistemaFacebookAdsConfig
        from projects.facebook_ads.services.sync_service import SyncService

        async with isolated_async_session() as session:
            # Buscar contas ativas com sync habilitado
            result = await session.execute(
                select(SistemaFacebookAdsConfig).where(
                    SistemaFacebookAdsConfig.is_active == True,
                    SistemaFacebookAdsConfig.sync_enabled == True,
                )
            )
            configs = result.scalars().all()

            logger.info("Sync incremental iniciado", total_configs=len(configs))

            for config in configs:
                try:
                    service = SyncService(session)
                    sync_history = await service.start_sync(config.id, "incremental")
                    await service.execute_sync(sync_history.id, "incremental")
                    await session.commit()
                    logger.info("Sync incremental concluído", config_id=config.id)
                except Exception as e:
                    await session.rollback()
                    logger.error(
                        "Erro no sync incremental",
                        config_id=config.id,
                        error=str(e),
                    )

            logger.info("Sync incremental finalizado para todas as contas")

    try:
        asyncio.run(_run())
    except Exception as e:
        logger.error("Erro fatal no sync incremental", error=str(e))
        raise self.retry(exc=e)


@celery_app.task(
    bind=True,
    name="projects.facebook_ads.jobs.sync_job.facebook_ads_sync_full",
    max_retries=1,
    default_retry_delay=600,
    acks_late=True,
)
def facebook_ads_sync_full(self):
    """
    Sync completo (campanhas + adsets + ads + insights) para todas as contas.
    Executado diariamente às 02:00 UTC.
    """
    async def _run():
        from sqlalchemy import select
        from shared.db.models.famachat_readonly import SistemaFacebookAdsConfig
        from projects.facebook_ads.services.sync_service import SyncService

        async with isolated_async_session() as session:
            result = await session.execute(
                select(SistemaFacebookAdsConfig).where(
                    SistemaFacebookAdsConfig.is_active == True,
                    SistemaFacebookAdsConfig.sync_enabled == True,
                )
            )
            configs = result.scalars().all()

            logger.info("Sync completo iniciado", total_configs=len(configs))

            for config in configs:
                try:
                    service = SyncService(session)
                    sync_history = await service.start_sync(config.id, "full")
                    await service.execute_sync(sync_history.id, "full")
                    await session.commit()
                    logger.info("Sync completo concluído", config_id=config.id)
                except Exception as e:
                    await session.rollback()
                    logger.error(
                        "Erro no sync completo",
                        config_id=config.id,
                        error=str(e),
                    )

            logger.info("Sync completo finalizado para todas as contas")

    try:
        asyncio.run(_run())
    except Exception as e:
        logger.error("Erro fatal no sync completo", error=str(e))
        raise self.retry(exc=e)
