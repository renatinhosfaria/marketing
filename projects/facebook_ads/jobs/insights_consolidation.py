"""
Job Celery para consolidação de insights (today → history).
Executado diariamente às 00:05 UTC.
"""

import asyncio

from app.celery import celery_app
from shared.db.session import isolated_async_session
from shared.core.logging import get_logger

logger = get_logger(__name__)


@celery_app.task(
    bind=True,
    name="projects.facebook_ads.jobs.insights_consolidation.facebook_ads_consolidation",
    max_retries=3,
    default_retry_delay=120,
    acks_late=True,
)
def facebook_ads_consolidation(self):
    """
    Consolida insights de ontem: busca dados finais da API e move para history.
    Limpa tabela today após consolidação.
    """
    async def _run():
        from sqlalchemy import select
        from shared.db.models.famachat_readonly import SistemaFacebookAdsConfig
        from projects.facebook_ads.services.sync_insights import SyncInsightsService

        async with isolated_async_session() as session:
            result = await session.execute(
                select(SistemaFacebookAdsConfig).where(
                    SistemaFacebookAdsConfig.is_active == True,
                    SistemaFacebookAdsConfig.sync_enabled == True,
                )
            )
            configs = result.scalars().all()

            logger.info("Consolidação de insights iniciada", total_configs=len(configs))

            for config in configs:
                try:
                    service = SyncInsightsService(session)
                    result = await service.sync_yesterday(config)
                    await session.commit()
                    logger.info(
                        "Consolidação concluída",
                        config_id=config.id,
                        **result,
                    )
                except Exception as e:
                    await session.rollback()
                    logger.error(
                        "Erro na consolidação",
                        config_id=config.id,
                        error=str(e),
                    )

            logger.info("Consolidação finalizada para todas as contas")

    try:
        asyncio.run(_run())
    except Exception as e:
        logger.error("Erro fatal na consolidação", error=str(e))
        raise self.retry(exc=e)
