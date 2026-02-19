"""
Serviço orquestrador de sincronização do Facebook Ads.
Gerencia o fluxo completo de sync com lock, checkpoint recovery e progress tracking.
"""

from datetime import datetime
from typing import Optional

from sqlalchemy import select, update, func, desc
from sqlalchemy.ext.asyncio import AsyncSession

from shared.core.logging import get_logger
from projects.facebook_ads.models.sync import SistemaFacebookAdsSyncHistory, SyncStatus, SyncType
from projects.facebook_ads.services.sync_campaigns import SyncCampaignsService
from projects.facebook_ads.services.sync_adsets_ads import SyncAdSetsAdsService
from projects.facebook_ads.services.sync_insights import SyncInsightsService
from shared.db.models.famachat_readonly import SistemaFacebookAdsConfig

logger = get_logger(__name__)


class SyncService:
    """Orquestra sincronização completa do Facebook Ads."""

    RECENT_DAYS_BACK = 3

    def __init__(self, db: AsyncSession):
        self.db = db
        self._campaigns_service = SyncCampaignsService(db)
        self._adsets_ads_service = SyncAdSetsAdsService(db)
        self._insights_service = SyncInsightsService(db)

    @staticmethod
    def _merge_insights_results(today_result: dict, recent_result: dict) -> dict:
        """Combina métricas de sync de hoje e de re-consolidação recente."""
        today_synced = int(today_result.get("synced", 0) or 0)
        recent_synced = int(recent_result.get("synced", 0) or 0)
        return {
            "synced": today_synced + recent_synced,
            "inserted": int(today_result.get("inserted", 0) or 0) + int(recent_result.get("inserted", 0) or 0),
            "updated": int(today_result.get("updated", 0) or 0) + int(recent_result.get("updated", 0) or 0),
            "errors": int(today_result.get("errors", 0) or 0) + int(recent_result.get("errors", 0) or 0),
            "today_synced": today_synced,
            "recent_synced": recent_synced,
        }

    async def get_config(self, config_id: int) -> SistemaFacebookAdsConfig:
        """Busca configuração e valida se está ativa."""
        result = await self.db.execute(
            select(SistemaFacebookAdsConfig).where(
                SistemaFacebookAdsConfig.id == config_id
            )
        )
        config = result.scalar_one_or_none()

        if not config:
            raise ValueError(f"Configuração {config_id} não encontrada")
        if not config.is_active:
            raise ValueError(f"Configuração {config_id} está inativa")

        return config

    async def start_sync(
        self,
        config_id: int,
        sync_type: str = "full",
        days_back: Optional[int] = None,
        date_range_start: Optional[datetime] = None,
        date_range_end: Optional[datetime] = None,
    ) -> SistemaFacebookAdsSyncHistory:
        """Cria registro de sync e inicia sincronização."""
        config = await self.get_config(config_id)

        # Verificar sync em andamento
        running = await self.db.execute(
            select(SistemaFacebookAdsSyncHistory).where(
                SistemaFacebookAdsSyncHistory.config_id == config_id,
                SistemaFacebookAdsSyncHistory.status == SyncStatus.RUNNING.value,
            )
        )
        if running.scalar_one_or_none():
            raise ValueError(f"Já existe uma sincronização em andamento para config {config_id}")

        # Criar registro
        sync_history = SistemaFacebookAdsSyncHistory(
            config_id=config_id,
            sync_type=sync_type,
            status=SyncStatus.PENDING.value,
            date_range_start=date_range_start,
            date_range_end=date_range_end,
        )
        self.db.add(sync_history)
        await self.db.flush()

        logger.info(
            "Sync iniciado",
            sync_id=sync_history.id,
            config_id=config_id,
            sync_type=sync_type,
        )

        return sync_history

    async def execute_sync(
        self,
        sync_id: int,
        sync_type: str = "full",
        days_back: Optional[int] = None,
    ) -> dict:
        """Executa a sincronização (chamado em background)."""
        # Buscar sync_history
        result = await self.db.execute(
            select(SistemaFacebookAdsSyncHistory).where(
                SistemaFacebookAdsSyncHistory.id == sync_id
            )
        )
        sync_history = result.scalar_one_or_none()

        if not sync_history:
            raise ValueError(f"Sync {sync_id} não encontrado")

        config = await self.get_config(sync_history.config_id)

        # Marcar como running
        sync_history.status = SyncStatus.RUNNING.value
        sync_history.started_at = datetime.utcnow()
        await self.db.flush()

        results = {
            "campaigns": {"synced": 0, "created": 0, "updated": 0, "errors": 0},
            "adsets": {"synced": 0, "created": 0, "updated": 0, "errors": 0},
            "ads": {"synced": 0, "created": 0, "updated": 0, "errors": 0},
            "insights": {"synced": 0, "inserted": 0, "updated": 0, "errors": 0},
        }

        try:
            if sync_type in ("full", "campaigns_only", "incremental"):
                # Step 1: Campanhas
                results["campaigns"] = await self._campaigns_service.sync(config)
                sync_history.campaigns_synced = results["campaigns"]["synced"]

                if sync_type != "campaigns_only":
                    # Step 2: Ad Sets
                    results["adsets"] = await self._adsets_ads_service.sync_adsets(config)
                    sync_history.adsets_synced = results["adsets"]["synced"]

                    # Step 3: Ads
                    results["ads"] = await self._adsets_ads_service.sync_ads(config)
                    sync_history.ads_synced = results["ads"]["synced"]

            if sync_type in ("full", "incremental"):
                # Step 4: Insights de hoje
                today_result = await self._insights_service.sync_today(config)
                recent_result = await self._insights_service.sync_recent_days(
                    config,
                    days_back=self.RECENT_DAYS_BACK,
                )
                results["insights"] = self._merge_insights_results(today_result, recent_result)
                sync_history.insights_synced = results["insights"].get("synced", 0)

            elif sync_type == "today_only":
                today_result = await self._insights_service.sync_today(config)
                recent_result = await self._insights_service.sync_recent_days(
                    config,
                    days_back=self.RECENT_DAYS_BACK,
                )
                results["insights"] = self._merge_insights_results(today_result, recent_result)
                sync_history.insights_synced = results["insights"].get("synced", 0)

            elif sync_type == "historical":
                results["insights"] = await self._insights_service.sync_historical(config, days_back)
                sync_history.insights_synced = results["insights"].get("synced", 0)

            # Completar
            sync_history.status = SyncStatus.COMPLETED.value
            sync_history.completed_at = datetime.utcnow()
            sync_history.entities_synced = sum(
                [
                    sync_history.campaigns_synced or 0,
                    sync_history.adsets_synced or 0,
                    sync_history.ads_synced or 0,
                    sync_history.insights_synced or 0,
                ]
            )
            sync_history.duration_ms = int(
                (sync_history.completed_at - sync_history.started_at).total_seconds() * 1000
            )

            # Atualizar last_sync_at na config
            await self.db.execute(
                update(SistemaFacebookAdsConfig)
                .where(SistemaFacebookAdsConfig.id == config.id)
                .values(last_sync_at=datetime.utcnow(), updated_at=datetime.utcnow())
            )

            await self.db.flush()

            logger.info(
                "Sync concluído com sucesso",
                sync_id=sync_id,
                config_id=config.id,
                results=results,
            )

            return results

        except Exception as e:
            sync_history.status = SyncStatus.FAILED.value
            sync_history.error_message = str(e)
            sync_history.completed_at = datetime.utcnow()
            sync_history.error_details = {"error": str(e), "partial_results": results}
            if sync_history.started_at:
                sync_history.duration_ms = int(
                    (sync_history.completed_at - sync_history.started_at).total_seconds() * 1000
                )
            await self.db.flush()

            logger.error(
                "Sync falhou",
                sync_id=sync_id,
                config_id=config.id,
                error=str(e),
            )
            raise

    async def get_sync_status(self, sync_id: int) -> Optional[SistemaFacebookAdsSyncHistory]:
        """Busca status de uma sincronização."""
        result = await self.db.execute(
            select(SistemaFacebookAdsSyncHistory).where(
                SistemaFacebookAdsSyncHistory.id == sync_id
            )
        )
        return result.scalar_one_or_none()

    async def get_sync_history(
        self,
        config_id: int,
        limit: int = 20,
        offset: int = 0,
    ) -> tuple[list[SistemaFacebookAdsSyncHistory], int]:
        """Lista histórico de sincronizações."""
        # Count
        count_result = await self.db.execute(
            select(func.count(SistemaFacebookAdsSyncHistory.id)).where(
                SistemaFacebookAdsSyncHistory.config_id == config_id
            )
        )
        total = count_result.scalar_one()

        # Data
        result = await self.db.execute(
            select(SistemaFacebookAdsSyncHistory)
            .where(SistemaFacebookAdsSyncHistory.config_id == config_id)
            .order_by(desc(SistemaFacebookAdsSyncHistory.started_at))
            .limit(limit)
            .offset(offset)
        )
        items = result.scalars().all()

        return items, total

    async def cancel_sync(self, sync_id: int) -> bool:
        """Cancela uma sincronização em andamento."""
        result = await self.db.execute(
            select(SistemaFacebookAdsSyncHistory).where(
                SistemaFacebookAdsSyncHistory.id == sync_id,
                SistemaFacebookAdsSyncHistory.status == SyncStatus.RUNNING.value,
            )
        )
        sync_history = result.scalar_one_or_none()

        if not sync_history:
            return False

        sync_history.status = SyncStatus.CANCELLED.value
        sync_history.completed_at = datetime.utcnow()
        await self.db.flush()

        logger.info("Sync cancelado", sync_id=sync_id)
        return True
