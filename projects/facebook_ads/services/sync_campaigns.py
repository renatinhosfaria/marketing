"""Serviço de sincronização de campanhas do Facebook Ads."""

from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from shared.core.logging import get_logger
from projects.facebook_ads.client.base import FacebookGraphClient
from projects.facebook_ads.client.campaigns import CampaignsClient
from projects.facebook_ads.security.token_encryption import decrypt_token
from projects.facebook_ads.utils.date_helpers import parse_facebook_datetime
from shared.db.models.famachat_readonly import (
    SistemaFacebookAdsConfig,
    SistemaFacebookAdsCampaigns,
)

logger = get_logger(__name__)


class SyncCampaignsService:
    """Sincroniza campanhas da Facebook Graph API para o banco local."""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def sync(self, config: SistemaFacebookAdsConfig) -> dict[str, int]:
        """Sincroniza todas as campanhas de uma conta."""
        access_token = decrypt_token(config.access_token)
        graph_client = FacebookGraphClient(access_token, config.account_id)
        campaigns_client = CampaignsClient(graph_client)

        try:
            logger.info("Iniciando sync de campanhas", config_id=config.id)

            # Buscar campanhas da API (todos os status)
            fb_campaigns = await campaigns_client.get_campaigns(
                config.account_id,
                status_filter=["ACTIVE", "PAUSED", "DELETED", "ARCHIVED"],
            )

            # Buscar campanhas existentes no banco
            result = await self.db.execute(
                select(SistemaFacebookAdsCampaigns).where(
                    SistemaFacebookAdsCampaigns.config_id == config.id
                )
            )
            existing = {c.campaign_id: c for c in result.scalars().all()}

            created = 0
            updated = 0
            errors = 0

            for fb_camp in fb_campaigns:
                try:
                    campaign_id = fb_camp.get("id", "")
                    values = {
                        "config_id": config.id,
                        "campaign_id": campaign_id,
                        "name": fb_camp.get("name", ""),
                        "objective": fb_camp.get("objective"),
                        "status": fb_camp.get("status", "UNKNOWN"),
                        "effective_status": fb_camp.get("effective_status"),
                        "daily_budget": self._parse_budget(fb_camp.get("daily_budget")),
                        "lifetime_budget": self._parse_budget(fb_camp.get("lifetime_budget")),
                        "budget_remaining": self._parse_budget(fb_camp.get("budget_remaining")),
                        "start_time": parse_facebook_datetime(fb_camp.get("start_time")),
                        "stop_time": parse_facebook_datetime(fb_camp.get("stop_time")),
                        "created_time": parse_facebook_datetime(fb_camp.get("created_time")),
                        "updated_time": parse_facebook_datetime(fb_camp.get("updated_time")),
                        "synced_at": datetime.utcnow(),
                    }

                    if campaign_id in existing:
                        # Update
                        obj = existing[campaign_id]
                        for key, value in values.items():
                            if key != "config_id":
                                setattr(obj, key, value)
                        updated += 1
                    else:
                        # Insert
                        obj = SistemaFacebookAdsCampaigns(**values)
                        self.db.add(obj)
                        created += 1

                except Exception as e:
                    logger.error("Erro ao processar campanha", campaign_id=fb_camp.get("id"), error=str(e))
                    errors += 1

            await self.db.flush()

            result = {
                "synced": len(fb_campaigns),
                "created": created,
                "updated": updated,
                "errors": errors,
            }
            logger.info("Sync de campanhas concluído", config_id=config.id, **result)
            return result

        finally:
            await graph_client.close()

    @staticmethod
    def _parse_budget(value: Any) -> float | None:
        """Parse budget value from Facebook (em centavos string -> reais decimal)."""
        if value is None:
            return None
        try:
            # Facebook retorna budgets em centavos como string
            return float(value) / 100
        except (ValueError, TypeError):
            return None
