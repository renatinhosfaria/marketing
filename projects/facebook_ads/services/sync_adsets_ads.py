"""Serviço de sincronização de ad sets e ads do Facebook Ads."""

from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from shared.core.logging import get_logger
from projects.facebook_ads.client.base import FacebookGraphClient
from projects.facebook_ads.client.adsets import AdSetsClient
from projects.facebook_ads.client.ads import AdsClient
from projects.facebook_ads.security.token_encryption import decrypt_token
from projects.facebook_ads.utils.date_helpers import parse_facebook_datetime
from shared.db.models.famachat_readonly import (
    SistemaFacebookAdsConfig,
    SistemaFacebookAdsAdsets,
    SistemaFacebookAdsAds,
)

logger = get_logger(__name__)


class SyncAdSetsAdsService:
    """Sincroniza ad sets e ads da Facebook Graph API."""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def sync_adsets(self, config: SistemaFacebookAdsConfig) -> dict[str, int]:
        """Sincroniza todos os ad sets de uma conta."""
        access_token = decrypt_token(config.access_token)
        graph_client = FacebookGraphClient(access_token, config.account_id)
        adsets_client = AdSetsClient(graph_client)

        try:
            logger.info("Iniciando sync de ad sets", config_id=config.id)

            fb_adsets = await adsets_client.get_adsets(
                config.account_id,
                status_filter=["ACTIVE", "PAUSED", "DELETED", "ARCHIVED"],
            )

            result_db = await self.db.execute(
                select(SistemaFacebookAdsAdsets).where(
                    SistemaFacebookAdsAdsets.config_id == config.id
                )
            )
            existing = {a.adset_id: a for a in result_db.scalars().all()}

            created = 0
            updated = 0
            errors = 0

            for fb_adset in fb_adsets:
                try:
                    adset_id = fb_adset.get("id", "")
                    values = {
                        "config_id": config.id,
                        "campaign_id": fb_adset.get("campaign_id", ""),
                        "adset_id": adset_id,
                        "name": fb_adset.get("name", ""),
                        "status": fb_adset.get("status", "UNKNOWN"),
                        "effective_status": fb_adset.get("effective_status"),
                        "daily_budget": self._parse_budget(fb_adset.get("daily_budget")),
                        "lifetime_budget": self._parse_budget(fb_adset.get("lifetime_budget")),
                        "budget_remaining": self._parse_budget(fb_adset.get("budget_remaining")),
                        "bid_amount": self._parse_budget(fb_adset.get("bid_amount")),
                        "bid_strategy": fb_adset.get("bid_strategy"),
                        "optimization_goal": fb_adset.get("optimization_goal"),
                        "billing_event": fb_adset.get("billing_event"),
                        "targeting": fb_adset.get("targeting"),
                        "start_time": parse_facebook_datetime(fb_adset.get("start_time")),
                        "end_time": parse_facebook_datetime(fb_adset.get("end_time")),
                        "created_time": parse_facebook_datetime(fb_adset.get("created_time")),
                        "updated_time": parse_facebook_datetime(fb_adset.get("updated_time")),
                        "synced_at": datetime.utcnow(),
                    }

                    if adset_id in existing:
                        obj = existing[adset_id]
                        for key, value in values.items():
                            if key != "config_id":
                                setattr(obj, key, value)
                        updated += 1
                    else:
                        obj = SistemaFacebookAdsAdsets(**values)
                        self.db.add(obj)
                        created += 1

                except Exception as e:
                    logger.error("Erro ao processar ad set", adset_id=fb_adset.get("id"), error=str(e))
                    errors += 1

            await self.db.flush()

            result = {"synced": len(fb_adsets), "created": created, "updated": updated, "errors": errors}
            logger.info("Sync de ad sets concluído", config_id=config.id, **result)
            return result

        finally:
            await graph_client.close()

    async def sync_ads(self, config: SistemaFacebookAdsConfig) -> dict[str, int]:
        """Sincroniza todos os anúncios de uma conta."""
        access_token = decrypt_token(config.access_token)
        graph_client = FacebookGraphClient(access_token, config.account_id)
        ads_client = AdsClient(graph_client)

        try:
            logger.info("Iniciando sync de anúncios", config_id=config.id)

            fb_ads = await ads_client.get_ads(
                config.account_id,
                status_filter=["ACTIVE", "PAUSED", "DELETED", "ARCHIVED"],
            )

            result_db = await self.db.execute(
                select(SistemaFacebookAdsAds).where(
                    SistemaFacebookAdsAds.config_id == config.id
                )
            )
            existing = {a.ad_id: a for a in result_db.scalars().all()}

            created = 0
            updated = 0
            errors = 0

            for fb_ad in fb_ads:
                try:
                    ad_id = fb_ad.get("id", "")
                    creative = fb_ad.get("creative", {})
                    values = {
                        "config_id": config.id,
                        "campaign_id": fb_ad.get("campaign_id", ""),
                        "adset_id": fb_ad.get("adset_id", ""),
                        "ad_id": ad_id,
                        "name": fb_ad.get("name", ""),
                        "status": fb_ad.get("status", "UNKNOWN"),
                        "effective_status": fb_ad.get("effective_status"),
                        "creative_id": creative.get("id") if isinstance(creative, dict) else str(creative) if creative else None,
                        "preview_shareable_link": fb_ad.get("preview_shareable_link"),
                        "tracking_specs": fb_ad.get("tracking_specs"),
                        "created_time": parse_facebook_datetime(fb_ad.get("created_time")),
                        "updated_time": parse_facebook_datetime(fb_ad.get("updated_time")),
                        "synced_at": datetime.utcnow(),
                    }

                    if ad_id in existing:
                        obj = existing[ad_id]
                        for key, value in values.items():
                            if key != "config_id":
                                setattr(obj, key, value)
                        updated += 1
                    else:
                        obj = SistemaFacebookAdsAds(**values)
                        self.db.add(obj)
                        created += 1

                except Exception as e:
                    logger.error("Erro ao processar anúncio", ad_id=fb_ad.get("id"), error=str(e))
                    errors += 1

            await self.db.flush()

            result = {"synced": len(fb_ads), "created": created, "updated": updated, "errors": errors}
            logger.info("Sync de anúncios concluído", config_id=config.id, **result)
            return result

        finally:
            await graph_client.close()

    @staticmethod
    def _parse_budget(value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value) / 100
        except (ValueError, TypeError):
            return None
