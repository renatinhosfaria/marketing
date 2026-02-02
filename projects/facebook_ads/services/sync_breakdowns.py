"""Serviço de sincronização de breakdowns avançados do Facebook Ads."""

from datetime import datetime
from decimal import Decimal

from sqlalchemy import delete, and_
from sqlalchemy.ext.asyncio import AsyncSession

from shared.core.logging import get_logger
from projects.facebook_ads.client.base import FacebookGraphClient
from projects.facebook_ads.client.insights import (
    InsightsClient, ADVANCED_BREAKDOWNS,
)
from projects.facebook_ads.security.token_encryption import decrypt_token
from projects.facebook_ads.utils.date_helpers import get_date_range
from projects.facebook_ads.utils.metrics_calculator import (
    calculate_ctr, calculate_cpc, calculate_cpl, extract_leads_from_actions,
)
from shared.db.models.famachat_readonly import (
    SistemaFacebookAdsConfig,
    SistemaFacebookAdsInsightsBreakdowns,
)

logger = get_logger(__name__)

BREAKDOWN_FIELDS = [
    "ad_id", "adset_id", "campaign_id",
    "impressions", "reach", "clicks", "spend",
    "actions", "action_values", "conversions", "conversion_values",
    "ctr", "cpc",
    "date_start", "date_stop",
]


class SyncBreakdownsService:
    """Sincroniza insights com breakdowns avançados."""

    BATCH_SIZE = 500

    def __init__(self, db: AsyncSession):
        self.db = db

    async def sync_breakdowns(
        self,
        config: SistemaFacebookAdsConfig,
        breakdown_types: list[str] | None = None,
        days_back: int = 30,
    ) -> dict[str, int]:
        """Sincroniza breakdowns para os tipos especificados."""
        types_to_sync = breakdown_types or list(ADVANCED_BREAKDOWNS.keys())
        since, until = get_date_range(days_back)
        time_range = {"since": since, "until": until}

        access_token = decrypt_token(config.access_token)
        graph_client = FacebookGraphClient(access_token, config.account_id)
        insights_client = InsightsClient(graph_client)

        total_inserted = 0
        total_errors = 0

        try:
            for bd_type in types_to_sync:
                bd_fields = ADVANCED_BREAKDOWNS.get(bd_type)
                if not bd_fields:
                    continue

                logger.info(
                    "Sincronizando breakdown",
                    config_id=config.id,
                    breakdown_type=bd_type,
                )

                try:
                    fb_insights = await insights_client.get_insights(
                        f"act_{config.account_id}",
                        time_range=time_range,
                        level="ad",
                        fields=BREAKDOWN_FIELDS,
                        breakdowns=bd_fields,
                    )

                    # Limpar dados anteriores deste breakdown/período
                    await self.db.execute(
                        delete(SistemaFacebookAdsInsightsBreakdowns).where(
                            and_(
                                SistemaFacebookAdsInsightsBreakdowns.config_id == config.id,
                                SistemaFacebookAdsInsightsBreakdowns.breakdown_type == bd_type,
                                SistemaFacebookAdsInsightsBreakdowns.date >= datetime.strptime(since, "%Y-%m-%d"),
                                SistemaFacebookAdsInsightsBreakdowns.date <= datetime.strptime(until, "%Y-%m-%d"),
                            )
                        )
                    )

                    inserted = 0
                    for insight in fb_insights:
                        try:
                            bd_value = None
                            for field in bd_fields:
                                bd_value = insight.get(field)
                                if bd_value:
                                    break

                            if not bd_value:
                                continue

                            actions = insight.get("actions", [])
                            impressions = int(insight.get("impressions", 0))
                            clicks = int(insight.get("clicks", 0))
                            spend = Decimal(str(insight.get("spend", "0")))
                            leads = extract_leads_from_actions(actions)

                            obj = SistemaFacebookAdsInsightsBreakdowns(
                                config_id=config.id,
                                ad_id=insight.get("ad_id", ""),
                                adset_id=insight.get("adset_id", ""),
                                campaign_id=insight.get("campaign_id", ""),
                                date=datetime.strptime(insight.get("date_start", ""), "%Y-%m-%d"),
                                breakdown_type=bd_type,
                                breakdown_value=str(bd_value),
                                impressions=impressions,
                                reach=int(insight.get("reach", 0)),
                                clicks=clicks,
                                spend=spend,
                                leads=leads,
                                conversions=int(insight.get("conversions", 0)) if insight.get("conversions") else 0,
                                conversion_values=Decimal(str(insight.get("conversion_values", "0"))) if insight.get("conversion_values") else None,
                                ctr=calculate_ctr(clicks, impressions),
                                cpc=calculate_cpc(spend, clicks),
                                cpl=calculate_cpl(spend, leads),
                                actions=actions if actions else None,
                            )
                            self.db.add(obj)
                            inserted += 1

                            if inserted % self.BATCH_SIZE == 0:
                                await self.db.flush()

                        except Exception as e:
                            logger.error("Erro ao processar breakdown", error=str(e))
                            total_errors += 1

                    await self.db.flush()
                    total_inserted += inserted
                    logger.info("Breakdown sincronizado", breakdown_type=bd_type, inserted=inserted)

                except Exception as e:
                    logger.error("Erro ao sincronizar breakdown", breakdown_type=bd_type, error=str(e))
                    total_errors += 1

            return {"inserted": total_inserted, "errors": total_errors}

        finally:
            await graph_client.close()
