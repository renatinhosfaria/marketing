"""
Serviço de sincronização de insights do Facebook Ads.
Implementa dual-table (today/history) e async reports para backfill.
"""

from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Optional

from sqlalchemy import select, delete, and_
from sqlalchemy.ext.asyncio import AsyncSession

from shared.core.logging import get_logger
from projects.facebook_ads.client.base import FacebookGraphClient
from projects.facebook_ads.client.insights import InsightsClient, INSIGHT_FIELDS
from projects.facebook_ads.config import fb_settings
from projects.facebook_ads.security.token_encryption import decrypt_token
from projects.facebook_ads.utils.date_helpers import is_today, get_date_range, get_today_sao_paulo
from projects.facebook_ads.utils.metrics_calculator import (
    calculate_ctr, calculate_cpc, calculate_cpm, calculate_cpl,
    calculate_frequency, calculate_cpp, extract_leads_from_actions,
    extract_roas_from_list, extract_video_metric, extract_video_avg_time,
    extract_action_stat_value, extract_landing_page_views,
)
from shared.db.models.famachat_readonly import (
    SistemaFacebookAdsConfig,
    SistemaFacebookAdsInsightsHistory,
    SistemaFacebookAdsInsightsToday,
)

logger = get_logger(__name__)


class SyncInsightsService:
    """Sincroniza insights do Facebook Ads com dual-table pattern."""

    BATCH_SIZE = 500

    def __init__(self, db: AsyncSession):
        self.db = db

    async def sync_today(self, config: SistemaFacebookAdsConfig) -> dict[str, int]:
        """Sincroniza insights do dia atual."""
        today = get_today_sao_paulo()
        time_range = {
            "since": today.strftime("%Y-%m-%d"),
            "until": today.strftime("%Y-%m-%d"),
        }

        logger.info("Sync de insights de hoje", config_id=config.id, date=str(today))

        access_token = decrypt_token(config.access_token)
        graph_client = FacebookGraphClient(access_token, config.account_id)
        insights_client = InsightsClient(graph_client)

        try:
            fb_insights = await insights_client.get_insights(
                f"act_{config.account_id}",
                time_range=time_range,
                level="ad",
            )

            # Limpar insights de hoje antes de inserir novos
            await self.db.execute(
                delete(SistemaFacebookAdsInsightsToday).where(
                    SistemaFacebookAdsInsightsToday.config_id == config.id
                )
            )

            inserted = 0
            errors = 0

            for insight in fb_insights:
                try:
                    obj = self._parse_insight_to_today(config.id, insight)
                    self.db.add(obj)
                    inserted += 1

                    if inserted % self.BATCH_SIZE == 0:
                        await self.db.flush()

                except Exception as e:
                    logger.error("Erro ao processar insight", error=str(e))
                    errors += 1

            await self.db.flush()

            result = {"synced": len(fb_insights), "inserted": inserted, "errors": errors}
            logger.info("Sync de hoje concluído", config_id=config.id, **result)
            return result

        finally:
            await graph_client.close()

    async def sync_historical(
        self,
        config: SistemaFacebookAdsConfig,
        days_back: Optional[int] = None,
    ) -> dict[str, int]:
        """Sincroniza insights históricos (usa async reports para períodos longos)."""
        days = days_back or fb_settings.facebook_sync_backfill_days
        since, until = get_date_range(days)
        time_range = {"since": since, "until": until}

        logger.info(
            "Sync histórico de insights",
            config_id=config.id,
            since=since,
            until=until,
            days=days,
        )

        access_token = decrypt_token(config.access_token)
        graph_client = FacebookGraphClient(access_token, config.account_id)
        insights_client = InsightsClient(graph_client)

        try:
            # Usa async report se período > threshold
            fb_insights = await insights_client.get_insights_smart(
                f"act_{config.account_id}",
                time_range=time_range,
                level="ad",
                days_threshold=fb_settings.facebook_sync_async_threshold_days,
            )

            inserted = 0
            updated = 0
            errors = 0

            for insight in fb_insights:
                try:
                    insight_date = insight.get("date_start", "")
                    ad_id = insight.get("ad_id", "")

                    if not insight_date or not ad_id:
                        continue

                    # Verificar se já existe
                    existing = await self.db.execute(
                        select(SistemaFacebookAdsInsightsHistory).where(
                            and_(
                                SistemaFacebookAdsInsightsHistory.config_id == config.id,
                                SistemaFacebookAdsInsightsHistory.ad_id == ad_id,
                                SistemaFacebookAdsInsightsHistory.date == datetime.strptime(insight_date, "%Y-%m-%d"),
                            )
                        )
                    )
                    existing_obj = existing.scalar_one_or_none()

                    if existing_obj:
                        self._update_insight_history(existing_obj, insight)
                        updated += 1
                    else:
                        obj = self._parse_insight_to_history(config.id, insight)
                        self.db.add(obj)
                        inserted += 1

                    if (inserted + updated) % self.BATCH_SIZE == 0:
                        await self.db.flush()

                except Exception as e:
                    logger.error("Erro ao processar insight histórico", error=str(e))
                    errors += 1

            await self.db.flush()

            result = {
                "synced": len(fb_insights),
                "inserted": inserted,
                "updated": updated,
                "errors": errors,
            }
            logger.info("Sync histórico concluído", config_id=config.id, **result)
            return result

        finally:
            await graph_client.close()

    async def sync_yesterday(self, config: SistemaFacebookAdsConfig) -> dict[str, int]:
        """Sincroniza insights de ontem (consolidação today -> history)."""
        yesterday = get_today_sao_paulo() - timedelta(days=1)
        time_range = {
            "since": yesterday.strftime("%Y-%m-%d"),
            "until": yesterday.strftime("%Y-%m-%d"),
        }

        logger.info("Sync de insights de ontem", config_id=config.id, date=str(yesterday))

        access_token = decrypt_token(config.access_token)
        graph_client = FacebookGraphClient(access_token, config.account_id)
        insights_client = InsightsClient(graph_client)

        try:
            fb_insights = await insights_client.get_insights(
                f"act_{config.account_id}",
                time_range=time_range,
                level="ad",
            )

            inserted = 0
            updated = 0
            errors = 0

            for insight in fb_insights:
                try:
                    ad_id = insight.get("ad_id", "")
                    if not ad_id:
                        continue

                    existing = await self.db.execute(
                        select(SistemaFacebookAdsInsightsHistory).where(
                            and_(
                                SistemaFacebookAdsInsightsHistory.config_id == config.id,
                                SistemaFacebookAdsInsightsHistory.ad_id == ad_id,
                                SistemaFacebookAdsInsightsHistory.date == yesterday,
                            )
                        )
                    )
                    existing_obj = existing.scalar_one_or_none()

                    if existing_obj:
                        self._update_insight_history(existing_obj, insight)
                        updated += 1
                    else:
                        obj = self._parse_insight_to_history(config.id, insight)
                        self.db.add(obj)
                        inserted += 1

                except Exception as e:
                    logger.error("Erro ao consolidar insight", error=str(e))
                    errors += 1

            await self.db.flush()
            return {"synced": len(fb_insights), "inserted": inserted, "updated": updated, "errors": errors}

        finally:
            await graph_client.close()

    def _parse_insight_to_today(self, config_id: int, insight: dict) -> SistemaFacebookAdsInsightsToday:
        """Parse insight da API para modelo InsightsToday."""
        actions = insight.get("actions", [])
        impressions = int(insight.get("impressions", 0))
        reach = int(insight.get("reach", 0))
        clicks = int(insight.get("clicks", 0))
        spend = Decimal(str(insight.get("spend", "0")))
        leads = extract_leads_from_actions(actions)

        return SistemaFacebookAdsInsightsToday(
            config_id=config_id,
            ad_id=insight.get("ad_id", ""),
            adset_id=insight.get("adset_id", ""),
            campaign_id=insight.get("campaign_id", ""),
            date=datetime.strptime(insight.get("date_start", ""), "%Y-%m-%d") if insight.get("date_start") else datetime.utcnow(),
            impressions=impressions,
            reach=reach,
            frequency=calculate_frequency(impressions, reach),
            clicks=clicks,
            unique_clicks=int(insight.get("unique_clicks", 0)),
            inline_link_clicks=int(insight.get("inline_link_clicks", 0)),
            outbound_clicks=self._extract_outbound_clicks(insight),
            spend=spend,
            cpc=calculate_cpc(spend, clicks),
            cpm=calculate_cpm(spend, impressions),
            ctr=calculate_ctr(clicks, impressions),
            conversions=self._extract_conversions(actions),
            conversion_values=self._extract_conversion_values(insight),
            leads=leads,
            cost_per_lead=calculate_cpl(spend, leads),
            video_views=self._extract_video_views(insight),
            video_p100_watched=self._extract_video_p100(insight),
            post_engagement=self._extract_engagement(actions, "post_engagement"),
            post_reactions=self._extract_engagement(actions, "post_reaction"),
            post_comments=self._extract_engagement(actions, "comment"),
            post_shares=self._extract_engagement(actions, "post"),
            actions=actions if actions else None,
            # Action values raw
            action_values=insight.get("action_values") if insight.get("action_values") else None,
            # Quality diagnostics
            quality_ranking=insight.get("quality_ranking"),
            engagement_rate_ranking=insight.get("engagement_rate_ranking"),
            conversion_rate_ranking=insight.get("conversion_rate_ranking"),
            # ROAS
            purchase_roas=extract_roas_from_list(insight.get("purchase_roas")),
            website_purchase_roas=extract_roas_from_list(insight.get("website_purchase_roas")),
            # Granular costs
            cpp=Decimal(str(insight["cpp"])) if insight.get("cpp") else calculate_cpp(spend, reach),
            cost_per_unique_click=Decimal(str(insight["cost_per_unique_click"])) if insight.get("cost_per_unique_click") else None,
            cost_per_inline_link_click=Decimal(str(insight["cost_per_inline_link_click"])) if insight.get("cost_per_inline_link_click") else None,
            cost_per_outbound_click=extract_action_stat_value(insight.get("cost_per_outbound_click"), decimal_result=True) or None,
            cost_per_thruplay=extract_action_stat_value(insight.get("cost_per_thruplay"), decimal_result=True) or None,
            # CTRs
            unique_ctr=Decimal(str(insight["unique_ctr"])) if insight.get("unique_ctr") else None,
            inline_link_click_ctr=Decimal(str(insight["inline_link_click_ctr"])) if insight.get("inline_link_click_ctr") else None,
            outbound_clicks_ctr=Decimal(str(insight["outbound_clicks_ctr"])) if insight.get("outbound_clicks_ctr") else None,
            # Video funnel
            video_plays=extract_video_metric(insight, "video_play_actions"),
            video_15s_watched=extract_video_metric(insight, "video_15_sec_watched_actions"),
            video_p25_watched=extract_video_metric(insight, "video_p25_watched_actions"),
            video_p50_watched=extract_video_metric(insight, "video_p50_watched_actions"),
            video_p75_watched=extract_video_metric(insight, "video_p75_watched_actions"),
            video_p95_watched=extract_video_metric(insight, "video_p95_watched_actions"),
            video_thruplay=extract_video_metric(insight, "video_thruplay_watched_actions"),
            video_avg_time=extract_video_avg_time(insight),
            # Unique metrics
            unique_inline_link_clicks=int(insight.get("unique_inline_link_clicks", 0)),
            unique_outbound_clicks=extract_action_stat_value(insight.get("unique_outbound_clicks")),
            unique_conversions=int(insight.get("unique_conversions", 0)) if insight.get("unique_conversions") else None,
            # Unique costs
            cost_per_unique_conversion=Decimal(str(insight["cost_per_unique_conversion"])) if insight.get("cost_per_unique_conversion") else None,
            cost_per_unique_outbound_click=extract_action_stat_value(insight.get("cost_per_unique_outbound_click"), decimal_result=True) or None,
            cost_per_inline_post_engagement=Decimal(str(insight["cost_per_inline_post_engagement"])) if insight.get("cost_per_inline_post_engagement") else None,
            # Unique CTRs
            unique_link_clicks_ctr=Decimal(str(insight["unique_link_clicks_ctr"])) if insight.get("unique_link_clicks_ctr") else None,
            unique_inline_link_click_ctr=Decimal(str(insight["unique_inline_link_click_ctr"])) if insight.get("unique_inline_link_click_ctr") else None,
            unique_outbound_clicks_ctr=Decimal(str(insight["unique_outbound_clicks_ctr"])) if insight.get("unique_outbound_clicks_ctr") else None,
            # Brand awareness
            estimated_ad_recallers=int(insight["estimated_ad_recallers"]) if insight.get("estimated_ad_recallers") else None,
            estimated_ad_recall_rate=Decimal(str(insight["estimated_ad_recall_rate"])) if insight.get("estimated_ad_recall_rate") else None,
            cost_per_estimated_ad_recallers=Decimal(str(insight["cost_per_estimated_ad_recallers"])) if insight.get("cost_per_estimated_ad_recallers") else None,
            # Landing page
            landing_page_views=extract_landing_page_views(actions),
            # Catalog
            converted_product_quantity=int(insight["converted_product_quantity"]) if insight.get("converted_product_quantity") else None,
            converted_product_value=Decimal(str(insight["converted_product_value"])) if insight.get("converted_product_value") else None,
            synced_at=datetime.utcnow(),
        )

    def _parse_insight_to_history(self, config_id: int, insight: dict) -> SistemaFacebookAdsInsightsHistory:
        """Parse insight da API para modelo InsightsHistory."""
        actions = insight.get("actions", [])
        impressions = int(insight.get("impressions", 0))
        reach = int(insight.get("reach", 0))
        clicks = int(insight.get("clicks", 0))
        spend = Decimal(str(insight.get("spend", "0")))
        leads = extract_leads_from_actions(actions)

        return SistemaFacebookAdsInsightsHistory(
            config_id=config_id,
            ad_id=insight.get("ad_id", ""),
            adset_id=insight.get("adset_id", ""),
            campaign_id=insight.get("campaign_id", ""),
            date=datetime.strptime(insight.get("date_start", ""), "%Y-%m-%d") if insight.get("date_start") else datetime.utcnow(),
            impressions=impressions,
            reach=reach,
            frequency=calculate_frequency(impressions, reach),
            clicks=clicks,
            unique_clicks=int(insight.get("unique_clicks", 0)),
            inline_link_clicks=int(insight.get("inline_link_clicks", 0)),
            outbound_clicks=self._extract_outbound_clicks(insight),
            spend=spend,
            cpc=calculate_cpc(spend, clicks),
            cpm=calculate_cpm(spend, impressions),
            ctr=calculate_ctr(clicks, impressions),
            conversions=self._extract_conversions(actions),
            conversion_values=self._extract_conversion_values(insight),
            leads=leads,
            cost_per_lead=calculate_cpl(spend, leads),
            video_views=self._extract_video_views(insight),
            video_p100_watched=self._extract_video_p100(insight),
            post_engagement=self._extract_engagement(actions, "post_engagement"),
            post_reactions=self._extract_engagement(actions, "post_reaction"),
            post_comments=self._extract_engagement(actions, "comment"),
            post_shares=self._extract_engagement(actions, "post"),
            actions=actions if actions else None,
            # Action values raw
            action_values=insight.get("action_values") if insight.get("action_values") else None,
            # Quality diagnostics
            quality_ranking=insight.get("quality_ranking"),
            engagement_rate_ranking=insight.get("engagement_rate_ranking"),
            conversion_rate_ranking=insight.get("conversion_rate_ranking"),
            # ROAS
            purchase_roas=extract_roas_from_list(insight.get("purchase_roas")),
            website_purchase_roas=extract_roas_from_list(insight.get("website_purchase_roas")),
            # Granular costs
            cpp=Decimal(str(insight["cpp"])) if insight.get("cpp") else calculate_cpp(spend, reach),
            cost_per_unique_click=Decimal(str(insight["cost_per_unique_click"])) if insight.get("cost_per_unique_click") else None,
            cost_per_inline_link_click=Decimal(str(insight["cost_per_inline_link_click"])) if insight.get("cost_per_inline_link_click") else None,
            cost_per_outbound_click=extract_action_stat_value(insight.get("cost_per_outbound_click"), decimal_result=True) or None,
            cost_per_thruplay=extract_action_stat_value(insight.get("cost_per_thruplay"), decimal_result=True) or None,
            # CTRs
            unique_ctr=Decimal(str(insight["unique_ctr"])) if insight.get("unique_ctr") else None,
            inline_link_click_ctr=Decimal(str(insight["inline_link_click_ctr"])) if insight.get("inline_link_click_ctr") else None,
            outbound_clicks_ctr=Decimal(str(insight["outbound_clicks_ctr"])) if insight.get("outbound_clicks_ctr") else None,
            # Video funnel
            video_plays=extract_video_metric(insight, "video_play_actions"),
            video_15s_watched=extract_video_metric(insight, "video_15_sec_watched_actions"),
            video_p25_watched=extract_video_metric(insight, "video_p25_watched_actions"),
            video_p50_watched=extract_video_metric(insight, "video_p50_watched_actions"),
            video_p75_watched=extract_video_metric(insight, "video_p75_watched_actions"),
            video_p95_watched=extract_video_metric(insight, "video_p95_watched_actions"),
            video_thruplay=extract_video_metric(insight, "video_thruplay_watched_actions"),
            video_avg_time=extract_video_avg_time(insight),
            # Unique metrics
            unique_inline_link_clicks=int(insight.get("unique_inline_link_clicks", 0)),
            unique_outbound_clicks=extract_action_stat_value(insight.get("unique_outbound_clicks")),
            unique_conversions=int(insight.get("unique_conversions", 0)) if insight.get("unique_conversions") else None,
            # Unique costs
            cost_per_unique_conversion=Decimal(str(insight["cost_per_unique_conversion"])) if insight.get("cost_per_unique_conversion") else None,
            cost_per_unique_outbound_click=extract_action_stat_value(insight.get("cost_per_unique_outbound_click"), decimal_result=True) or None,
            cost_per_inline_post_engagement=Decimal(str(insight["cost_per_inline_post_engagement"])) if insight.get("cost_per_inline_post_engagement") else None,
            # Unique CTRs
            unique_link_clicks_ctr=Decimal(str(insight["unique_link_clicks_ctr"])) if insight.get("unique_link_clicks_ctr") else None,
            unique_inline_link_click_ctr=Decimal(str(insight["unique_inline_link_click_ctr"])) if insight.get("unique_inline_link_click_ctr") else None,
            unique_outbound_clicks_ctr=Decimal(str(insight["unique_outbound_clicks_ctr"])) if insight.get("unique_outbound_clicks_ctr") else None,
            # Brand awareness
            estimated_ad_recallers=int(insight["estimated_ad_recallers"]) if insight.get("estimated_ad_recallers") else None,
            estimated_ad_recall_rate=Decimal(str(insight["estimated_ad_recall_rate"])) if insight.get("estimated_ad_recall_rate") else None,
            cost_per_estimated_ad_recallers=Decimal(str(insight["cost_per_estimated_ad_recallers"])) if insight.get("cost_per_estimated_ad_recallers") else None,
            # Landing page
            landing_page_views=extract_landing_page_views(actions),
            # Catalog
            converted_product_quantity=int(insight["converted_product_quantity"]) if insight.get("converted_product_quantity") else None,
            converted_product_value=Decimal(str(insight["converted_product_value"])) if insight.get("converted_product_value") else None,
            consolidated_at=datetime.utcnow(),
        )

    def _update_insight_history(self, obj: SistemaFacebookAdsInsightsHistory, insight: dict) -> None:
        """Atualiza insight existente com dados mais recentes."""
        actions = insight.get("actions", [])
        impressions = int(insight.get("impressions", 0))
        reach = int(insight.get("reach", 0))
        clicks = int(insight.get("clicks", 0))
        spend = Decimal(str(insight.get("spend", "0")))
        leads = extract_leads_from_actions(actions)

        obj.impressions = impressions
        obj.reach = reach
        obj.frequency = calculate_frequency(impressions, reach)
        obj.clicks = clicks
        obj.unique_clicks = int(insight.get("unique_clicks", 0))
        obj.inline_link_clicks = int(insight.get("inline_link_clicks", 0))
        obj.outbound_clicks = self._extract_outbound_clicks(insight)
        obj.spend = spend
        obj.cpc = calculate_cpc(spend, clicks)
        obj.cpm = calculate_cpm(spend, impressions)
        obj.ctr = calculate_ctr(clicks, impressions)
        obj.conversions = self._extract_conversions(actions)
        obj.conversion_values = self._extract_conversion_values(insight)
        obj.leads = leads
        obj.cost_per_lead = calculate_cpl(spend, leads)
        obj.video_views = self._extract_video_views(insight)
        obj.video_p100_watched = self._extract_video_p100(insight)
        obj.post_engagement = self._extract_engagement(actions, "post_engagement")
        obj.post_reactions = self._extract_engagement(actions, "post_reaction")
        obj.post_comments = self._extract_engagement(actions, "comment")
        obj.post_shares = self._extract_engagement(actions, "post")
        obj.actions = actions if actions else None
        obj.action_values = insight.get("action_values") if insight.get("action_values") else None
        obj.quality_ranking = insight.get("quality_ranking")
        obj.engagement_rate_ranking = insight.get("engagement_rate_ranking")
        obj.conversion_rate_ranking = insight.get("conversion_rate_ranking")
        obj.purchase_roas = extract_roas_from_list(insight.get("purchase_roas"))
        obj.website_purchase_roas = extract_roas_from_list(insight.get("website_purchase_roas"))
        obj.cpp = Decimal(str(insight["cpp"])) if insight.get("cpp") else calculate_cpp(spend, reach)
        obj.cost_per_unique_click = Decimal(str(insight["cost_per_unique_click"])) if insight.get("cost_per_unique_click") else None
        obj.cost_per_inline_link_click = Decimal(str(insight["cost_per_inline_link_click"])) if insight.get("cost_per_inline_link_click") else None
        obj.cost_per_outbound_click = extract_action_stat_value(insight.get("cost_per_outbound_click"), decimal_result=True) or None
        obj.cost_per_thruplay = extract_action_stat_value(insight.get("cost_per_thruplay"), decimal_result=True) or None
        obj.unique_ctr = Decimal(str(insight["unique_ctr"])) if insight.get("unique_ctr") else None
        obj.inline_link_click_ctr = Decimal(str(insight["inline_link_click_ctr"])) if insight.get("inline_link_click_ctr") else None
        obj.outbound_clicks_ctr = Decimal(str(insight["outbound_clicks_ctr"])) if insight.get("outbound_clicks_ctr") else None
        obj.video_plays = extract_video_metric(insight, "video_play_actions")
        obj.video_15s_watched = extract_video_metric(insight, "video_15_sec_watched_actions")
        obj.video_p25_watched = extract_video_metric(insight, "video_p25_watched_actions")
        obj.video_p50_watched = extract_video_metric(insight, "video_p50_watched_actions")
        obj.video_p75_watched = extract_video_metric(insight, "video_p75_watched_actions")
        obj.video_p95_watched = extract_video_metric(insight, "video_p95_watched_actions")
        obj.video_thruplay = extract_video_metric(insight, "video_thruplay_watched_actions")
        obj.video_avg_time = extract_video_avg_time(insight)
        obj.unique_inline_link_clicks = int(insight.get("unique_inline_link_clicks", 0))
        obj.unique_outbound_clicks = extract_action_stat_value(insight.get("unique_outbound_clicks"))
        obj.unique_conversions = int(insight.get("unique_conversions", 0)) if insight.get("unique_conversions") else None
        obj.cost_per_unique_conversion = Decimal(str(insight["cost_per_unique_conversion"])) if insight.get("cost_per_unique_conversion") else None
        obj.cost_per_unique_outbound_click = extract_action_stat_value(insight.get("cost_per_unique_outbound_click"), decimal_result=True) or None
        obj.cost_per_inline_post_engagement = Decimal(str(insight["cost_per_inline_post_engagement"])) if insight.get("cost_per_inline_post_engagement") else None
        obj.unique_link_clicks_ctr = Decimal(str(insight["unique_link_clicks_ctr"])) if insight.get("unique_link_clicks_ctr") else None
        obj.unique_inline_link_click_ctr = Decimal(str(insight["unique_inline_link_click_ctr"])) if insight.get("unique_inline_link_click_ctr") else None
        obj.unique_outbound_clicks_ctr = Decimal(str(insight["unique_outbound_clicks_ctr"])) if insight.get("unique_outbound_clicks_ctr") else None
        obj.estimated_ad_recallers = int(insight["estimated_ad_recallers"]) if insight.get("estimated_ad_recallers") else None
        obj.estimated_ad_recall_rate = Decimal(str(insight["estimated_ad_recall_rate"])) if insight.get("estimated_ad_recall_rate") else None
        obj.cost_per_estimated_ad_recallers = Decimal(str(insight["cost_per_estimated_ad_recallers"])) if insight.get("cost_per_estimated_ad_recallers") else None
        obj.landing_page_views = extract_landing_page_views(actions)
        obj.converted_product_quantity = int(insight["converted_product_quantity"]) if insight.get("converted_product_quantity") else None
        obj.converted_product_value = Decimal(str(insight["converted_product_value"])) if insight.get("converted_product_value") else None
        obj.consolidated_at = datetime.utcnow()

    @staticmethod
    def _extract_outbound_clicks(insight: dict) -> int:
        outbound = insight.get("outbound_clicks", [])
        if isinstance(outbound, list):
            for item in outbound:
                if isinstance(item, dict) and item.get("action_type") == "outbound_click":
                    try:
                        return int(item.get("value", 0))
                    except (ValueError, TypeError):
                        pass
        return 0

    @staticmethod
    def _extract_conversions(actions: list | None) -> int:
        if not actions or not isinstance(actions, list):
            return 0
        conversion_types = {"offsite_conversion", "onsite_conversion", "app_install"}
        total = 0
        for action in actions:
            if isinstance(action, dict):
                action_type = action.get("action_type", "")
                if any(ct in action_type for ct in conversion_types):
                    try:
                        total += int(action.get("value", 0))
                    except (ValueError, TypeError):
                        pass
        return total

    @staticmethod
    def _extract_conversion_values(insight: dict) -> Decimal | None:
        values = insight.get("action_values") or insight.get("conversion_values")
        if not values or not isinstance(values, list):
            return None
        total = Decimal("0")
        for v in values:
            if isinstance(v, dict):
                try:
                    total += Decimal(str(v.get("value", "0")))
                except Exception:
                    pass
        return total if total > 0 else None

    @staticmethod
    def _extract_video_views(insight: dict) -> int:
        views = insight.get("video_30_sec_watched_actions", [])
        if isinstance(views, list):
            for item in views:
                if isinstance(item, dict):
                    try:
                        return int(item.get("value", 0))
                    except (ValueError, TypeError):
                        pass
        return 0

    @staticmethod
    def _extract_video_p100(insight: dict) -> int:
        p100 = insight.get("video_p100_watched_actions", [])
        if isinstance(p100, list):
            for item in p100:
                if isinstance(item, dict):
                    try:
                        return int(item.get("value", 0))
                    except (ValueError, TypeError):
                        pass
        return 0

    @staticmethod
    def _extract_engagement(actions: list | None, action_type: str) -> int:
        if not actions or not isinstance(actions, list):
            return 0
        for action in actions:
            if isinstance(action, dict) and action.get("action_type") == action_type:
                try:
                    return int(action.get("value", 0))
                except (ValueError, TypeError):
                    pass
        return 0
