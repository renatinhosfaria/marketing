"""
Cliente para Facebook Ads Insights API.
Suporta queries síncronas e assíncronas (AdReportRun).
"""

import asyncio
import json
from datetime import datetime
from typing import Optional

from shared.core.logging import get_logger
from projects.facebook_ads.client.base import FacebookAPIError, FacebookGraphClient
from projects.facebook_ads.config import fb_settings

logger = get_logger(__name__)

# Campos de insights padrão
INSIGHT_FIELDS = [
    # Identificação
    "ad_id",
    "ad_name",
    "adset_id",
    "adset_name",
    "campaign_id",
    "campaign_name",
    "account_id",
    "account_name",
    "objective",
    "date_start",
    "date_stop",
    # Alcance e impressões
    "impressions",
    "reach",
    "frequency",
    # Cliques
    "clicks",
    "unique_clicks",
    "inline_link_clicks",
    "outbound_clicks",
    "unique_inline_link_clicks",
    "unique_outbound_clicks",
    # Custo
    "spend",
    "cpc",
    "cpm",
    "cpp",
    "cost_per_unique_click",
    "cost_per_inline_link_click",
    "cost_per_outbound_click",
    "cost_per_thruplay",
    "cost_per_unique_conversion",
    "cost_per_unique_outbound_click",
    "cost_per_inline_post_engagement",
    "cost_per_estimated_ad_recallers",
    # CTRs
    "ctr",
    "unique_ctr",
    "inline_link_click_ctr",
    "outbound_clicks_ctr",
    "unique_link_clicks_ctr",
    "unique_inline_link_click_ctr",
    "unique_outbound_clicks_ctr",
    # Conversões
    "actions",
    "action_values",
    "cost_per_action_type",
    "conversions",
    "conversion_values",
    "cost_per_conversion",
    "unique_conversions",
    # ROAS
    "purchase_roas",
    "website_purchase_roas",
    # Diagnósticos de qualidade
    "quality_ranking",
    "engagement_rate_ranking",
    "conversion_rate_ranking",
    # Vídeo — funil completo
    "video_play_actions",
    "video_15_sec_watched_actions",
    "video_p25_watched_actions",
    "video_p50_watched_actions",
    "video_p75_watched_actions",
    "video_p95_watched_actions",
    "video_30_sec_watched_actions",
    "video_p100_watched_actions",
    "video_thruplay_watched_actions",
    "video_avg_time_watched_actions",
    # Brand awareness
    "estimated_ad_recallers",
    "estimated_ad_recall_rate",
    # Catálogo
    "catalog_segment_actions",
    "catalog_segment_value",
    "converted_product_quantity",
    "converted_product_value",
]

# Breakdowns padrão
STANDARD_BREAKDOWNS = [
    "age",
    "gender",
    "country",
    "publisher_platform",
    "device_platform",
]

# Breakdowns avançados (queries separadas — não combinar entre si)
ADVANCED_BREAKDOWNS = {
    "platform_position": ["publisher_platform", "platform_position"],
    "region": ["region"],
    "impression_device": ["impression_device"],
    "frequency_value": ["frequency_value"],
    "hourly": ["hourly_stats_aggregated_by_advertiser_time_zone"],
}

# Action breakdowns
ACTION_BREAKDOWNS = ["action_type", "action_device"]

ADVANCED_ACTION_BREAKDOWNS = [
    "action_destination",
    "action_video_sound",
    "action_carousel_card_id",
    "action_carousel_card_name",
    "action_reaction",
]

# Dynamic Creative Asset breakdowns (para campanhas Advantage+/DCO)
DCO_BREAKDOWNS = [
    "body_asset",
    "title_asset",
    "image_asset",
    "video_asset",
    "call_to_action_asset",
    "description_asset",
    "link_url_asset",
    "ad_format_asset",
]

# Janelas de atribuição 2026 (7d_view e 28d_view REMOVIDOS)
ATTRIBUTION_WINDOWS = ["1d_view", "1d_click", "7d_click", "28d_click"]


class InsightsClient:
    """Client para Facebook Ads Insights API."""

    POLL_INTERVAL = 5  # seconds
    MAX_POLL_ATTEMPTS = 120  # 10 minutes max

    def __init__(self, graph_client: FacebookGraphClient):
        self.client = graph_client

    async def get_insights(
        self,
        entity_id: str,
        time_range: Optional[dict] = None,
        date_preset: Optional[str] = None,
        level: str = "ad",
        fields: Optional[list[str]] = None,
        breakdowns: Optional[list[str]] = None,
        action_breakdowns: Optional[list[str]] = None,
        time_increment: str = "1",
        limit: int = 100,
        filtering: Optional[list[dict]] = None,
    ) -> list[dict]:
        """Fetch insights synchronously. For large datasets, use get_insights_async."""
        params: dict = {
            "fields": ",".join(fields or INSIGHT_FIELDS),
            "level": level,
            "time_increment": time_increment,
        }

        if time_range:
            params["time_range"] = str(time_range).replace("'", '"')
        elif date_preset:
            params["date_preset"] = date_preset

        if breakdowns:
            params["breakdowns"] = ",".join(breakdowns)

        if action_breakdowns:
            params["action_breakdowns"] = ",".join(action_breakdowns)

        if filtering:
            params["filtering"] = json.dumps(filtering)

        endpoint = f"{entity_id}/insights"
        logger.info("Buscando insights", entity_id=entity_id, level=level)

        results = await self.client.get_all(endpoint, params, limit=limit)
        logger.info("Insights obtidos", count=len(results))
        return results

    async def get_insights_async(
        self,
        entity_id: str,
        time_range: Optional[dict] = None,
        date_preset: Optional[str] = None,
        level: str = "ad",
        fields: Optional[list[str]] = None,
        breakdowns: Optional[list[str]] = None,
        time_increment: str = "1",
    ) -> list[dict]:
        """Fetch insights asynchronously using AdReportRun. For large datasets."""
        params: dict = {
            "fields": ",".join(fields or INSIGHT_FIELDS),
            "level": level,
            "time_increment": time_increment,
        }

        if time_range:
            params["time_range"] = str(time_range).replace("'", '"')
        elif date_preset:
            params["date_preset"] = date_preset

        if breakdowns:
            params["breakdowns"] = ",".join(breakdowns)

        # Create async report
        endpoint = f"{entity_id}/insights"
        logger.info("Criando relatório assíncrono", entity_id=entity_id)

        response = await self.client.post(
            endpoint, params={"async": "true", **params}
        )
        report_run_id = response.get("report_run_id") or response.get("id")

        if not report_run_id:
            raise FacebookAPIError(
                "Falha ao criar relatório assíncrono - sem report_run_id"
            )

        logger.info("Relatório criado", report_run_id=report_run_id)

        # Poll for completion
        for attempt in range(self.MAX_POLL_ATTEMPTS):
            await asyncio.sleep(self.POLL_INTERVAL)

            status_response = await self.client.get(
                report_run_id,
                params={
                    "fields": "async_status,async_percent_completion"
                },
            )

            async_status = status_response.get("async_status", "")
            completion = status_response.get(
                "async_percent_completion", 0
            )

            logger.debug(
                "Status do relatório",
                report_run_id=report_run_id,
                status=async_status,
                completion=completion,
            )

            if async_status == "Job Completed" and completion == 100:
                break
            elif async_status == "Job Failed":
                raise FacebookAPIError(
                    f"Relatório assíncrono falhou: {status_response}",
                    code=-3,
                )
            elif async_status == "Job Skipped":
                logger.warning(
                    "Relatório pulado pelo Facebook",
                    report_run_id=report_run_id,
                )
                return []
        else:
            raise FacebookAPIError(
                f"Timeout aguardando relatório assíncrono "
                f"({self.MAX_POLL_ATTEMPTS * self.POLL_INTERVAL}s)",
                code=-4,
            )

        # Fetch results
        logger.info(
            "Obtendo resultados do relatório",
            report_run_id=report_run_id,
        )
        results = await self.client.get_all(f"{report_run_id}/insights")
        logger.info("Resultados obtidos", count=len(results))
        return results

    async def get_insights_smart(
        self,
        entity_id: str,
        time_range: Optional[dict] = None,
        date_preset: Optional[str] = None,
        level: str = "ad",
        fields: Optional[list[str]] = None,
        breakdowns: Optional[list[str]] = None,
        days_threshold: Optional[int] = None,
    ) -> list[dict]:
        """Smart fetch: uses async for large date ranges, sync for small ones."""
        threshold = (
            days_threshold or fb_settings.facebook_sync_async_threshold_days
        )

        use_async = False
        if time_range:
            since = datetime.strptime(time_range["since"], "%Y-%m-%d")
            until = datetime.strptime(time_range["until"], "%Y-%m-%d")
            days = (until - since).days
            use_async = days > threshold

        if use_async:
            logger.info(
                "Usando relatório assíncrono (período longo)", days=days
            )
            return await self.get_insights_async(
                entity_id,
                time_range=time_range,
                date_preset=date_preset,
                level=level,
                fields=fields,
                breakdowns=breakdowns,
            )
        else:
            return await self.get_insights(
                entity_id,
                time_range=time_range,
                date_preset=date_preset,
                level=level,
                fields=fields,
                breakdowns=breakdowns,
            )

    async def get_account_insights(
        self,
        ad_account_id: str,
        time_range: Optional[dict] = None,
        date_preset: str = "last_30d",
    ) -> list[dict]:
        """Shortcut para insights no nível da conta."""
        entity_id = f"act_{ad_account_id}"
        return await self.get_insights(
            entity_id,
            time_range=time_range,
            date_preset=date_preset,
            level="account",
        )
