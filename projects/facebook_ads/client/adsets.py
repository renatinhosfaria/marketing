"""Cliente para ad sets da Facebook Graph API."""

import json
from typing import Optional

from shared.core.logging import get_logger
from projects.facebook_ads.client.base import FacebookGraphClient

logger = get_logger(__name__)

ADSET_FIELDS = [
    "id",
    "name",
    "campaign_id",
    "status",
    "effective_status",
    "daily_budget",
    "lifetime_budget",
    "budget_remaining",
    "bid_amount",
    "bid_strategy",
    "optimization_goal",
    "billing_event",
    "targeting",
    "start_time",
    "end_time",
    "created_time",
    "updated_time",
]


class AdSetsClient:
    """Client para operações com ad sets."""

    def __init__(self, graph_client: FacebookGraphClient):
        self.client = graph_client

    async def get_adsets(
        self,
        ad_account_id: str,
        campaign_id: Optional[str] = None,
        status_filter: Optional[list[str]] = None,
        fields: Optional[list[str]] = None,
    ) -> list[dict]:
        """Busca ad sets. Se campaign_id fornecido, filtra por campanha."""
        params: dict = {
            "fields": ",".join(fields or ADSET_FIELDS),
        }

        filters = []
        if status_filter:
            filters.append(
                {
                    "field": "effective_status",
                    "operator": "IN",
                    "value": status_filter,
                }
            )

        if filters:
            params["filtering"] = json.dumps(filters)

        if campaign_id:
            endpoint = f"{campaign_id}/adsets"
        else:
            acct = ad_account_id if ad_account_id.startswith("act_") else f"act_{ad_account_id}"
            endpoint = f"{acct}/adsets"

        logger.info(
            "Buscando ad sets",
            account_id=ad_account_id,
            campaign_id=campaign_id,
        )
        results = await self.client.get_all(endpoint, params)
        logger.info("Ad sets encontrados", count=len(results))
        return results

    async def get_adset(
        self, adset_id: str, fields: Optional[list[str]] = None
    ) -> dict:
        """Busca um ad set específico."""
        params = {"fields": ",".join(fields or ADSET_FIELDS)}
        return await self.client.get(adset_id, params)
