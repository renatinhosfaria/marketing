"""Cliente para campanhas da Facebook Graph API."""

import json
from typing import Optional

from shared.core.logging import get_logger
from projects.facebook_ads.client.base import FacebookGraphClient

logger = get_logger(__name__)

CAMPAIGN_FIELDS = [
    "id",
    "name",
    "objective",
    "status",
    "effective_status",
    "daily_budget",
    "lifetime_budget",
    "budget_remaining",
    "start_time",
    "stop_time",
    "created_time",
    "updated_time",
    "buying_type",
    "bid_strategy",
    "special_ad_categories",
]


class CampaignsClient:
    """Client para operações com campanhas."""

    def __init__(self, graph_client: FacebookGraphClient):
        self.client = graph_client

    async def get_campaigns(
        self,
        ad_account_id: str,
        status_filter: Optional[list[str]] = None,
        fields: Optional[list[str]] = None,
    ) -> list[dict]:
        """Busca todas as campanhas de uma ad account."""
        params: dict = {
            "fields": ",".join(fields or CAMPAIGN_FIELDS),
        }
        if status_filter:
            # Facebook expects effective_status as JSON array
            params["filtering"] = json.dumps(
                [
                    {
                        "field": "effective_status",
                        "operator": "IN",
                        "value": status_filter,
                    }
                ]
            )

        acct = ad_account_id if ad_account_id.startswith("act_") else f"act_{ad_account_id}"
        endpoint = f"{acct}/campaigns"
        logger.info("Buscando campanhas", account_id=ad_account_id)

        results = await self.client.get_all(endpoint, params)
        logger.info(
            "Campanhas encontradas",
            count=len(results),
            account_id=ad_account_id,
        )
        return results

    async def get_campaign(
        self, campaign_id: str, fields: Optional[list[str]] = None
    ) -> dict:
        """Busca uma campanha específica."""
        params = {"fields": ",".join(fields or CAMPAIGN_FIELDS)}
        return await self.client.get(campaign_id, params)
