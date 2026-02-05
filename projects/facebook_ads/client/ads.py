"""Cliente para ads (anúncios) da Facebook Graph API."""

import json
from typing import Optional

from shared.core.logging import get_logger
from projects.facebook_ads.client.base import FacebookGraphClient

logger = get_logger(__name__)

AD_FIELDS = [
    "id",
    "name",
    "campaign_id",
    "adset_id",
    "status",
    "effective_status",
    "creative",
    "tracking_specs",
    "preview_shareable_link",
    "created_time",
    "updated_time",
]


class AdsClient:
    """Client para operações com anúncios."""

    def __init__(self, graph_client: FacebookGraphClient):
        self.client = graph_client

    async def get_ads(
        self,
        ad_account_id: str,
        campaign_id: Optional[str] = None,
        adset_id: Optional[str] = None,
        status_filter: Optional[list[str]] = None,
        fields: Optional[list[str]] = None,
    ) -> list[dict]:
        """Busca anúncios. Pode filtrar por campanha ou ad set."""
        params: dict = {
            "fields": ",".join(fields or AD_FIELDS),
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

        if adset_id:
            endpoint = f"{adset_id}/ads"
        elif campaign_id:
            endpoint = f"{campaign_id}/ads"
        else:
            acct = ad_account_id if ad_account_id.startswith("act_") else f"act_{ad_account_id}"
            endpoint = f"{acct}/ads"

        logger.info("Buscando anúncios", account_id=ad_account_id)
        results = await self.client.get_all(endpoint, params)
        logger.info("Anúncios encontrados", count=len(results))
        return results

    async def get_ad(
        self, ad_id: str, fields: Optional[list[str]] = None
    ) -> dict:
        """Busca um anúncio específico."""
        params = {"fields": ",".join(fields or AD_FIELDS)}
        return await self.client.get(ad_id, params)
