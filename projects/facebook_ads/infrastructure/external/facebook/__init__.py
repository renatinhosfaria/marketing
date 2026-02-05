"""Facebook Graph API clients.

This module re-exports from the original client/ location.
New code should import from here; the actual implementation
will be migrated here over time.
"""
from projects.facebook_ads.client.base import (
    FacebookGraphClient,
    FacebookAPIError,
    TokenExpiredError,
    RateLimitError,
)
from projects.facebook_ads.client.campaigns import CampaignsClient
from projects.facebook_ads.client.adsets import AdsetsClient
from projects.facebook_ads.client.ads import AdsClient
from projects.facebook_ads.client.insights import InsightsClient

__all__ = [
    "FacebookGraphClient",
    "FacebookAPIError",
    "TokenExpiredError",
    "RateLimitError",
    "CampaignsClient",
    "AdsetsClient",
    "AdsClient",
    "InsightsClient",
]
