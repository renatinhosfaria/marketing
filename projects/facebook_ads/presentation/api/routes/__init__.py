"""Facebook Ads API routes.

Re-exports from the original api/ location.
"""
from projects.facebook_ads.api import health
from projects.facebook_ads.api import oauth
from projects.facebook_ads.api import config_endpoints as config
from projects.facebook_ads.api import sync
from projects.facebook_ads.api import campaigns
from projects.facebook_ads.api import insights

__all__ = ["health", "oauth", "config", "sync", "campaigns", "insights"]
