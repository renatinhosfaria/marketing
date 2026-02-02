"""
Modelos do módulo Facebook Ads.
Reutiliza modelos existentes de famachat_readonly e adiciona tabelas novas.
"""

# Reutilizar modelos existentes (READ → READ-WRITE no novo módulo)
from shared.db.models.famachat_readonly import (
    SistemaFacebookAdsConfig,
    SistemaFacebookAdsCampaigns,
    SistemaFacebookAdsAdsets,
    SistemaFacebookAdsAds,
    SistemaFacebookAdsInsightsHistory,
    SistemaFacebookAdsInsightsToday,
    SistemaFacebookAdsInsightsBreakdowns,
)

# Novos modelos do módulo ML
from projects.facebook_ads.models.sync import SistemaFacebookAdsSyncHistory
from projects.facebook_ads.models.management import (
    MLFacebookAdsManagementLog,
    MLFacebookAdsRateLimitLog,
)

__all__ = [
    "SistemaFacebookAdsConfig",
    "SistemaFacebookAdsCampaigns",
    "SistemaFacebookAdsAdsets",
    "SistemaFacebookAdsAds",
    "SistemaFacebookAdsInsightsHistory",
    "SistemaFacebookAdsInsightsToday",
    "SistemaFacebookAdsInsightsBreakdowns",
    "SistemaFacebookAdsSyncHistory",
    "MLFacebookAdsManagementLog",
    "MLFacebookAdsRateLimitLog",
]
