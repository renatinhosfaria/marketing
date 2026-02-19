"""
Modelos de banco de dados compartilhados.

- famachat_readonly: Tabelas do FamaChat Node.js (read-only)
- ml_readonly: Tabelas ML (read-only para consultas auxiliares)
  IMPORTANTE: ml_readonly NÃO é importado aqui para evitar conflito
  de classes SQLAlchemy com projects/ml/db/models.py no processo ML API.
"""

from shared.db.models.famachat_readonly import (
    SistemaUsers,
    SistemaFacebookAdsConfig,
    SistemaFacebookAdsCampaigns,
    SistemaFacebookAdsAdsets,
    SistemaFacebookAdsAds,
    SistemaFacebookAdsInsightsHistory,
    SistemaFacebookAdsInsightsToday,
    FacebookAdsConfig,
    FacebookAdsCampaign,
    FacebookAdsAdset,
    FacebookAdsAd,
    FacebookAdsInsight,
    FacebookAdsInsightToday,
)

__all__ = [
    # Users read-only
    "SistemaUsers",
    # Facebook Ads read-only
    "SistemaFacebookAdsConfig",
    "SistemaFacebookAdsCampaigns",
    "SistemaFacebookAdsAdsets",
    "SistemaFacebookAdsAds",
    "SistemaFacebookAdsInsightsHistory",
    "SistemaFacebookAdsInsightsToday",
    "FacebookAdsConfig",
    "FacebookAdsCampaign",
    "FacebookAdsAdset",
    "FacebookAdsAd",
    "FacebookAdsInsight",
    "FacebookAdsInsightToday",
]
