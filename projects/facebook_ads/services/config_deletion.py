from __future__ import annotations

from sqlalchemy import delete, text
from sqlalchemy.ext.asyncio import AsyncSession

from projects.facebook_ads.models.management import (
    MLFacebookAdsManagementLog,
    MLFacebookAdsRateLimitLog,
)
from projects.facebook_ads.models.sync import SistemaFacebookAdsSyncHistory
from projects.ml.db.models import (
    MLAnomaly,
    MLCampaignClassification,
    MLFeature,
    MLForecast,
    MLPrediction,
    MLRecommendation,
    MLTrainedModel,
    MLTrainingJob,
)
from shared.db.models.famachat_readonly import (
    SistemaFacebookAdsAds,
    SistemaFacebookAdsAdsets,
    SistemaFacebookAdsCampaigns,
    SistemaFacebookAdsConfig,
    SistemaFacebookAdsInsightsHistory,
    SistemaFacebookAdsInsightsToday,
)
from shared.core.logging import get_logger

logger = get_logger(__name__)

_OPTIONAL_DELETE_TABLES = {
    MLFacebookAdsManagementLog.__tablename__,
    MLFacebookAdsRateLimitLog.__tablename__,
    SistemaFacebookAdsSyncHistory.__tablename__,
}


async def _get_missing_tables(db: AsyncSession, table_names: set[str]) -> set[str]:
    missing: set[str] = set()
    for table_name in table_names:
        result = await db.execute(
            text("SELECT to_regclass(:table_name)"),
            {"table_name": table_name},
        )
        if result.scalar() is None:
            missing.add(table_name)
    return missing


def build_hard_delete_statements(
    config_id: int,
    skip_tables: set[str] | None = None,
):
    statements = []
    skip_tables = skip_tables or set()

    def should_include(table_name: str) -> bool:
        return table_name not in skip_tables

    delete_statements = [
        (MLPrediction, MLPrediction.config_id == config_id),
        (MLCampaignClassification, MLCampaignClassification.config_id == config_id),
        (MLRecommendation, MLRecommendation.config_id == config_id),
        (MLAnomaly, MLAnomaly.config_id == config_id),
        (MLFeature, MLFeature.config_id == config_id),
        (MLForecast, MLForecast.config_id == config_id),
        (MLTrainingJob, MLTrainingJob.config_id == config_id),
        (MLTrainedModel, MLTrainedModel.config_id == config_id),
        (MLFacebookAdsManagementLog, MLFacebookAdsManagementLog.config_id == config_id),
        (MLFacebookAdsRateLimitLog, MLFacebookAdsRateLimitLog.config_id == config_id),
        (SistemaFacebookAdsSyncHistory, SistemaFacebookAdsSyncHistory.config_id == config_id),
        (SistemaFacebookAdsInsightsToday, SistemaFacebookAdsInsightsToday.config_id == config_id),
        (SistemaFacebookAdsInsightsHistory, SistemaFacebookAdsInsightsHistory.config_id == config_id),
        (SistemaFacebookAdsAds, SistemaFacebookAdsAds.config_id == config_id),
        (SistemaFacebookAdsAdsets, SistemaFacebookAdsAdsets.config_id == config_id),
        (SistemaFacebookAdsCampaigns, SistemaFacebookAdsCampaigns.config_id == config_id),
        (SistemaFacebookAdsConfig, SistemaFacebookAdsConfig.id == config_id),
    ]

    for model, condition in delete_statements:
        if should_include(model.__tablename__):
            statements.append(delete(model).where(condition))

    return statements


async def hard_delete_config(db: AsyncSession, config_id: int) -> None:
    missing_tables = await _get_missing_tables(db, _OPTIONAL_DELETE_TABLES)
    if missing_tables:
        logger.warning(
            "Tabelas ausentes durante hard delete de config",
            config_id=config_id,
            missing_tables=sorted(missing_tables),
        )

    for statement in build_hard_delete_statements(
        config_id,
        skip_tables=missing_tables,
    ):
        await db.execute(statement)
