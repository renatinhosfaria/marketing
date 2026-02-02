from __future__ import annotations

from typing import Sequence

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from projects.agent.db.models import AgentCheckpoint, AgentConversation, AgentWrite
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


def build_hard_delete_statements(
    config_id: int,
    thread_ids: Sequence[str],
):
    statements = []

    if thread_ids:
        statements.append(
            delete(AgentCheckpoint).where(AgentCheckpoint.thread_id.in_(thread_ids))
        )
        statements.append(
            delete(AgentWrite).where(AgentWrite.thread_id.in_(thread_ids))
        )

    statements.extend(
        [
            delete(AgentConversation).where(AgentConversation.config_id == config_id),
            delete(MLPrediction).where(MLPrediction.config_id == config_id),
            delete(MLCampaignClassification).where(
                MLCampaignClassification.config_id == config_id
            ),
            delete(MLRecommendation).where(MLRecommendation.config_id == config_id),
            delete(MLAnomaly).where(MLAnomaly.config_id == config_id),
            delete(MLFeature).where(MLFeature.config_id == config_id),
            delete(MLForecast).where(MLForecast.config_id == config_id),
            delete(MLTrainingJob).where(MLTrainingJob.config_id == config_id),
            delete(MLTrainedModel).where(MLTrainedModel.config_id == config_id),
            delete(MLFacebookAdsManagementLog).where(
                MLFacebookAdsManagementLog.config_id == config_id
            ),
            delete(MLFacebookAdsRateLimitLog).where(
                MLFacebookAdsRateLimitLog.config_id == config_id
            ),
            delete(SistemaFacebookAdsSyncHistory).where(
                SistemaFacebookAdsSyncHistory.config_id == config_id
            ),
            delete(SistemaFacebookAdsInsightsToday).where(
                SistemaFacebookAdsInsightsToday.config_id == config_id
            ),
            delete(SistemaFacebookAdsInsightsHistory).where(
                SistemaFacebookAdsInsightsHistory.config_id == config_id
            ),
            delete(SistemaFacebookAdsAds).where(
                SistemaFacebookAdsAds.config_id == config_id
            ),
            delete(SistemaFacebookAdsAdsets).where(
                SistemaFacebookAdsAdsets.config_id == config_id
            ),
            delete(SistemaFacebookAdsCampaigns).where(
                SistemaFacebookAdsCampaigns.config_id == config_id
            ),
            delete(SistemaFacebookAdsConfig).where(
                SistemaFacebookAdsConfig.id == config_id
            ),
        ]
    )

    return statements


async def hard_delete_config(db: AsyncSession, config_id: int) -> None:
    result = await db.execute(
        select(AgentConversation.thread_id).where(
            AgentConversation.config_id == config_id
        )
    )
    thread_ids = result.scalars().all()

    for statement in build_hard_delete_statements(config_id, thread_ids):
        await db.execute(statement)
