"""ML persistence models.

Re-exports from the original db/models.py location.
"""
from projects.ml.db.models import (
    ModelType,
    ModelStatus,
    PredictionType,
    CampaignTier,
    RecommendationType,
    AnomalySeverity,
    JobStatus,
    MLTrainedModel,
    MLPrediction,
    MLCampaignClassification,
    MLRecommendation,
    MLAnomaly,
    MLFeature,
    MLForecast,
    MLTrainingJob,
)

__all__ = [
    # Enums
    "ModelType",
    "ModelStatus",
    "PredictionType",
    "CampaignTier",
    "RecommendationType",
    "AnomalySeverity",
    "JobStatus",
    # Models
    "MLTrainedModel",
    "MLPrediction",
    "MLCampaignClassification",
    "MLRecommendation",
    "MLAnomaly",
    "MLFeature",
    "MLForecast",
    "MLTrainingJob",
]
