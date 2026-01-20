"""Database models."""

from app.db.models.ml_models import (
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
    MLFeature,
    MLForecast,
    MLAnomaly,
    MLRecommendation,
    MLTrainingJob,
)

from app.db.models.agent_models import (
    MessageRole,
    AgentConversation,
    AgentMessage,
    AgentCheckpoint,
    AgentWrite,
    AgentFeedback,
)

__all__ = [
    # ML Models Enums
    "ModelType",
    "ModelStatus",
    "PredictionType",
    "CampaignTier",
    "RecommendationType",
    "AnomalySeverity",
    "JobStatus",
    # ML Models
    "MLTrainedModel",
    "MLPrediction",
    "MLCampaignClassification",
    "MLFeature",
    "MLForecast",
    "MLAnomaly",
    "MLRecommendation",
    "MLTrainingJob",
    # Agent Models Enums
    "MessageRole",
    # Agent Models
    "AgentConversation",
    "AgentMessage",
    "AgentCheckpoint",
    "AgentWrite",
    "AgentFeedback",
]
