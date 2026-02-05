"""
Modelos SQLAlchemy READ-ONLY para tabelas de ML.
Usados pelo Agent para leitura de dados produzidos pelo ML.

IMPORTANTE: Estes modelos são duplicatas somente-leitura dos modelos canônicos
em projects/ml/db/models.py. Quando o schema ML mudar, atualizar AMBOS os arquivos.
"""

import enum
from datetime import datetime
from typing import Any, Optional

from sqlalchemy import (
    Integer,
    String,
    Text,
    Boolean,
    Float,
    Numeric,
    DateTime,
    JSON,
    Enum,
    ForeignKey,
    Index,
)
from sqlalchemy.orm import Mapped, mapped_column

from shared.db.session import Base


# ==================== ENUMS ====================

class ModelType(str, enum.Enum):
    """Tipos de modelos ML disponíveis."""
    TIME_SERIES_CPL = "TIME_SERIES_CPL"
    TIME_SERIES_LEADS = "TIME_SERIES_LEADS"
    CAMPAIGN_CLASSIFIER = "CAMPAIGN_CLASSIFIER"
    ANOMALY_DETECTOR = "ANOMALY_DETECTOR"
    RECOMMENDER = "RECOMMENDER"


class ModelStatus(str, enum.Enum):
    """Status do modelo ML."""
    TRAINING = "TRAINING"
    READY = "READY"
    ACTIVE = "ACTIVE"
    DEPRECATED = "DEPRECATED"
    FAILED = "FAILED"


class PredictionType(str, enum.Enum):
    """Tipos de previsões."""
    CPL_FORECAST = "CPL_FORECAST"
    LEADS_FORECAST = "LEADS_FORECAST"
    SPEND_FORECAST = "SPEND_FORECAST"


class CampaignTier(str, enum.Enum):
    """Tiers de classificação de campanhas."""
    HIGH_PERFORMER = "HIGH_PERFORMER"
    MODERATE = "MODERATE"
    LOW = "LOW"
    UNDERPERFORMER = "UNDERPERFORMER"


class RecommendationType(str, enum.Enum):
    """Tipos de recomendações."""
    BUDGET_INCREASE = "BUDGET_INCREASE"
    BUDGET_DECREASE = "BUDGET_DECREASE"
    PAUSE_CAMPAIGN = "PAUSE_CAMPAIGN"
    SCALE_UP = "SCALE_UP"
    CREATIVE_REFRESH = "CREATIVE_REFRESH"
    AUDIENCE_REVIEW = "AUDIENCE_REVIEW"
    REACTIVATE = "REACTIVATE"
    OPTIMIZE_SCHEDULE = "OPTIMIZE_SCHEDULE"


class AnomalySeverity(str, enum.Enum):
    """Severidade de anomalias."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class JobStatus(str, enum.Enum):
    """Status de jobs de treinamento."""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


# ==================== MODELOS READ-ONLY ====================
# Nota: Não incluem relationships de escrita nem back_populates.
# O Agent só lê estes dados, nunca escreve.

class MLPrediction(Base):
    """Previsões geradas pelos modelos (READ-ONLY para Agent)."""
    __tablename__ = "ml_predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    model_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("ml_trained_models.id", ondelete="SET NULL")
    )
    config_id: Mapped[int] = mapped_column(Integer, nullable=False)
    entity_type: Mapped[str] = mapped_column(String(50), nullable=False)
    entity_id: Mapped[str] = mapped_column(String(100), nullable=False)
    prediction_type: Mapped[PredictionType] = mapped_column(
        Enum(PredictionType), nullable=False
    )
    forecast_date: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    horizon_days: Mapped[int] = mapped_column(Integer, default=1)
    predicted_value: Mapped[float] = mapped_column(Float, nullable=False)
    confidence_lower: Mapped[Optional[float]] = mapped_column(Float)
    confidence_upper: Mapped[Optional[float]] = mapped_column(Float)
    actual_value: Mapped[Optional[float]] = mapped_column(Float)
    absolute_error: Mapped[Optional[float]] = mapped_column(Float)
    percentage_error: Mapped[Optional[float]] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow
    )

    __table_args__ = (
        Index(
            "ix_ml_predictions_entity_date",
            "entity_type", "entity_id", "forecast_date"
        ),
        Index("ix_ml_predictions_config_type", "config_id", "prediction_type"),
        {"extend_existing": True},
    )


class MLCampaignClassification(Base):
    """Classificação de campanhas por tier (READ-ONLY para Agent)."""
    __tablename__ = "ml_campaign_classifications"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    config_id: Mapped[int] = mapped_column(Integer, nullable=False)
    campaign_id: Mapped[str] = mapped_column(String(100), nullable=False)
    tier: Mapped[CampaignTier] = mapped_column(
        Enum(CampaignTier), nullable=False
    )
    confidence_score: Mapped[float] = mapped_column(Float, nullable=False)
    metrics_snapshot: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON)
    feature_importances: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON)
    previous_tier: Mapped[Optional[CampaignTier]] = mapped_column(
        Enum(CampaignTier)
    )
    tier_change_direction: Mapped[Optional[str]] = mapped_column(String(20))
    classified_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow
    )
    valid_until: Mapped[Optional[datetime]] = mapped_column(DateTime)
    model_version: Mapped[Optional[str]] = mapped_column(String(50))

    __table_args__ = (
        Index(
            "ix_ml_classifications_campaign",
            "config_id", "campaign_id", "classified_at"
        ),
        Index("ix_ml_classifications_tier", "config_id", "tier"),
        {"extend_existing": True},
    )


class MLRecommendation(Base):
    """Recomendações de otimização (READ-ONLY para Agent)."""
    __tablename__ = "ml_recommendations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    config_id: Mapped[int] = mapped_column(Integer, nullable=False)
    entity_type: Mapped[str] = mapped_column(String(50), nullable=False)
    entity_id: Mapped[str] = mapped_column(String(100), nullable=False)
    recommendation_type: Mapped[RecommendationType] = mapped_column(
        Enum(RecommendationType), nullable=False
    )
    priority: Mapped[int] = mapped_column(Integer, default=5)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    suggested_action: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON)
    confidence_score: Mapped[float] = mapped_column(Float, default=0.5)
    reasoning: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    was_applied: Mapped[bool] = mapped_column(Boolean, default=False)
    applied_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    applied_by: Mapped[Optional[int]] = mapped_column(Integer)
    dismissed: Mapped[bool] = mapped_column(Boolean, default=False)
    dismissed_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    dismissed_by: Mapped[Optional[int]] = mapped_column(Integer)
    dismissed_reason: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow
    )
    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime)

    __table_args__ = (
        Index(
            "ix_ml_recommendations_entity",
            "config_id", "entity_type", "entity_id"
        ),
        Index("ix_ml_recommendations_active", "config_id", "is_active"),
        Index("ix_ml_recommendations_type", "config_id", "recommendation_type"),
        {"extend_existing": True},
    )


class MLAnomaly(Base):
    """Anomalias detectadas (READ-ONLY para Agent)."""
    __tablename__ = "ml_anomalies"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    config_id: Mapped[int] = mapped_column(Integer, nullable=False)
    entity_type: Mapped[str] = mapped_column(String(50), nullable=False)
    entity_id: Mapped[str] = mapped_column(String(100), nullable=False)
    anomaly_type: Mapped[str] = mapped_column(String(50), nullable=False)
    metric_name: Mapped[str] = mapped_column(String(50), nullable=False)
    observed_value: Mapped[float] = mapped_column(Float, nullable=False)
    expected_value: Mapped[float] = mapped_column(Float, nullable=False)
    deviation_score: Mapped[float] = mapped_column(Float, nullable=False)
    severity: Mapped[AnomalySeverity] = mapped_column(
        Enum(AnomalySeverity), nullable=False
    )
    is_acknowledged: Mapped[bool] = mapped_column(Boolean, default=False)
    acknowledged_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    acknowledged_by: Mapped[Optional[int]] = mapped_column(Integer)
    resolution_notes: Mapped[Optional[str]] = mapped_column(Text)
    anomaly_date: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    detected_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow
    )
    recommendation_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("ml_recommendations.id", ondelete="SET NULL")
    )

    __table_args__ = (
        Index(
            "ix_ml_anomalies_entity",
            "config_id", "entity_type", "entity_id", "anomaly_date"
        ),
        Index("ix_ml_anomalies_severity", "config_id", "severity"),
        Index("ix_ml_anomalies_unacknowledged", "config_id", "is_acknowledged"),
        {"extend_existing": True},
    )


class MLForecast(Base):
    """Forecasts de métricas (READ-ONLY para Agent)."""
    __tablename__ = "ml_forecasts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    config_id: Mapped[int] = mapped_column(Integer, nullable=False)
    entity_type: Mapped[str] = mapped_column(String(50), nullable=False)
    entity_id: Mapped[str] = mapped_column(String(100), nullable=False)
    target_metric: Mapped[str] = mapped_column(String(50), nullable=False)
    horizon_days: Mapped[int] = mapped_column(Integer, default=7)
    method: Mapped[str] = mapped_column(String(50), nullable=False)
    predictions: Mapped[Optional[list[dict]]] = mapped_column(JSON)
    forecast_date: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    window_days: Mapped[Optional[int]] = mapped_column(Integer)
    model_version: Mapped[Optional[str]] = mapped_column(String(50))
    insufficient_data: Mapped[bool] = mapped_column(Boolean, default=False)
    prediction_type: Mapped[Optional[PredictionType]] = mapped_column(
        Enum(PredictionType)
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow
    )

    __table_args__ = (
        Index(
            "ix_ml_forecasts_entity",
            "config_id", "entity_type", "entity_id", "forecast_date"
        ),
        Index("ix_ml_forecasts_metric", "config_id", "target_metric"),
        {"extend_existing": True},
    )
