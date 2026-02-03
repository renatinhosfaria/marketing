"""
Modelos SQLAlchemy READ-WRITE para tabelas de ML.
Estas tabelas são gerenciadas exclusivamente pelo microserviço ML.
"""

import enum
from datetime import datetime
from decimal import Decimal
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
from sqlalchemy.orm import Mapped, mapped_column, relationship

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
    """Tipos de recomendacoes."""
    # Generic (all levels)
    BUDGET_INCREASE = "BUDGET_INCREASE"
    BUDGET_DECREASE = "BUDGET_DECREASE"
    PAUSE = "PAUSE"
    SCALE_UP = "SCALE_UP"
    REACTIVATE = "REACTIVATE"

    # Adset specific
    AUDIENCE_REVIEW = "AUDIENCE_REVIEW"
    AUDIENCE_EXPANSION = "AUDIENCE_EXPANSION"
    AUDIENCE_NARROWING = "AUDIENCE_NARROWING"

    # Ad specific
    CREATIVE_REFRESH = "CREATIVE_REFRESH"
    CREATIVE_TEST = "CREATIVE_TEST"
    CREATIVE_WINNER = "CREATIVE_WINNER"

    # Legacy (deprecated, kept for compatibility)
    PAUSE_CAMPAIGN = "PAUSE_CAMPAIGN"
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


# ==================== MODELOS ====================

class MLTrainedModel(Base):
    """
    Registro de modelos ML treinados.
    Armazena metadados, métricas e referência ao arquivo serializado.
    """
    __tablename__ = "ml_trained_models"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    model_type: Mapped[ModelType] = mapped_column(
        Enum(ModelType), nullable=False
    )
    version: Mapped[str] = mapped_column(String(50), nullable=False)
    config_id: Mapped[Optional[int]] = mapped_column(Integer)

    # Armazenamento
    model_path: Mapped[str] = mapped_column(String(500), nullable=False)
    parameters: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON)
    feature_columns: Mapped[Optional[list[str]]] = mapped_column(JSON)

    # Métricas de treinamento
    training_metrics: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON)
    validation_metrics: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON)

    # Status
    status: Mapped[ModelStatus] = mapped_column(
        Enum(ModelStatus), default=ModelStatus.TRAINING
    )
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)

    # Período de dados de treinamento
    training_data_start: Mapped[Optional[datetime]] = mapped_column(DateTime)
    training_data_end: Mapped[Optional[datetime]] = mapped_column(DateTime)
    samples_count: Mapped[Optional[int]] = mapped_column(Integer)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow
    )
    trained_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    last_used_at: Mapped[Optional[datetime]] = mapped_column(DateTime)

    # Relacionamentos
    predictions: Mapped[list["MLPrediction"]] = relationship(
        back_populates="model", lazy="selectin"
    )
    training_jobs: Mapped[list["MLTrainingJob"]] = relationship(
        back_populates="model", lazy="selectin"
    )

    __table_args__ = (
        Index("ix_ml_trained_models_type_active", "model_type", "is_active"),
        Index("ix_ml_trained_models_config_type", "config_id", "model_type"),
        {"extend_existing": True},
    )


class MLPrediction(Base):
    """
    Previsões geradas pelos modelos (CPL, Leads, etc).
    Inclui campos para validação posterior com valores reais.
    """
    __tablename__ = "ml_predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    model_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("ml_trained_models.id", ondelete="SET NULL")
    )
    config_id: Mapped[int] = mapped_column(Integer, nullable=False)

    # Entidade alvo
    entity_type: Mapped[str] = mapped_column(
        String(50), nullable=False
    )  # campaign, adset, ad
    entity_id: Mapped[str] = mapped_column(String(100), nullable=False)

    # Tipo e horizonte
    prediction_type: Mapped[PredictionType] = mapped_column(
        Enum(PredictionType), nullable=False
    )
    forecast_date: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    horizon_days: Mapped[int] = mapped_column(Integer, default=1)

    # Valores previstos
    predicted_value: Mapped[float] = mapped_column(Float, nullable=False)
    confidence_lower: Mapped[Optional[float]] = mapped_column(Float)
    confidence_upper: Mapped[Optional[float]] = mapped_column(Float)

    # Validação posterior
    actual_value: Mapped[Optional[float]] = mapped_column(Float)
    absolute_error: Mapped[Optional[float]] = mapped_column(Float)
    percentage_error: Mapped[Optional[float]] = mapped_column(Float)

    # Timestamp
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow
    )

    # Relacionamentos
    model: Mapped["MLTrainedModel"] = relationship(
        back_populates="predictions"
    )

    __table_args__ = (
        Index(
            "ix_ml_predictions_entity_date",
            "entity_type", "entity_id", "forecast_date"
        ),
        Index("ix_ml_predictions_config_type", "config_id", "prediction_type"),
        {"extend_existing": True},
    )


class MLClassification(Base):
    """
    Classificacao de entidades (campaign, adset, ad) por tier de performance.
    Mantem historico de mudancas de tier.
    """
    __tablename__ = "ml_classifications"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    config_id: Mapped[int] = mapped_column(Integer, nullable=False)
    entity_id: Mapped[str] = mapped_column(String(100), nullable=False)
    entity_type: Mapped[str] = mapped_column(
        String(20), nullable=False, default="campaign"
    )
    parent_id: Mapped[Optional[str]] = mapped_column(String(100))

    # Classificacao atual
    tier: Mapped[CampaignTier] = mapped_column(
        Enum(CampaignTier), nullable=False
    )
    confidence_score: Mapped[float] = mapped_column(Float, nullable=False)

    # Snapshot de metricas usadas
    metrics_snapshot: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON)
    feature_importances: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON)

    # Historico de mudancas
    previous_tier: Mapped[Optional[CampaignTier]] = mapped_column(
        Enum(CampaignTier)
    )
    tier_change_direction: Mapped[Optional[str]] = mapped_column(
        String(20)
    )  # improved, declined, stable

    # Validade
    classified_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow
    )
    valid_until: Mapped[Optional[datetime]] = mapped_column(DateTime)

    # Versao do modelo
    model_version: Mapped[Optional[str]] = mapped_column(String(50))

    __table_args__ = (
        Index(
            "ix_ml_classifications_entity",
            "config_id", "entity_type", "entity_id", "classified_at"
        ),
        Index("ix_ml_classifications_tier", "config_id", "tier"),
        Index("ix_ml_classifications_parent", "config_id", "parent_id"),
        {"extend_existing": True},
    )


# Backward compatibility alias
MLCampaignClassification = MLClassification


class MLRecommendation(Base):
    """
    Recomendações de otimização geradas pelo sistema.
    Rastreia aplicação e dismissal das sugestões.
    """
    __tablename__ = "ml_recommendations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    config_id: Mapped[int] = mapped_column(Integer, nullable=False)

    # Entidade alvo
    entity_type: Mapped[str] = mapped_column(String(50), nullable=False)
    entity_id: Mapped[str] = mapped_column(String(100), nullable=False)

    # Detalhes da recomendação
    recommendation_type: Mapped[RecommendationType] = mapped_column(
        Enum(RecommendationType), nullable=False
    )
    priority: Mapped[int] = mapped_column(Integer, default=5)  # 1-10

    title: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)

    # Ação sugerida
    suggested_action: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON)
    # { field, current_value, suggested_value, expected_impact }
    confidence_score: Mapped[float] = mapped_column(Float, default=0.5)
    reasoning: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON)

    # Status
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    was_applied: Mapped[bool] = mapped_column(Boolean, default=False)
    applied_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    applied_by: Mapped[Optional[int]] = mapped_column(Integer)

    dismissed: Mapped[bool] = mapped_column(Boolean, default=False)
    dismissed_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    dismissed_by: Mapped[Optional[int]] = mapped_column(Integer)
    dismissed_reason: Mapped[Optional[str]] = mapped_column(Text)

    # Timestamps
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
    """
    Anomalias detectadas em métricas de campanhas.
    """
    __tablename__ = "ml_anomalies"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    config_id: Mapped[int] = mapped_column(Integer, nullable=False)

    # Entidade
    entity_type: Mapped[str] = mapped_column(String(50), nullable=False)
    entity_id: Mapped[str] = mapped_column(String(100), nullable=False)

    # Detalhes da anomalia
    anomaly_type: Mapped[str] = mapped_column(
        String(50), nullable=False
    )  # spend_spike, cpl_spike, performance_drop
    metric_name: Mapped[str] = mapped_column(String(50), nullable=False)

    # Valores
    observed_value: Mapped[float] = mapped_column(Float, nullable=False)
    expected_value: Mapped[float] = mapped_column(Float, nullable=False)
    deviation_score: Mapped[float] = mapped_column(Float, nullable=False)

    # Severidade
    severity: Mapped[AnomalySeverity] = mapped_column(
        Enum(AnomalySeverity), nullable=False
    )

    # Acknowledgment
    is_acknowledged: Mapped[bool] = mapped_column(Boolean, default=False)
    acknowledged_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    acknowledged_by: Mapped[Optional[int]] = mapped_column(Integer)
    resolution_notes: Mapped[Optional[str]] = mapped_column(Text)

    # Timestamps
    anomaly_date: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    detected_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow
    )

    # FK opcional para recomendação gerada
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


class MLFeature(Base):
    """
    Features extraidas de entidades (campaign, adset, ad) para uso em modelos ML.
    Armazena features calculadas por entidade e janela de tempo.
    """
    __tablename__ = "ml_features"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    config_id: Mapped[int] = mapped_column(Integer, nullable=False)
    entity_id: Mapped[str] = mapped_column(String(100), nullable=False)
    entity_type: Mapped[str] = mapped_column(
        String(20), nullable=False, default="campaign"
    )
    parent_id: Mapped[Optional[str]] = mapped_column(String(100))

    # Configuracao
    window_days: Mapped[int] = mapped_column(Integer, nullable=False)
    feature_date: Mapped[datetime] = mapped_column(DateTime, nullable=False)

    # Features calculadas
    features: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON)
    insufficient_data: Mapped[bool] = mapped_column(Boolean, default=False)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow
    )

    __table_args__ = (
        Index(
            "ix_ml_features_entity",
            "config_id", "entity_type", "entity_id", "feature_date"
        ),
        Index("ix_ml_features_window", "config_id", "window_days"),
        Index("ix_ml_features_parent", "config_id", "parent_id"),
        {"extend_existing": True},
    )


class MLForecast(Base):
    """
    Forecasts de metricas (CPL, leads, spend, etc).
    Armazena previsoes geradas pelos modelos de series temporais.
    """
    __tablename__ = "ml_forecasts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    config_id: Mapped[int] = mapped_column(Integer, nullable=False)

    # Entidade alvo
    entity_type: Mapped[str] = mapped_column(String(50), nullable=False)
    entity_id: Mapped[str] = mapped_column(String(100), nullable=False)

    # Detalhes do forecast
    target_metric: Mapped[str] = mapped_column(String(50), nullable=False)
    horizon_days: Mapped[int] = mapped_column(Integer, default=7)
    method: Mapped[str] = mapped_column(String(50), nullable=False)
    predictions: Mapped[Optional[list[dict]]] = mapped_column(JSON)
    forecast_date: Mapped[datetime] = mapped_column(DateTime, nullable=False)

    # Configuracao do modelo
    window_days: Mapped[Optional[int]] = mapped_column(Integer)
    model_version: Mapped[Optional[str]] = mapped_column(String(50))
    insufficient_data: Mapped[bool] = mapped_column(Boolean, default=False)

    # Tipo de previsao
    prediction_type: Mapped[Optional[PredictionType]] = mapped_column(
        Enum(PredictionType)
    )

    # Timestamps
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


class MLTrainingJob(Base):
    """
    Jobs de treinamento de modelos (Celery tasks).
    Rastreia progresso e resultados.
    """
    __tablename__ = "ml_training_jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    model_type: Mapped[ModelType] = mapped_column(
        Enum(ModelType), nullable=False
    )
    config_id: Mapped[Optional[int]] = mapped_column(Integer)

    # Celery
    celery_task_id: Mapped[Optional[str]] = mapped_column(String(255))
    status: Mapped[JobStatus] = mapped_column(
        Enum(JobStatus), default=JobStatus.PENDING
    )
    progress: Mapped[float] = mapped_column(Float, default=0.0)

    # Resultado
    model_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("ml_trained_models.id", ondelete="SET NULL")
    )
    error_message: Mapped[Optional[str]] = mapped_column(Text)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow
    )
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime)

    # Relacionamentos
    model: Mapped[Optional["MLTrainedModel"]] = relationship(
        back_populates="training_jobs"
    )

    __table_args__ = (
        Index("ix_ml_training_jobs_status", "status"),
        Index("ix_ml_training_jobs_config", "config_id", "model_type"),
        {"extend_existing": True},
    )
