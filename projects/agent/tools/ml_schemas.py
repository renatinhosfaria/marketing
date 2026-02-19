"""
Modelos Pydantic para validacao de respostas da ML API.

IMPORTANTE: Estes schemas refletem os endpoints REAIS da ML API existente.
Usados nas tools para validar que a resposta esta no formato esperado
antes de retornar ao LLM. Detecta breaking changes cedo.
"""

from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


# --- Anomalias (POST /anomalies/detect, GET /anomalies) ---

class MLAnomalyItem(BaseModel):
    """Reflete AnomalyResponse real em projects/ml/api/anomalies.py"""
    entity_type: str
    entity_id: str
    anomaly_type: str
    metric_name: str
    observed_value: float
    expected_value: float
    deviation_score: float
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    is_acknowledged: bool = False


class MLDetectResponse(BaseModel):
    """Reflete DetectResponse real."""
    config_id: int
    detected_count: int
    anomalies: List[MLAnomalyItem]


# --- Classificacoes (GET /classifications, POST /classifications/classify) ---

class MLClassificationItem(BaseModel):
    """Reflete ClassificationResponse real em projects/ml/api/classifications.py"""
    id: int
    config_id: int
    entity_type: str
    entity_id: str
    tier: str  # HIGH_PERFORMER, MODERATE, LOW, UNDERPERFORMER
    confidence_score: float = Field(ge=0, le=1)
    metrics_snapshot: Optional[dict] = None
    classified_at: datetime


# --- Previsoes (POST /predictions/cpl, /predictions/leads) ---

class MLPredictionItem(BaseModel):
    """Reflete PredictionResponse real em projects/ml/api/predictions.py"""
    id: int
    entity_type: str
    entity_id: str
    prediction_type: str
    forecast_date: datetime
    predicted_value: float
    confidence_lower: Optional[float] = None
    confidence_upper: Optional[float] = None
    created_at: datetime


# --- Recomendacoes (GET /recommendations) ---

class MLRecommendationItem(BaseModel):
    """Reflete RecommendationResponse real em projects/ml/api/recommendations.py"""
    id: int
    config_id: int
    entity_type: str
    entity_id: str
    recommendation_type: str
    action_type: Optional[str] = None
    priority: int
    title: str
    description: str
    suggested_action: Optional[dict] = None
    confidence_score: float
    is_active: bool
