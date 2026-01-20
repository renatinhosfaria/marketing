"""
Repositório para operações CRUD nas tabelas de ML.
"""

from datetime import datetime, timedelta, date
from typing import Optional, Sequence

from sqlalchemy import select, update, and_, or_, desc, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.ml_models import (
    MLTrainedModel,
    MLPrediction,
    MLCampaignClassification,
    MLRecommendation,
    MLAnomaly,
    MLTrainingJob,
    MLFeature,
    MLForecast,
    ModelType,
    ModelStatus,
    CampaignTier,
    RecommendationType,
    AnomalySeverity,
    JobStatus,
    PredictionType,
)
from app.core.logging import get_logger

logger = get_logger(__name__)


class MLRepository:
    """
    Repositório para tabelas de ML.
    Operações READ-WRITE.
    """

    def __init__(self, session: AsyncSession):
        self.session = session

    # ==================== MODELS ====================

    async def create_model(
        self,
        name: str,
        model_type: ModelType,
        version: str,
        model_path: str,
        config_id: Optional[int] = None,
        parameters: Optional[dict] = None,
        feature_columns: Optional[list] = None,
    ) -> MLTrainedModel:
        """Cria registro de novo modelo."""
        model = MLTrainedModel(
            name=name,
            model_type=model_type,
            version=version,
            model_path=model_path,
            config_id=config_id,
            parameters=parameters,
            feature_columns=feature_columns,
            status=ModelStatus.TRAINING,
        )
        self.session.add(model)
        await self.session.flush()
        return model

    async def get_model(self, model_id: int) -> Optional[MLTrainedModel]:
        """Obtém modelo por ID."""
        result = await self.session.execute(
            select(MLTrainedModel).where(MLTrainedModel.id == model_id)
        )
        return result.scalar_one_or_none()

    async def get_active_model(
        self,
        model_type: ModelType,
        config_id: Optional[int] = None,
    ) -> Optional[MLTrainedModel]:
        """Obtém modelo ativo de um tipo específico."""
        query = select(MLTrainedModel).where(
            and_(
                MLTrainedModel.model_type == model_type,
                MLTrainedModel.is_active == True,
                MLTrainedModel.status == ModelStatus.ACTIVE,
            )
        )
        if config_id:
            query = query.where(MLTrainedModel.config_id == config_id)

        query = query.order_by(desc(MLTrainedModel.trained_at)).limit(1)
        result = await self.session.execute(query)
        return result.scalar_one_or_none()

    async def update_model_status(
        self,
        model_id: int,
        status: ModelStatus,
        training_metrics: Optional[dict] = None,
        validation_metrics: Optional[dict] = None,
    ) -> None:
        """Atualiza status do modelo."""
        values = {"status": status}
        if training_metrics:
            values["training_metrics"] = training_metrics
        if validation_metrics:
            values["validation_metrics"] = validation_metrics
        if status == ModelStatus.ACTIVE:
            values["trained_at"] = datetime.utcnow()

        await self.session.execute(
            update(MLTrainedModel)
            .where(MLTrainedModel.id == model_id)
            .values(**values)
        )

    async def activate_model(self, model_id: int, model_type: ModelType) -> None:
        """Ativa um modelo e desativa outros do mesmo tipo."""
        # Desativar modelos anteriores
        await self.session.execute(
            update(MLTrainedModel)
            .where(
                and_(
                    MLTrainedModel.model_type == model_type,
                    MLTrainedModel.is_active == True,
                )
            )
            .values(is_active=False)
        )
        # Ativar novo modelo
        await self.session.execute(
            update(MLTrainedModel)
            .where(MLTrainedModel.id == model_id)
            .values(is_active=True, status=ModelStatus.ACTIVE)
        )

    # ==================== PREDICTIONS ====================

    async def create_prediction(
        self,
        model_id: Optional[int],
        config_id: int,
        entity_type: str,
        entity_id: str,
        prediction_type: PredictionType,
        forecast_date: datetime,
        predicted_value: float,
        horizon_days: int = 1,
        confidence_lower: Optional[float] = None,
        confidence_upper: Optional[float] = None,
    ) -> MLPrediction:
        """Cria nova previsão."""
        prediction = MLPrediction(
            model_id=model_id,
            config_id=config_id,
            entity_type=entity_type,
            entity_id=entity_id,
            prediction_type=prediction_type,
            forecast_date=forecast_date,
            predicted_value=predicted_value,
            horizon_days=horizon_days,
            confidence_lower=confidence_lower,
            confidence_upper=confidence_upper,
        )
        self.session.add(prediction)
        await self.session.flush()
        return prediction

    async def get_predictions(
        self,
        config_id: int,
        entity_type: str,
        entity_id: str,
        prediction_type: Optional[PredictionType] = None,
        limit: int = 30,
    ) -> Sequence[MLPrediction]:
        """Obtém previsões de uma entidade."""
        query = select(MLPrediction).where(
            and_(
                MLPrediction.config_id == config_id,
                MLPrediction.entity_type == entity_type,
                MLPrediction.entity_id == entity_id,
            )
        )
        if prediction_type:
            query = query.where(MLPrediction.prediction_type == prediction_type)

        query = query.order_by(desc(MLPrediction.forecast_date)).limit(limit)
        result = await self.session.execute(query)
        return result.scalars().all()

    async def update_prediction_actual(
        self,
        prediction_id: int,
        actual_value: float,
    ) -> None:
        """Atualiza previsão com valor real (para validação)."""
        result = await self.session.execute(
            select(MLPrediction).where(MLPrediction.id == prediction_id)
        )
        prediction = result.scalar_one_or_none()
        if prediction:
            absolute_error = abs(actual_value - prediction.predicted_value)
            percentage_error = (
                (absolute_error / actual_value * 100) if actual_value != 0 else 0
            )
            await self.session.execute(
                update(MLPrediction)
                .where(MLPrediction.id == prediction_id)
                .values(
                    actual_value=actual_value,
                    absolute_error=absolute_error,
                    percentage_error=percentage_error,
                )
            )

    # ==================== CLASSIFICATIONS ====================

    async def create_classification(
        self,
        config_id: int,
        campaign_id: str,
        tier: CampaignTier,
        confidence_score: float,
        metrics_snapshot: Optional[dict] = None,
        feature_importances: Optional[dict] = None,
        previous_tier: Optional[CampaignTier] = None,
        model_version: Optional[str] = None,
    ) -> MLCampaignClassification:
        """Cria nova classificação de campanha."""
        tier_change = "stable"
        if previous_tier:
            tier_order = [
                CampaignTier.UNDERPERFORMER,
                CampaignTier.LOW,
                CampaignTier.MODERATE,
                CampaignTier.HIGH_PERFORMER,
            ]
            current_idx = tier_order.index(tier)
            prev_idx = tier_order.index(previous_tier)
            if current_idx > prev_idx:
                tier_change = "improved"
            elif current_idx < prev_idx:
                tier_change = "declined"

        classification = MLCampaignClassification(
            config_id=config_id,
            campaign_id=campaign_id,
            tier=tier,
            confidence_score=confidence_score,
            metrics_snapshot=metrics_snapshot,
            feature_importances=feature_importances,
            previous_tier=previous_tier,
            tier_change_direction=tier_change,
            model_version=model_version,
            valid_until=datetime.utcnow() + timedelta(days=1),
        )
        self.session.add(classification)
        await self.session.flush()
        return classification

    async def get_latest_classification(
        self,
        config_id: int,
        campaign_id: str,
    ) -> Optional[MLCampaignClassification]:
        """Obtém classificação mais recente de uma campanha."""
        result = await self.session.execute(
            select(MLCampaignClassification)
            .where(
                and_(
                    MLCampaignClassification.config_id == config_id,
                    MLCampaignClassification.campaign_id == campaign_id,
                )
            )
            .order_by(desc(MLCampaignClassification.classified_at))
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def get_classifications_by_tier(
        self,
        config_id: int,
        tier: Optional[CampaignTier] = None,
    ) -> Sequence[MLCampaignClassification]:
        """Obtém classificações por tier."""
        query = select(MLCampaignClassification).where(
            and_(
                MLCampaignClassification.config_id == config_id,
                MLCampaignClassification.valid_until > datetime.utcnow(),
            )
        )
        if tier:
            query = query.where(MLCampaignClassification.tier == tier)

        result = await self.session.execute(query)
        return result.scalars().all()

    def _apply_classification_filters(
        self,
        query,
        config_id: int,
        campaign_id: Optional[str] = None,
        tier: Optional[CampaignTier] = None,
        min_confidence: Optional[float] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ):
        query = query.where(MLCampaignClassification.config_id == config_id)
        if campaign_id:
            query = query.where(
                MLCampaignClassification.campaign_id == campaign_id
            )
        if tier:
            query = query.where(MLCampaignClassification.tier == tier)
        if min_confidence is not None:
            query = query.where(
                MLCampaignClassification.confidence_score >= min_confidence
            )
        if start_date:
            query = query.where(
                MLCampaignClassification.classified_at >= start_date
            )
        if end_date:
            query = query.where(
                MLCampaignClassification.classified_at <= end_date
            )
        return query

    async def get_classifications(
        self,
        config_id: int,
        campaign_id: Optional[str] = None,
        tier: Optional[CampaignTier] = None,
        min_confidence: Optional[float] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Sequence[MLCampaignClassification]:
        """Obtém classificações com filtros e paginacao."""
        query = select(MLCampaignClassification)
        query = self._apply_classification_filters(
            query,
            config_id=config_id,
            campaign_id=campaign_id,
            tier=tier,
            min_confidence=min_confidence,
            start_date=start_date,
            end_date=end_date,
        )
        query = query.order_by(desc(MLCampaignClassification.classified_at))
        if offset:
            query = query.offset(offset)
        if limit:
            query = query.limit(limit)

        result = await self.session.execute(query)
        return result.scalars().all()

    async def count_classifications(
        self,
        config_id: int,
        campaign_id: Optional[str] = None,
        tier: Optional[CampaignTier] = None,
        min_confidence: Optional[float] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> int:
        """Conta classificacoes com filtros."""
        query = select(func.count(MLCampaignClassification.id))
        query = self._apply_classification_filters(
            query,
            config_id=config_id,
            campaign_id=campaign_id,
            tier=tier,
            min_confidence=min_confidence,
            start_date=start_date,
            end_date=end_date,
        )
        result = await self.session.execute(query)
        return int(result.scalar() or 0)

    async def get_classification_counts_by_tier(
        self,
        config_id: int,
        campaign_id: Optional[str] = None,
        min_confidence: Optional[float] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> dict[str, int]:
        """Obtém contagem por tier com filtros."""
        query = select(
            MLCampaignClassification.tier,
            func.count(MLCampaignClassification.id),
        )
        query = self._apply_classification_filters(
            query,
            config_id=config_id,
            campaign_id=campaign_id,
            tier=None,
            min_confidence=min_confidence,
            start_date=start_date,
            end_date=end_date,
        )
        query = query.group_by(MLCampaignClassification.tier)
        result = await self.session.execute(query)
        counts = {}
        for tier_val, count in result.all():
            if tier_val:
                counts[tier_val.value] = int(count)
        return counts

    # ==================== RECOMMENDATIONS ====================

    async def create_recommendation(
        self,
        config_id: int,
        entity_type: str,
        entity_id: str,
        recommendation_type: RecommendationType,
        title: str,
        description: str,
        priority: int = 5,
        suggested_action: Optional[dict] = None,
        confidence_score: float = 0.5,
        reasoning: Optional[dict] = None,
        expires_at: Optional[datetime] = None,
    ) -> MLRecommendation:
        """Cria nova recomendação."""
        recommendation = MLRecommendation(
            config_id=config_id,
            entity_type=entity_type,
            entity_id=entity_id,
            recommendation_type=recommendation_type,
            title=title,
            description=description,
            priority=priority,
            suggested_action=suggested_action,
            confidence_score=confidence_score,
            reasoning=reasoning,
            expires_at=expires_at or (datetime.utcnow() + timedelta(days=7)),
        )
        self.session.add(recommendation)
        await self.session.flush()
        return recommendation

    async def get_recommendation(
        self,
        recommendation_id: int,
    ) -> Optional[MLRecommendation]:
        """Obtém recomendacao por ID."""
        result = await self.session.execute(
            select(MLRecommendation).where(MLRecommendation.id == recommendation_id)
        )
        return result.scalar_one_or_none()

    async def get_recommendations(
        self,
        config_id: int,
        entity_type: Optional[str] = None,
        entity_id: Optional[str] = None,
        recommendation_type: Optional[RecommendationType] = None,
        active_only: bool = False,
        status: Optional[str] = None,
        min_priority: Optional[int] = None,
        is_active: Optional[bool] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Sequence[MLRecommendation]:
        """Obtém recomendacoes com filtros."""
        query = select(MLRecommendation).where(
            MLRecommendation.config_id == config_id
        )

        if entity_type:
            query = query.where(MLRecommendation.entity_type == entity_type)
        if entity_id:
            query = query.where(MLRecommendation.entity_id == entity_id)
        if recommendation_type:
            query = query.where(
                MLRecommendation.recommendation_type == recommendation_type
            )
        if min_priority is not None:
            query = query.where(MLRecommendation.priority >= min_priority)
        if start_date:
            query = query.where(MLRecommendation.created_at >= start_date)
        if end_date:
            query = query.where(MLRecommendation.created_at <= end_date)

        now = datetime.utcnow()
        if status:
            status = status.lower()
            if status == "active":
                query = query.where(
                    and_(
                        MLRecommendation.is_active == True,
                        MLRecommendation.dismissed == False,
                        or_(
                            MLRecommendation.expires_at.is_(None),
                            MLRecommendation.expires_at > now,
                        ),
                    )
                )
            elif status == "expired":
                query = query.where(
                    and_(
                        MLRecommendation.dismissed == False,
                        MLRecommendation.was_applied == False,
                        MLRecommendation.expires_at.is_not(None),
                        MLRecommendation.expires_at <= now,
                    )
                )
            elif status == "dismissed":
                query = query.where(MLRecommendation.dismissed == True)
            elif status == "applied":
                query = query.where(MLRecommendation.was_applied == True)
        elif active_only:
            query = query.where(
                and_(
                    MLRecommendation.is_active == True,
                    MLRecommendation.dismissed == False,
                    or_(
                        MLRecommendation.expires_at.is_(None),
                        MLRecommendation.expires_at > now,
                    ),
                )
            )
        elif is_active is not None:
            query = query.where(MLRecommendation.is_active == is_active)

        query = query.order_by(
            desc(MLRecommendation.priority),
            desc(MLRecommendation.created_at),
        )
        if offset:
            query = query.offset(offset)
        if limit:
            query = query.limit(limit)

        result = await self.session.execute(query)
        return result.scalars().all()

    async def count_recommendations(
        self,
        config_id: int,
        entity_type: Optional[str] = None,
        entity_id: Optional[str] = None,
        recommendation_type: Optional[RecommendationType] = None,
        status: Optional[str] = None,
        min_priority: Optional[int] = None,
        is_active: Optional[bool] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> int:
        """Conta recomendacoes com filtros."""
        query = select(func.count(MLRecommendation.id)).where(
            MLRecommendation.config_id == config_id
        )

        if entity_type:
            query = query.where(MLRecommendation.entity_type == entity_type)
        if entity_id:
            query = query.where(MLRecommendation.entity_id == entity_id)
        if recommendation_type:
            query = query.where(
                MLRecommendation.recommendation_type == recommendation_type
            )
        if min_priority is not None:
            query = query.where(MLRecommendation.priority >= min_priority)
        if start_date:
            query = query.where(MLRecommendation.created_at >= start_date)
        if end_date:
            query = query.where(MLRecommendation.created_at <= end_date)

        now = datetime.utcnow()
        if status:
            status = status.lower()
            if status == "active":
                query = query.where(
                    and_(
                        MLRecommendation.is_active == True,
                        MLRecommendation.dismissed == False,
                        or_(
                            MLRecommendation.expires_at.is_(None),
                            MLRecommendation.expires_at > now,
                        ),
                    )
                )
            elif status == "expired":
                query = query.where(
                    and_(
                        MLRecommendation.dismissed == False,
                        MLRecommendation.was_applied == False,
                        MLRecommendation.expires_at.is_not(None),
                        MLRecommendation.expires_at <= now,
                    )
                )
            elif status == "dismissed":
                query = query.where(MLRecommendation.dismissed == True)
            elif status == "applied":
                query = query.where(MLRecommendation.was_applied == True)
        elif is_active is not None:
            query = query.where(MLRecommendation.is_active == is_active)

        result = await self.session.execute(query)
        return int(result.scalar() or 0)

    async def get_active_recommendations(
        self,
        config_id: int,
        entity_type: Optional[str] = None,
        entity_id: Optional[str] = None,
        recommendation_type: Optional[RecommendationType] = None,
        limit: int = 50,
    ) -> Sequence[MLRecommendation]:
        """Obtém recomendações ativas."""
        query = select(MLRecommendation).where(
            and_(
                MLRecommendation.config_id == config_id,
                MLRecommendation.is_active == True,
                MLRecommendation.dismissed == False,
                or_(
                    MLRecommendation.expires_at.is_(None),
                    MLRecommendation.expires_at > datetime.utcnow(),
                ),
            )
        )
        if entity_type:
            query = query.where(MLRecommendation.entity_type == entity_type)
        if entity_id:
            query = query.where(MLRecommendation.entity_id == entity_id)
        if recommendation_type:
            query = query.where(
                MLRecommendation.recommendation_type == recommendation_type
            )

        query = query.order_by(desc(MLRecommendation.priority)).limit(limit)
        result = await self.session.execute(query)
        return result.scalars().all()

    async def recommendation_exists(
        self,
        config_id: int,
        entity_type: str,
        entity_id: str,
        recommendation_type: RecommendationType,
        since: datetime,
    ) -> bool:
        """Verifica se ja existe recomendacao recente para evitar duplicacao."""
        query = select(func.count(MLRecommendation.id)).where(
            and_(
                MLRecommendation.config_id == config_id,
                MLRecommendation.entity_type == entity_type,
                MLRecommendation.entity_id == entity_id,
                MLRecommendation.recommendation_type == recommendation_type,
                MLRecommendation.created_at >= since,
            )
        )
        result = await self.session.execute(query)
        return (result.scalar() or 0) > 0

    async def expire_recommendations(self, config_id: int) -> int:
        """Expira recomendacoes que passaram do prazo."""
        now = datetime.utcnow()
        result = await self.session.execute(
            update(MLRecommendation)
            .where(
                and_(
                    MLRecommendation.config_id == config_id,
                    MLRecommendation.is_active == True,
                    MLRecommendation.dismissed == False,
                    MLRecommendation.was_applied == False,
                    MLRecommendation.expires_at.is_not(None),
                    MLRecommendation.expires_at <= now,
                )
            )
            .values(is_active=False)
        )
        return int(result.rowcount or 0)

    async def dismiss_recommendation(
        self,
        recommendation_id: int,
        user_id: int,
        reason: Optional[str] = None,
    ) -> None:
        """Descarta uma recomendação."""
        await self.session.execute(
            update(MLRecommendation)
            .where(MLRecommendation.id == recommendation_id)
            .values(
                dismissed=True,
                dismissed_at=datetime.utcnow(),
                dismissed_by=user_id,
                dismissed_reason=reason,
                is_active=False,
            )
        )

    async def apply_recommendation(
        self,
        recommendation_id: int,
        user_id: int,
    ) -> None:
        """Marca recomendação como aplicada."""
        await self.session.execute(
            update(MLRecommendation)
            .where(MLRecommendation.id == recommendation_id)
            .values(
                was_applied=True,
                applied_at=datetime.utcnow(),
                applied_by=user_id,
                is_active=False,
            )
        )

    # ==================== ANOMALIES ====================

    async def create_anomaly(
        self,
        config_id: int,
        entity_type: str,
        entity_id: str,
        anomaly_type: str,
        metric_name: str,
        observed_value: float,
        expected_value: float,
        deviation_score: float,
        severity: AnomalySeverity,
        anomaly_date: datetime,
        recommendation_id: Optional[int] = None,
    ) -> MLAnomaly:
        """Cria registro de anomalia."""
        anomaly = MLAnomaly(
            config_id=config_id,
            entity_type=entity_type,
            entity_id=entity_id,
            anomaly_type=anomaly_type,
            metric_name=metric_name,
            observed_value=observed_value,
            expected_value=expected_value,
            deviation_score=deviation_score,
            severity=severity,
            anomaly_date=anomaly_date,
            recommendation_id=recommendation_id,
        )
        self.session.add(anomaly)
        await self.session.flush()
        return anomaly

    async def get_anomalies(
        self,
        config_id: int,
        severity: Optional[AnomalySeverity] = None,
        acknowledged: Optional[bool] = None,
        days: int = 7,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        anomaly_type: Optional[str] = None,
        entity_id: Optional[str] = None,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> Sequence[MLAnomaly]:
        """Obtém anomalias recentes."""
        query = select(MLAnomaly).where(MLAnomaly.config_id == config_id)

        if start_date or end_date:
            if start_date:
                query = query.where(MLAnomaly.detected_at >= start_date)
            if end_date:
                query = query.where(MLAnomaly.detected_at <= end_date)
        else:
            since = datetime.utcnow() - timedelta(days=days)
            query = query.where(MLAnomaly.detected_at >= since)

        if severity:
            query = query.where(MLAnomaly.severity == severity)
        if acknowledged is not None:
            query = query.where(MLAnomaly.is_acknowledged == acknowledged)
        if anomaly_type:
            query = query.where(MLAnomaly.anomaly_type == anomaly_type)
        if entity_id:
            query = query.where(MLAnomaly.entity_id == entity_id)

        query = query.order_by(desc(MLAnomaly.detected_at))
        if offset:
            query = query.offset(offset)
        if limit:
            query = query.limit(limit)
        result = await self.session.execute(query)
        return result.scalars().all()

    async def count_anomalies(
        self,
        config_id: int,
        severity: Optional[AnomalySeverity] = None,
        acknowledged: Optional[bool] = None,
        days: int = 7,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        anomaly_type: Optional[str] = None,
        entity_id: Optional[str] = None,
    ) -> int:
        """Conta anomalias com filtros."""
        query = select(func.count(MLAnomaly.id)).where(
            MLAnomaly.config_id == config_id
        )

        if start_date or end_date:
            if start_date:
                query = query.where(MLAnomaly.detected_at >= start_date)
            if end_date:
                query = query.where(MLAnomaly.detected_at <= end_date)
        else:
            since = datetime.utcnow() - timedelta(days=days)
            query = query.where(MLAnomaly.detected_at >= since)

        if severity:
            query = query.where(MLAnomaly.severity == severity)
        if acknowledged is not None:
            query = query.where(MLAnomaly.is_acknowledged == acknowledged)
        if anomaly_type:
            query = query.where(MLAnomaly.anomaly_type == anomaly_type)
        if entity_id:
            query = query.where(MLAnomaly.entity_id == entity_id)

        result = await self.session.execute(query)
        return int(result.scalar() or 0)

    async def acknowledge_anomaly(
        self,
        anomaly_id: int,
        user_id: int,
        notes: Optional[str] = None,
    ) -> None:
        """Marca anomalia como reconhecida."""
        await self.session.execute(
            update(MLAnomaly)
            .where(MLAnomaly.id == anomaly_id)
            .values(
                is_acknowledged=True,
                acknowledged_at=datetime.utcnow(),
                acknowledged_by=user_id,
                resolution_notes=notes,
            )
        )

    # ==================== FEATURES ====================

    async def feature_exists(
        self,
        config_id: int,
        campaign_id: str,
        window_days: int,
        feature_date: date,
    ) -> bool:
        """Verifica se features ja existem para data e campanha."""
        query = select(func.count(MLFeature.id)).where(
            and_(
                MLFeature.config_id == config_id,
                MLFeature.campaign_id == campaign_id,
                MLFeature.window_days == window_days,
                MLFeature.feature_date == feature_date,
            )
        )
        result = await self.session.execute(query)
        return (result.scalar() or 0) > 0

    async def create_feature(
        self,
        config_id: int,
        campaign_id: str,
        window_days: int,
        feature_date: date,
        features: Optional[dict],
        insufficient_data: bool = False,
    ) -> MLFeature:
        """Cria registro de features."""
        record = MLFeature(
            config_id=config_id,
            campaign_id=campaign_id,
            window_days=window_days,
            feature_date=feature_date,
            features=features,
            insufficient_data=insufficient_data,
        )
        self.session.add(record)
        await self.session.flush()
        return record

    # ==================== FORECASTS ====================

    async def forecast_exists(
        self,
        config_id: int,
        entity_type: str,
        entity_id: str,
        target_metric: str,
        horizon_days: int,
        forecast_date: date,
    ) -> bool:
        """Verifica se forecast ja existe para a data."""
        query = select(func.count(MLForecast.id)).where(
            and_(
                MLForecast.config_id == config_id,
                MLForecast.entity_type == entity_type,
                MLForecast.entity_id == entity_id,
                MLForecast.target_metric == target_metric,
                MLForecast.horizon_days == horizon_days,
                MLForecast.forecast_date == forecast_date,
            )
        )
        result = await self.session.execute(query)
        return (result.scalar() or 0) > 0

    async def create_forecast(
        self,
        config_id: int,
        entity_type: str,
        entity_id: str,
        target_metric: str,
        horizon_days: int,
        method: str,
        predictions: Optional[list[dict]],
        forecast_date: date,
        window_days: Optional[int] = None,
        model_version: Optional[str] = None,
        insufficient_data: bool = False,
    ) -> MLForecast:
        """Cria registro de forecast."""
        record = MLForecast(
            config_id=config_id,
            entity_type=entity_type,
            entity_id=entity_id,
            target_metric=target_metric,
            horizon_days=horizon_days,
            method=method,
            predictions=predictions,
            forecast_date=forecast_date,
            window_days=window_days,
            model_version=model_version,
            insufficient_data=insufficient_data,
        )
        self.session.add(record)
        await self.session.flush()
        return record

    async def get_forecasts(
        self,
        config_id: int,
        entity_type: Optional[str] = None,
        entity_id: Optional[str] = None,
        target_metric: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Sequence[MLForecast]:
        """Obtém forecasts com filtros."""
        query = select(MLForecast).where(MLForecast.config_id == config_id)
        if entity_type:
            query = query.where(MLForecast.entity_type == entity_type)
        if entity_id:
            query = query.where(MLForecast.entity_id == entity_id)
        if target_metric:
            query = query.where(MLForecast.target_metric == target_metric)
        if start_date:
            start = start_date.date() if isinstance(start_date, datetime) else start_date
            query = query.where(MLForecast.forecast_date >= start)
        if end_date:
            end = end_date.date() if isinstance(end_date, datetime) else end_date
            query = query.where(MLForecast.forecast_date <= end)

        query = query.order_by(desc(MLForecast.forecast_date))
        if offset:
            query = query.offset(offset)
        if limit:
            query = query.limit(limit)

        result = await self.session.execute(query)
        return result.scalars().all()

    async def count_forecasts(
        self,
        config_id: int,
        entity_type: Optional[str] = None,
        entity_id: Optional[str] = None,
        target_metric: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> int:
        """Conta forecasts com filtros."""
        query = select(func.count(MLForecast.id)).where(
            MLForecast.config_id == config_id
        )
        if entity_type:
            query = query.where(MLForecast.entity_type == entity_type)
        if entity_id:
            query = query.where(MLForecast.entity_id == entity_id)
        if target_metric:
            query = query.where(MLForecast.target_metric == target_metric)
        if start_date:
            start = start_date.date() if isinstance(start_date, datetime) else start_date
            query = query.where(MLForecast.forecast_date >= start)
        if end_date:
            end = end_date.date() if isinstance(end_date, datetime) else end_date
            query = query.where(MLForecast.forecast_date <= end)

        result = await self.session.execute(query)
        return int(result.scalar() or 0)

    # ==================== TRAINING JOBS ====================

    async def create_training_job(
        self,
        model_type: ModelType,
        config_id: Optional[int] = None,
        celery_task_id: Optional[str] = None,
    ) -> MLTrainingJob:
        """Cria job de treinamento."""
        job = MLTrainingJob(
            model_type=model_type,
            config_id=config_id,
            celery_task_id=celery_task_id,
            status=JobStatus.PENDING,
        )
        self.session.add(job)
        await self.session.flush()
        return job

    async def get_training_job(self, job_id: int) -> Optional[MLTrainingJob]:
        """Obtém job por ID."""
        result = await self.session.execute(
            select(MLTrainingJob).where(MLTrainingJob.id == job_id)
        )
        return result.scalar_one_or_none()

    async def update_job_status(
        self,
        job_id: int,
        status: JobStatus,
        progress: Optional[float] = None,
        model_id: Optional[int] = None,
        error_message: Optional[str] = None,
    ) -> None:
        """Atualiza status do job."""
        values = {"status": status}
        if progress is not None:
            values["progress"] = progress
        if model_id:
            values["model_id"] = model_id
        if error_message:
            values["error_message"] = error_message
        if status == JobStatus.RUNNING:
            values["started_at"] = datetime.utcnow()
        if status in [JobStatus.COMPLETED, JobStatus.FAILED]:
            values["completed_at"] = datetime.utcnow()

        await self.session.execute(
            update(MLTrainingJob)
            .where(MLTrainingJob.id == job_id)
            .values(**values)
        )
