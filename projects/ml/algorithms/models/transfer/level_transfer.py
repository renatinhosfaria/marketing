"""
Transfer Learning Module for Cross-Level Classification.

Enables training a global model on campaign data and applying it to classify
new adsets/ads with a confidence penalty.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import joblib

from projects.ml.algorithms.models.classification.campaign_classifier import (
    CampaignClassifier,
    ClassificationResult,
    create_training_labels,
    get_classifier,
)
from projects.ml.services.feature_engineering import CampaignFeatures, EntityFeatures
from projects.ml.db.models import CampaignTier, ModelType
from shared.config import settings
from shared.core.logging import get_logger

logger = get_logger(__name__)


# Confidence penalty for transfer learning predictions (15% reduction)
CONFIDENCE_PENALTY = 0.85


@dataclass
class TransferClassificationResult:
    """Result from transfer learning classification."""
    entity_id: str
    entity_type: str
    config_id: int
    tier: CampaignTier
    confidence_score: float
    probabilities: dict[str, float]
    feature_importances: dict[str, float]
    metrics_snapshot: dict
    is_transfer: bool = True


class LevelTransferLearning:
    """
    Transfer Learning for cross-level classification.

    Trains a global model on campaign data, then applies it to classify
    new adsets/ads with a confidence penalty because the model was trained
    on a different entity level.
    """

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize transfer learning module.

        Args:
            model_path: Optional path to a pre-trained global model
        """
        self._classifier: Optional[CampaignClassifier] = None
        self._is_trained = False
        self._model_version = "1.0.0-transfer"
        self._avg_cpl: float = 50.0
        self._avg_ctr: float = 2.0

        if model_path and Path(model_path).exists():
            self._load_model(model_path)

    async def train_global_model(
        self,
        config_id: int,
        data_service,
        ml_repo,
    ) -> dict:
        """
        Train global model on all campaigns for a config.

        Args:
            config_id: Configuration ID to train for
            data_service: Service to fetch campaign features
            ml_repo: Repository to save model records

        Returns:
            Dict with training metrics
        """
        logger.info(
            "Starting global model training",
            config_id=config_id
        )

        # Fetch all campaign features
        features_list = await data_service.get_all_campaign_features(config_id)

        if not features_list or len(features_list) < 4:
            logger.warning(
                "Insufficient data for global model training",
                config_id=config_id,
                samples=len(features_list) if features_list else 0
            )
            return {
                "success": False,
                "error": "Insufficient training data",
                "samples": len(features_list) if features_list else 0,
            }

        # Get or calculate global stats
        try:
            global_stats = await data_service.get_global_stats(config_id)
            avg_cpl = global_stats.get("avg_cpl", 50.0)
            avg_ctr = global_stats.get("avg_ctr", 2.0)
        except Exception:
            # Calculate from features if service method unavailable
            global_stats = self._calculate_global_stats(features_list)
            avg_cpl = global_stats["avg_cpl"]
            avg_ctr = global_stats["avg_ctr"]

        self._avg_cpl = avg_cpl
        self._avg_ctr = avg_ctr

        # Create training labels using heuristics
        labels = create_training_labels(features_list, avg_cpl)

        # Initialize and train classifier
        self._classifier = get_classifier()

        try:
            metrics = self._classifier.train(
                features_list=features_list,
                labels=labels,
                avg_cpl=avg_cpl,
                avg_ctr=avg_ctr,
                test_size=0.2,
            )
        except Exception as e:
            logger.error(
                "Failed to train global model",
                config_id=config_id,
                error=str(e)
            )
            return {
                "success": False,
                "error": str(e),
            }

        self._is_trained = True

        # Save model to disk
        model_dir = Path(settings.models_storage_path) / "transfer"
        model_dir.mkdir(parents=True, exist_ok=True)

        model_path = model_dir / f"global_{config_id}.joblib"

        model_data = {
            "classifier_model": self._classifier.model,
            "classifier_scaler": self._classifier.scaler,
            "classifier_label_encoder": self._classifier.label_encoder,
            "avg_cpl": self._avg_cpl,
            "avg_ctr": self._avg_ctr,
            "version": self._model_version,
            "config_id": config_id,
            "trained_at": datetime.utcnow().isoformat(),
            "samples_count": len(features_list),
        }

        joblib.dump(model_data, model_path)
        logger.info("Global transfer model saved", path=str(model_path))

        # Register model in database
        try:
            await ml_repo.save_model_record(
                name=f"global_transfer_{config_id}",
                model_type=ModelType.CAMPAIGN_CLASSIFIER,
                version=self._model_version,
                config_id=config_id,
                model_path=str(model_path),
                training_metrics=metrics,
                samples_count=len(features_list),
                status="READY",
            )
        except Exception as e:
            logger.warning(
                "Failed to save model record to database",
                error=str(e)
            )

        logger.info(
            "Global model training completed",
            config_id=config_id,
            accuracy=metrics.get("accuracy"),
            samples=len(features_list)
        )

        return {
            "success": True,
            "metrics": metrics,
            "samples": len(features_list),
            "model_path": str(model_path),
            **metrics,
        }

    def classify_new_entity(
        self,
        entity_features: Union[EntityFeatures, CampaignFeatures],
        avg_cpl: float,
        avg_ctr: float,
    ) -> TransferClassificationResult:
        """
        Classify a new entity using the global model with confidence penalty.

        Args:
            entity_features: Features of the entity to classify
            avg_cpl: Average CPL for normalization
            avg_ctr: Average CTR for normalization

        Returns:
            TransferClassificationResult with reduced confidence
        """
        # Ensure we have CampaignFeatures format
        campaign_features = self._ensure_campaign_features(entity_features)

        # Determine entity type and id
        if isinstance(entity_features, EntityFeatures):
            entity_type = entity_features.entity_type
            entity_id = entity_features.entity_id
        else:
            entity_type = "campaign"
            entity_id = entity_features.campaign_id

        # Use classifier if trained, otherwise use rules-based fallback
        if self._is_trained and self._classifier is not None:
            result = self._classifier.classify(
                campaign_features=campaign_features,
                avg_cpl=avg_cpl,
                avg_ctr=avg_ctr,
            )
        else:
            # Use a new classifier instance for rules-based classification
            classifier = get_classifier()
            result = classifier.classify_by_rules(
                campaign_features=campaign_features,
                avg_cpl=avg_cpl,
                avg_ctr=avg_ctr,
            )

        # Apply confidence penalty for transfer learning
        penalized_confidence = result.confidence_score * CONFIDENCE_PENALTY

        # Also adjust probabilities
        penalized_probabilities = {}
        for tier, prob in result.probabilities.items():
            if tier == result.tier.value:
                penalized_probabilities[tier] = prob * CONFIDENCE_PENALTY
            else:
                # Distribute the reduced confidence to other tiers
                penalized_probabilities[tier] = prob + (
                    (result.confidence_score - penalized_confidence) /
                    max(len(result.probabilities) - 1, 1)
                )

        # Normalize probabilities to ensure they sum to 1.0
        total = sum(penalized_probabilities.values())
        if total > 0:
            penalized_probabilities = {k: v / total for k, v in penalized_probabilities.items()}

        return TransferClassificationResult(
            entity_id=entity_id,
            entity_type=entity_type,
            config_id=campaign_features.config_id,
            tier=result.tier,
            confidence_score=penalized_confidence,
            probabilities=penalized_probabilities,
            feature_importances=result.feature_importances,
            metrics_snapshot=result.metrics_snapshot,
            is_transfer=True,
        )

    def _calculate_global_stats(
        self,
        features_list: list[CampaignFeatures]
    ) -> dict:
        """
        Calculate global feature statistics from a list of features.

        Args:
            features_list: List of CampaignFeatures

        Returns:
            Dict with avg_cpl and avg_ctr
        """
        if not features_list:
            return {"avg_cpl": 50.0, "avg_ctr": 2.0}

        total_spend = sum(f.spend_7d for f in features_list)
        total_leads = sum(f.leads_7d for f in features_list)
        total_clicks = sum(f.clicks_7d for f in features_list)
        total_impressions = sum(f.impressions_7d for f in features_list)

        avg_cpl = total_spend / total_leads if total_leads > 0 else 50.0
        avg_ctr = (total_clicks / total_impressions * 100) if total_impressions > 0 else 2.0

        return {
            "avg_cpl": avg_cpl,
            "avg_ctr": avg_ctr,
            "total_spend": total_spend,
            "total_leads": total_leads,
        }

    def _ensure_campaign_features(
        self,
        features: Union[EntityFeatures, CampaignFeatures]
    ) -> CampaignFeatures:
        """
        Ensure features are in CampaignFeatures format.

        Args:
            features: Either EntityFeatures or CampaignFeatures

        Returns:
            CampaignFeatures instance
        """
        if isinstance(features, CampaignFeatures):
            return features

        return self._to_campaign_features(features)

    def _to_campaign_features(
        self,
        features: EntityFeatures
    ) -> CampaignFeatures:
        """
        Convert EntityFeatures to CampaignFeatures format.

        Maps the generic entity features to the campaign-specific format
        required by the classifier.

        Args:
            features: EntityFeatures instance

        Returns:
            CampaignFeatures with mapped fields
        """
        return CampaignFeatures(
            campaign_id=features.entity_id,
            config_id=features.config_id,
            spend_7d=features.spend_7d,
            impressions_7d=features.impressions_7d,
            clicks_7d=features.clicks_7d,
            leads_7d=features.leads_7d,
            cpl_7d=features.cpl_7d,
            ctr_7d=features.ctr_7d,
            cpc_7d=features.cpc_7d,
            conversion_rate_7d=features.conversion_rate_7d,
            cpl_trend=features.cpl_trend,
            leads_trend=features.leads_trend,
            spend_trend=features.spend_trend,
            ctr_trend=features.ctr_trend,
            cpl_14d=features.cpl_14d,
            leads_14d=features.leads_14d,
            cpl_30d=features.cpl_30d,
            leads_30d=features.leads_30d,
            avg_daily_spend_30d=features.avg_daily_spend_30d,
            cpl_std_7d=features.cpl_std_7d,
            leads_std_7d=features.leads_std_7d,
            best_day_of_week=features.best_day_of_week,
            worst_day_of_week=features.worst_day_of_week,
            frequency_7d=features.frequency_7d,
            reach_7d=features.reach_7d,
            days_with_leads_7d=features.days_with_leads_7d,
            days_active=features.days_active,
            is_active=features.is_active,
            has_budget=features.has_budget,
            computed_at=features.computed_at or datetime.now(),
        )

    def _load_model(self, model_path: str) -> None:
        """
        Load a pre-trained global model.

        Args:
            model_path: Path to the saved model file
        """
        try:
            model_data = joblib.load(model_path)

            self._classifier = get_classifier()
            self._classifier.model = model_data["classifier_model"]
            self._classifier.scaler = model_data["classifier_scaler"]
            self._classifier.label_encoder = model_data["classifier_label_encoder"]
            self._classifier.is_fitted = True

            self._avg_cpl = model_data.get("avg_cpl", 50.0)
            self._avg_ctr = model_data.get("avg_ctr", 2.0)
            self._model_version = model_data.get("version", "1.0.0-transfer")
            self._is_trained = True

            logger.info(
                "Global transfer model loaded",
                path=model_path,
                version=self._model_version
            )
        except Exception as e:
            logger.error(
                "Failed to load global transfer model",
                path=model_path,
                error=str(e)
            )
            raise


def get_level_transfer(model_path: Optional[str] = None) -> LevelTransferLearning:
    """
    Factory function to create LevelTransferLearning instances.

    Args:
        model_path: Optional path to a pre-trained model

    Returns:
        New LevelTransferLearning instance
    """
    return LevelTransferLearning(model_path=model_path)
