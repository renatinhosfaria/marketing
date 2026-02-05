"""Tests for LevelTransferLearning module."""
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import asdict

from projects.ml.services.feature_engineering import CampaignFeatures, EntityFeatures
from projects.ml.db.models import CampaignTier, ModelType


class TestLevelTransferLearning:
    """Tests for LevelTransferLearning class."""

    @pytest.fixture
    def sample_campaign_features(self) -> CampaignFeatures:
        """Generate sample campaign features."""
        return CampaignFeatures(
            campaign_id="campaign_123",
            config_id=1,
            spend_7d=500.0,
            impressions_7d=10000,
            clicks_7d=200,
            leads_7d=10,
            cpl_7d=50.0,
            ctr_7d=2.0,
            cpc_7d=2.5,
            conversion_rate_7d=5.0,
            cpl_trend=-5.0,
            leads_trend=10.0,
            spend_trend=5.0,
            ctr_trend=2.0,
            cpl_14d=55.0,
            leads_14d=18,
            cpl_30d=52.0,
            leads_30d=40,
            avg_daily_spend_30d=70.0,
            cpl_std_7d=8.0,
            leads_std_7d=2.0,
            best_day_of_week=2,
            worst_day_of_week=5,
            frequency_7d=1.5,
            reach_7d=6000,
            days_with_leads_7d=5,
            days_active=30,
            is_active=True,
            has_budget=True,
            computed_at=datetime.now(),
        )

    @pytest.fixture
    def sample_entity_features(self) -> EntityFeatures:
        """Generate sample adset entity features."""
        return EntityFeatures(
            entity_type="adset",
            entity_id="adset_456",
            config_id=1,
            parent_id="campaign_123",
            spend_7d=250.0,
            impressions_7d=5000,
            clicks_7d=100,
            leads_7d=5,
            cpl_7d=50.0,
            ctr_7d=2.0,
            cpc_7d=2.5,
            conversion_rate_7d=5.0,
            cpl_trend=-3.0,
            leads_trend=8.0,
            spend_trend=4.0,
            ctr_trend=1.5,
            cpl_14d=52.0,
            leads_14d=9,
            cpl_30d=51.0,
            leads_30d=20,
            avg_daily_spend_30d=35.0,
            cpl_std_7d=6.0,
            leads_std_7d=1.5,
            best_day_of_week=2,
            worst_day_of_week=5,
            frequency_7d=1.4,
            reach_7d=3000,
            days_with_leads_7d=4,
            days_active=25,
            is_active=True,
            has_budget=True,
            share_of_parent_spend=50.0,
            share_of_parent_leads=50.0,
            performance_vs_siblings=5.0,
        )

    @pytest.fixture
    def multiple_campaign_features(self) -> list[CampaignFeatures]:
        """Generate multiple campaign features for training."""
        features_list = []
        for i in range(20):
            features_list.append(
                CampaignFeatures(
                    campaign_id=f"campaign_{i}",
                    config_id=1,
                    spend_7d=100.0 + i * 50,
                    impressions_7d=2000 + i * 1000,
                    clicks_7d=40 + i * 20,
                    leads_7d=2 + i,
                    cpl_7d=50.0 + (i % 5) * 10 - 20,
                    ctr_7d=2.0 + (i % 3) * 0.5,
                    cpc_7d=2.5 + (i % 4) * 0.3,
                    conversion_rate_7d=5.0 + (i % 4),
                    cpl_trend=-10 + (i % 5) * 5,
                    leads_trend=5 + (i % 3) * 3,
                    spend_trend=3 + (i % 4) * 2,
                    ctr_trend=1 + (i % 3),
                    cpl_14d=52.0 + (i % 5) * 5,
                    leads_14d=4 + i * 2,
                    cpl_30d=51.0 + (i % 5) * 4,
                    leads_30d=10 + i * 4,
                    avg_daily_spend_30d=35.0 + i * 5,
                    cpl_std_7d=5.0 + (i % 3) * 2,
                    leads_std_7d=1.0 + (i % 4) * 0.5,
                    best_day_of_week=i % 7,
                    worst_day_of_week=(i + 3) % 7,
                    frequency_7d=1.2 + (i % 3) * 0.3,
                    reach_7d=1500 + i * 500,
                    days_with_leads_7d=3 + (i % 4),
                    days_active=20 + i,
                    is_active=True,
                    has_budget=True,
                    computed_at=datetime.now(),
                )
            )
        return features_list

    @pytest.mark.asyncio
    async def test_train_global_model(self, multiple_campaign_features):
        """Should train a global model on campaign data."""
        from projects.ml.algorithms.models.transfer.level_transfer import (
            LevelTransferLearning,
        )

        transfer = LevelTransferLearning()

        # Mock data_service to return campaign features
        mock_data_service = AsyncMock()
        mock_data_service.get_all_campaign_features = AsyncMock(
            return_value=multiple_campaign_features
        )
        mock_data_service.get_global_stats = AsyncMock(
            return_value={"avg_cpl": 50.0, "avg_ctr": 2.0}
        )

        # Mock ml_repo for saving model
        mock_ml_repo = AsyncMock()
        mock_ml_repo.save_model_record = AsyncMock(return_value=1)

        config_id = 1

        with patch("pathlib.Path.mkdir"):
            with patch("joblib.dump"):
                result = await transfer.train_global_model(
                    config_id=config_id,
                    data_service=mock_data_service,
                    ml_repo=mock_ml_repo,
                )

        # Verify result
        assert result["success"] is True
        assert "accuracy" in result.get("metrics", {}) or "samples" in result

        # Verify data service was called
        mock_data_service.get_all_campaign_features.assert_called_once_with(config_id)

        # Verify model record was saved
        mock_ml_repo.save_model_record.assert_called_once()
        call_kwargs = mock_ml_repo.save_model_record.call_args
        assert call_kwargs is not None

    def test_classify_new_entity_reduces_confidence(
        self, sample_entity_features, sample_campaign_features
    ):
        """Should reduce confidence when using transfer learning (15% penalty)."""
        from projects.ml.algorithms.models.transfer.level_transfer import (
            LevelTransferLearning,
            CONFIDENCE_PENALTY,
        )

        transfer = LevelTransferLearning()

        # Mock the classifier to return a high confidence result
        mock_classifier = MagicMock()
        mock_result = MagicMock()
        mock_result.tier = CampaignTier.HIGH_PERFORMER
        mock_result.confidence_score = 0.90
        mock_result.probabilities = {
            "HIGH_PERFORMER": 0.90,
            "MODERATE": 0.05,
            "LOW": 0.03,
            "UNDERPERFORMER": 0.02,
        }
        mock_result.feature_importances = {}
        mock_result.metrics_snapshot = {}
        mock_classifier.classify.return_value = mock_result

        transfer._classifier = mock_classifier
        transfer._is_trained = True

        avg_cpl = 50.0
        avg_ctr = 2.0

        result = transfer.classify_new_entity(
            entity_features=sample_entity_features,
            avg_cpl=avg_cpl,
            avg_ctr=avg_ctr,
        )

        # Verify confidence was reduced by 15%
        expected_confidence = 0.90 * CONFIDENCE_PENALTY
        assert result.confidence_score == pytest.approx(expected_confidence, rel=0.01)

        # Verify CONFIDENCE_PENALTY is 0.85 (15% reduction)
        assert CONFIDENCE_PENALTY == 0.85

    def test_to_campaign_features_conversion(self, sample_entity_features):
        """Should convert EntityFeatures to CampaignFeatures format."""
        from projects.ml.algorithms.models.transfer.level_transfer import (
            LevelTransferLearning,
        )

        transfer = LevelTransferLearning()

        campaign_features = transfer._to_campaign_features(sample_entity_features)

        assert isinstance(campaign_features, CampaignFeatures)
        assert campaign_features.campaign_id == sample_entity_features.entity_id
        assert campaign_features.config_id == sample_entity_features.config_id
        assert campaign_features.spend_7d == sample_entity_features.spend_7d
        assert campaign_features.cpl_7d == sample_entity_features.cpl_7d
        assert campaign_features.ctr_7d == sample_entity_features.ctr_7d
        assert campaign_features.leads_7d == sample_entity_features.leads_7d

    def test_ensure_campaign_features_passthrough(self, sample_campaign_features):
        """Should pass through CampaignFeatures unchanged."""
        from projects.ml.algorithms.models.transfer.level_transfer import (
            LevelTransferLearning,
        )

        transfer = LevelTransferLearning()

        result = transfer._ensure_campaign_features(sample_campaign_features)

        assert result is sample_campaign_features
        assert result.campaign_id == sample_campaign_features.campaign_id

    def test_ensure_campaign_features_converts_entity(self, sample_entity_features):
        """Should convert EntityFeatures to CampaignFeatures."""
        from projects.ml.algorithms.models.transfer.level_transfer import (
            LevelTransferLearning,
        )

        transfer = LevelTransferLearning()

        result = transfer._ensure_campaign_features(sample_entity_features)

        assert isinstance(result, CampaignFeatures)
        assert result.campaign_id == sample_entity_features.entity_id

    def test_calculate_global_stats(self, multiple_campaign_features):
        """Should calculate global feature statistics."""
        from projects.ml.algorithms.models.transfer.level_transfer import (
            LevelTransferLearning,
        )

        transfer = LevelTransferLearning()

        stats = transfer._calculate_global_stats(multiple_campaign_features)

        assert "avg_cpl" in stats
        assert "avg_ctr" in stats
        assert stats["avg_cpl"] > 0
        assert stats["avg_ctr"] > 0

    def test_factory_function_returns_instance(self):
        """Factory function should return new LevelTransferLearning instance."""
        from projects.ml.algorithms.models.transfer.level_transfer import (
            get_level_transfer,
            LevelTransferLearning,
        )

        instance = get_level_transfer()

        assert isinstance(instance, LevelTransferLearning)

    def test_factory_function_returns_new_instances(self):
        """Factory function should return new instances each time."""
        from projects.ml.algorithms.models.transfer.level_transfer import (
            get_level_transfer,
        )

        instance1 = get_level_transfer()
        instance2 = get_level_transfer()

        assert instance1 is not instance2

    def test_classify_without_trained_model_uses_fallback(self, sample_entity_features):
        """Should handle classification without trained model gracefully."""
        from projects.ml.algorithms.models.transfer.level_transfer import (
            LevelTransferLearning,
        )

        transfer = LevelTransferLearning()

        # Without training, should use fallback rules-based classification
        result = transfer.classify_new_entity(
            entity_features=sample_entity_features,
            avg_cpl=50.0,
            avg_ctr=2.0,
        )

        # Should still return a result with reduced confidence
        assert result is not None
        assert result.tier in list(CampaignTier)
        # Confidence should still be reduced by the penalty
        assert result.confidence_score <= 0.85  # Max with penalty applied
