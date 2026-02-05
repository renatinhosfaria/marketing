"""Tests for ML training tasks."""
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime

from projects.ml.jobs.training_tasks import train_campaign_classifier


class TestTrainCampaignClassifier:
    """Tests for train_campaign_classifier task."""

    @patch('shared.db.session.create_isolated_async_session_maker')
    @patch('projects.ml.jobs.training_tasks._train_classifier_for_config')
    def test_train_campaign_classifier_success(
        self,
        mock_train_func,
        mock_session_maker,
    ):
        """Should train classifier and return metrics."""
        # Arrange
        mock_engine = MagicMock()
        mock_engine.dispose = AsyncMock()
        mock_session_maker.return_value = (mock_engine, MagicMock())

        mock_train_func.return_value = {
            "model_id": 1,
            "samples_used": 50,
            "metrics": {"accuracy": 0.85, "f1_weighted": 0.82},
        }

        # Act
        result = train_campaign_classifier(config_id=1, entity_type="campaign")

        # Assert
        assert result["status"] == "success"
        assert result["model_id"] == 1
        assert result["metrics"]["accuracy"] == 0.85

    @patch('shared.db.session.create_isolated_async_session_maker')
    @patch('projects.ml.jobs.training_tasks._train_classifier_for_config')
    def test_train_campaign_classifier_insufficient_data(
        self,
        mock_train_func,
        mock_session_maker,
    ):
        """Should return error when insufficient data."""
        mock_engine = MagicMock()
        mock_engine.dispose = AsyncMock()
        mock_session_maker.return_value = (mock_engine, MagicMock())

        mock_train_func.return_value = {
            "status": "insufficient_data",
            "samples_available": 10,
            "samples_required": 30,
        }

        result = train_campaign_classifier(config_id=1)

        assert result["status"] == "insufficient_data"
