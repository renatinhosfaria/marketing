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


class TestTrainClassifiersAll:
    """Tests for train_classifiers_all dispatcher task."""

    @patch('projects.ml.jobs.training_tasks.sync_engine')
    @patch('projects.ml.jobs.training_tasks.train_campaign_classifier')
    @patch('sqlalchemy.orm.sessionmaker')
    def test_dispatches_tasks_for_active_configs(
        self,
        mock_sessionmaker,
        mock_train_task,
        mock_engine,
    ):
        """Should dispatch training tasks for all active configs."""
        from projects.ml.jobs.training_tasks import train_classifiers_all

        # Mock the session and query
        mock_session = MagicMock()
        mock_config1 = MagicMock(id=1, name="Config 1")
        mock_config2 = MagicMock(id=2, name="Config 2")
        mock_session.query.return_value.filter.return_value.all.return_value = [
            mock_config1, mock_config2
        ]
        mock_sessionmaker.return_value = lambda: mock_session

        result = train_classifiers_all()

        assert result["status"] == "dispatched"
        assert result["configs_count"] == 2
        assert len(result["tasks"]) == 6  # 2 configs * 3 entity types


class TestTuneProphetForConfig:
    """Tests for tune_prophet_for_config task."""

    @patch('projects.ml.jobs.training_tasks.create_isolated_async_session_maker')
    @patch('projects.ml.jobs.training_tasks._tune_prophet_for_config')
    def test_tune_prophet_success(
        self,
        mock_tune_func,
        mock_session_maker,
    ):
        """Should tune Prophet and save params."""
        from projects.ml.jobs.training_tasks import tune_prophet_for_config

        mock_engine = MagicMock()
        mock_engine.dispose = AsyncMock()
        mock_session_maker.return_value = (mock_engine, MagicMock())

        mock_tune_func.return_value = {
            "status": "success",
            "entities_tuned": 5,
            "mean_mape_improvement": 12.5,
        }

        result = tune_prophet_for_config(config_id=1, metric='cpl')

        assert result["status"] == "success"
        assert result["entities_tuned"] == 5

    @patch('projects.ml.jobs.training_tasks.create_isolated_async_session_maker')
    @patch('projects.ml.jobs.training_tasks._tune_prophet_for_config')
    def test_tune_prophet_with_different_metric(
        self,
        mock_tune_func,
        mock_session_maker,
    ):
        """Should accept different metrics like leads and spend."""
        from projects.ml.jobs.training_tasks import tune_prophet_for_config

        mock_engine = MagicMock()
        mock_engine.dispose = AsyncMock()
        mock_session_maker.return_value = (mock_engine, MagicMock())

        mock_tune_func.return_value = {
            "entities_tuned": 3,
            "entities_skipped": 2,
            "by_entity_type": {"campaign": {"tuned": 1, "skipped": 1}},
        }

        result = tune_prophet_for_config(config_id=1, metric='leads')

        assert result["status"] == "success"
        assert result["entities_tuned"] == 3
        mock_tune_func.assert_called_once()


class TestTuneProphetAll:
    """Tests for tune_prophet_all dispatcher task."""

    @patch('projects.ml.jobs.training_tasks.sync_engine')
    @patch('projects.ml.jobs.training_tasks.tune_prophet_for_config')
    @patch('sqlalchemy.orm.sessionmaker')
    def test_dispatches_tasks_for_active_configs(
        self,
        mock_sessionmaker,
        mock_tune_task,
        mock_engine,
    ):
        """Should dispatch tuning tasks for all active configs."""
        from projects.ml.jobs.training_tasks import tune_prophet_all

        # Mock the session and query
        mock_session = MagicMock()
        mock_config1 = MagicMock(id=1, name="Config 1")
        mock_config2 = MagicMock(id=2, name="Config 2")
        mock_session.query.return_value.filter.return_value.all.return_value = [
            mock_config1, mock_config2
        ]
        mock_sessionmaker.return_value = lambda: mock_session

        result = tune_prophet_all(metric='cpl')

        assert result["status"] == "dispatched"
        assert result["configs_count"] == 2
        assert len(result["tasks"]) == 2  # One task per config
