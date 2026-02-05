"""Tests for Prophet hyperparameter tuning."""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock


class TestProphetTuner:
    """Tests for ProphetTuner class."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample time series data."""
        dates = pd.date_range(start='2025-01-01', periods=60, freq='D')
        np.random.seed(42)
        values = 50 + np.random.randn(60) * 10 + np.sin(np.arange(60) * 2 * np.pi / 7) * 5
        return pd.DataFrame({'ds': dates, 'y': values})

    def test_tune_returns_best_params(self, sample_data):
        """Should return best parameters after tuning."""
        from projects.ml.algorithms.models.timeseries.prophet_tuner import ProphetTuner

        tuner = ProphetTuner(metric='cpl')
        result = tuner.tune(
            df=sample_data,
            horizon='7 days',
            initial='30 days',
            period='7 days',
        )

        assert result.best_params is not None
        assert 'changepoint_prior_scale' in result.best_params
        assert result.mean_mape >= 0
        assert result.training_samples == 60

    def test_tune_with_insufficient_data_raises(self):
        """Should raise error with insufficient data."""
        from projects.ml.algorithms.models.timeseries.prophet_tuner import ProphetTuner

        small_data = pd.DataFrame({
            'ds': pd.date_range(start='2025-01-01', periods=10),
            'y': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        })

        tuner = ProphetTuner(metric='cpl')

        with pytest.raises(ValueError, match="(?i)insufficient"):
            tuner.tune(df=small_data)

    def test_save_and_load_params(self, sample_data, tmp_path):
        """Should save and load tuned parameters."""
        from projects.ml.algorithms.models.timeseries.prophet_tuner import ProphetTuner

        tuner = ProphetTuner(metric='cpl')
        tuner.best_params = {'changepoint_prior_scale': 0.05}

        param_path = tmp_path / "params.json"
        tuner.save_params(param_path)

        loaded = ProphetTuner.load_params(param_path)
        assert loaded == {'changepoint_prior_scale': 0.05}
