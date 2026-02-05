"""Tests for Ensemble Forecaster."""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class TestEnsembleForecaster:
    """Tests for EnsembleForecaster class."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample time series data."""
        dates = pd.date_range(start='2025-01-01', periods=45, freq='D')
        np.random.seed(42)
        values = 50 + np.random.randn(45) * 10
        return pd.DataFrame({'date': dates, 'cpl': values})

    def test_forecast_combines_methods(self, sample_data):
        """Should combine predictions from multiple methods."""
        from projects.ml.algorithms.models.timeseries.ensemble_forecaster import (
            EnsembleForecaster
        )

        forecaster = EnsembleForecaster()
        result = forecaster.forecast(
            df=sample_data,
            metric='cpl',
            entity_type='campaign',
            entity_id='123',
            horizon_days=7,
        )

        assert len(result.forecasts) == 7
        assert result.method == 'ensemble'
        assert all(f.predicted_value >= 0 for f in result.forecasts)

    def test_calibrate_weights(self, sample_data):
        """Should calibrate weights based on validation performance."""
        from projects.ml.algorithms.models.timeseries.ensemble_forecaster import (
            EnsembleForecaster
        )

        forecaster = EnsembleForecaster()
        weights = forecaster.calibrate_weights(
            historical_data=sample_data,
            metric='cpl',
            validation_days=7,
        )

        assert sum(weights.values()) == pytest.approx(1.0, rel=0.01)
        assert all(w >= 0 for w in weights.values())
