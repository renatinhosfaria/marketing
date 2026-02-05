"""
Ensemble forecaster combining multiple prediction methods.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import pandas as pd

from projects.ml.algorithms.models.timeseries.forecaster import (
    TimeSeriesForecaster,
    ForecastResult,
    ForecastSeries,
    get_forecaster,
)
from shared.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class EnsembleResult:
    """Individual ensemble prediction result."""
    predicted_value: float
    confidence_lower: float
    confidence_upper: float
    method_weights: dict[str, float]
    individual_predictions: dict[str, float]


class EnsembleForecaster:
    """
    Combines Prophet, EMA, and Linear forecasts using weighted average.
    Weights are determined by historical performance (inverse MAPE).
    """

    METHODS = ['ema', 'linear']  # Prophet optional

    def __init__(self, include_prophet: bool = True):
        """
        Initialize ensemble forecaster.

        Args:
            include_prophet: Whether to include Prophet in ensemble
        """
        self.forecasters: dict[str, TimeSeriesForecaster] = {}
        self.weights: dict[str, float] = {}

        # Always include EMA and Linear
        self.forecasters['ema'] = get_forecaster(method='ema')
        self.forecasters['linear'] = get_forecaster(method='linear')

        # Try to add Prophet
        if include_prophet:
            try:
                prophet_forecaster = get_forecaster(method='prophet')
                self.forecasters['prophet'] = prophet_forecaster
            except Exception:
                logger.info("Prophet not available for ensemble")

    def calibrate_weights(
        self,
        historical_data: pd.DataFrame,
        metric: str,
        validation_days: int = 14,
    ) -> dict[str, float]:
        """
        Calibrate ensemble weights based on historical validation.

        Args:
            historical_data: Full historical data with 'date' and metric columns
            metric: Metric to forecast
            validation_days: Days to use for validation

        Returns:
            Dict of method -> weight
        """
        if len(historical_data) <= validation_days + 7:
            # Not enough data, use equal weights
            n_methods = len(self.forecasters)
            self.weights = {k: 1.0 / n_methods for k in self.forecasters}
            return self.weights

        # Split data
        train_data = historical_data.iloc[:-validation_days].copy()
        val_data = historical_data.iloc[-validation_days:].copy()

        mapes = {}
        for method_name, forecaster in self.forecasters.items():
            try:
                forecast = forecaster.forecast(
                    df=train_data,
                    metric=metric,
                    entity_type='validation',
                    entity_id='calibration',
                    horizon_days=validation_days,
                )

                validation = forecaster.validate_forecast(forecast, val_data)
                mapes[method_name] = (
                    validation.mape if validation.mape < float('inf') else 100.0
                )
            except Exception as e:
                logger.warning(f"Calibration failed for {method_name}: {e}")
                mapes[method_name] = 100.0

        # Convert MAPE to weights (inverse, normalized)
        inverse_mapes = {k: 1.0 / (v + 1.0) for k, v in mapes.items()}
        total = sum(inverse_mapes.values())

        self.weights = {k: v / total for k, v in inverse_mapes.items()}

        logger.info(
            "Ensemble weights calibrated",
            weights=self.weights,
            mapes=mapes,
        )

        return self.weights

    def forecast(
        self,
        df: pd.DataFrame,
        metric: str,
        entity_type: str,
        entity_id: str,
        horizon_days: int = 7,
    ) -> ForecastSeries:
        """
        Generate ensemble forecast combining multiple methods.

        Args:
            df: Historical data
            metric: Metric to forecast
            entity_type: Entity type
            entity_id: Entity ID
            horizon_days: Forecast horizon

        Returns:
            ForecastSeries with ensemble predictions
        """
        # Get predictions from each method
        predictions: dict[str, ForecastSeries] = {}

        for method_name, forecaster in self.forecasters.items():
            try:
                result = forecaster.forecast(
                    df, metric, entity_type, entity_id, horizon_days
                )
                predictions[method_name] = result
            except Exception as e:
                logger.warning(f"Forecast failed for {method_name}: {e}")
                continue

        if not predictions:
            raise ValueError("All forecast methods failed")

        # Use equal weights if not calibrated
        if not self.weights:
            self.weights = {k: 1.0 / len(predictions) for k in predictions}

        # Combine forecasts
        ensemble_forecasts = []

        for i in range(horizon_days):
            weighted_sum = 0.0
            lower_sum = 0.0
            upper_sum = 0.0
            total_weight = 0.0
            individual = {}

            for method_name, forecast_series in predictions.items():
                if i < len(forecast_series.forecasts):
                    weight = self.weights.get(method_name, 0)
                    f = forecast_series.forecasts[i]

                    weighted_sum += f.predicted_value * weight
                    lower_sum += f.confidence_lower * weight
                    upper_sum += f.confidence_upper * weight
                    total_weight += weight
                    individual[method_name] = f.predicted_value

            if total_weight > 0:
                # Get forecast date from first available method
                first_method = list(predictions.keys())[0]
                forecast_date = predictions[first_method].forecasts[i].forecast_date

                ensemble_forecasts.append(ForecastResult(
                    entity_type=entity_type,
                    entity_id=entity_id,
                    metric=metric,
                    forecast_date=forecast_date,
                    predicted_value=weighted_sum / total_weight,
                    confidence_lower=lower_sum / total_weight,
                    confidence_upper=upper_sum / total_weight,
                    confidence_level=0.95,
                    method='ensemble',
                    model_version='1.0.0',
                ))

        # Prepare historical data
        historical = df.tail(14).to_dict('records')
        for h in historical:
            if 'date' in h and hasattr(h['date'], 'isoformat'):
                h['date'] = h['date'].isoformat()

        return ForecastSeries(
            entity_type=entity_type,
            entity_id=entity_id,
            metric=metric,
            forecasts=ensemble_forecasts,
            historical=historical,
            method='ensemble',
            created_at=datetime.utcnow(),
        )


def get_ensemble_forecaster(include_prophet: bool = True) -> EnsembleForecaster:
    """Factory function for EnsembleForecaster."""
    return EnsembleForecaster(include_prophet=include_prophet)
