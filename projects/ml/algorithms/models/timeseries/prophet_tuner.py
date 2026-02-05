"""
Prophet hyperparameter tuning using cross-validation.
"""

from dataclasses import dataclass
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Optional
import json
import warnings

import numpy as np
import pandas as pd

from shared.core.logging import get_logger

logger = get_logger(__name__)

# Check Prophet availability
try:
    from prophet import Prophet
    from prophet.diagnostics import cross_validation, performance_metrics
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    logger.warning("Prophet not available for hyperparameter tuning")


@dataclass
class TuningResult:
    """Result of hyperparameter tuning.

    Attributes:
        best_params: Best hyperparameters found
        cv_metrics: Cross-validation metrics for each parameter combination
        mean_mape: Mean Absolute Percentage Error (0-100 scale, e.g., 5.0 means 5%)
        mean_rmse: Root Mean Squared Error (same units as the target metric)
        training_samples: Number of data points used for tuning
        tuned_at: Timestamp when tuning was completed
    """
    best_params: dict
    cv_metrics: list[dict]
    mean_mape: float  # Percentage scale: 0-100 (e.g., 5.0 = 5%)
    mean_rmse: float
    training_samples: int
    tuned_at: datetime


class ProphetTuner:
    """
    Hyperparameter tuning for Prophet models using cross-validation.

    Searches over a grid of parameters to find the best configuration
    for a given time series.
    """

    # Reduced grid for faster tuning (full grid takes too long)
    PARAM_GRID = {
        'changepoint_prior_scale': [0.01, 0.05, 0.1],
        'seasonality_prior_scale': [0.1, 1.0, 10.0],
    }

    MIN_DATA_POINTS = 30  # Minimum data points for tuning

    def __init__(self, metric: str = 'cpl'):
        """
        Initialize tuner.

        Args:
            metric: Name of the metric being tuned (for logging)
        """
        self.metric = metric
        self.best_params: Optional[dict] = None

    def tune(
        self,
        df: pd.DataFrame,
        horizon: str = '7 days',
        initial: str = '30 days',
        period: str = '7 days',
    ) -> TuningResult:
        """
        Find best Prophet parameters using time series cross-validation.

        Args:
            df: DataFrame with 'ds' and 'y' columns
            horizon: Forecast horizon for CV
            initial: Initial training period
            period: Period between cutoff dates

        Returns:
            TuningResult with best params and metrics

        Raises:
            ValueError: If insufficient data or Prophet unavailable
        """
        if not PROPHET_AVAILABLE:
            raise ValueError("Prophet is not installed")

        if len(df) < self.MIN_DATA_POINTS:
            raise ValueError(
                f"Insufficient data for tuning. Need {self.MIN_DATA_POINTS}, got {len(df)}"
            )

        # Ensure correct column names
        if 'ds' not in df.columns or 'y' not in df.columns:
            raise ValueError("DataFrame must have 'ds' and 'y' columns")

        best_mape = float('inf')
        best_params = {}
        all_results = []

        # Generate parameter combinations
        param_combinations = [
            dict(zip(self.PARAM_GRID.keys(), v))
            for v in product(*self.PARAM_GRID.values())
        ]

        logger.info(
            "Starting Prophet hyperparameter tuning",
            metric=self.metric,
            combinations=len(param_combinations),
            data_points=len(df),
        )

        for params in param_combinations:
            try:
                mape, rmse, mae = self._evaluate_params(
                    df, params, horizon, initial, period
                )

                all_results.append({
                    'params': params,
                    'mape': mape,
                    'rmse': rmse,
                    'mae': mae,
                })

                if mape < best_mape:
                    best_mape = mape
                    best_params = params
                    logger.debug(
                        "New best params found",
                        params=params,
                        mape=f"{mape:.2f}%",
                    )

            except Exception as e:
                logger.warning(
                    f"CV failed for params {params}: {e}"
                )
                continue

        if not best_params:
            # Fallback to defaults if all combinations failed
            best_params = {
                'changepoint_prior_scale': 0.05,
                'seasonality_prior_scale': 1.0,
            }
            logger.warning("All CV combinations failed, using defaults")

        self.best_params = best_params

        logger.info(
            "Prophet tuning completed",
            metric=self.metric,
            best_params=best_params,
            best_mape=f"{best_mape:.2f}%",
        )

        return TuningResult(
            best_params=best_params,
            cv_metrics=all_results,
            mean_mape=best_mape if best_mape != float('inf') else 0.0,
            mean_rmse=min((r['rmse'] for r in all_results), default=0.0),
            training_samples=len(df),
            tuned_at=datetime.utcnow(),
        )

    def _evaluate_params(
        self,
        df: pd.DataFrame,
        params: dict,
        horizon: str,
        initial: str,
        period: str,
    ) -> tuple[float, float, float]:
        """
        Evaluate a parameter combination using cross-validation.

        Returns:
            Tuple of (mape, rmse, mae) where:
            - mape: Mean Absolute Percentage Error on 0-100 scale (e.g., 5.0 = 5%)
            - rmse: Root Mean Squared Error (same units as target)
            - mae: Mean Absolute Error (same units as target)
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            model = Prophet(
                yearly_seasonality=len(df) >= 365,
                weekly_seasonality=True,
                daily_seasonality=False,
                **params
            )

            model.fit(df)

            df_cv = cross_validation(
                model,
                initial=initial,
                period=period,
                horizon=horizon,
            )
            df_metrics = performance_metrics(df_cv)

            return (
                df_metrics['mape'].mean() * 100,  # Convert to percentage
                df_metrics['rmse'].mean(),
                df_metrics['mae'].mean(),
            )

    def save_params(self, path: Path) -> None:
        """
        Save tuned parameters to JSON file.

        Args:
            path: Path to save parameters
        """
        data = {
            'best_params': self.best_params,
            'metric': self.metric,
            'tuned_at': datetime.utcnow().isoformat(),
        }

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info("Prophet params saved", path=str(path))

    @classmethod
    def load_params(cls, path: Path) -> Optional[dict]:
        """
        Load tuned parameters from JSON file.

        Args:
            path: Path to load parameters from

        Returns:
            Best parameters dict or None if file doesn't exist
        """
        path = Path(path)
        if not path.exists():
            return None

        with open(path) as f:
            data = json.load(f)
            return data.get('best_params')


def get_prophet_tuner(metric: str = 'cpl') -> ProphetTuner:
    """
    Factory function to create ProphetTuner instances.

    Args:
        metric: Metric name for logging

    Returns:
        New ProphetTuner instance
    """
    return ProphetTuner(metric=metric)
