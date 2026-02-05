# Design: ML Phases 4-7 Implementation

**Date**: 2026-02-05
**Status**: Approved

## Overview

Implementation of advanced ML training capabilities for the Facebook Ads ML module:
- **Phase 4**: XGBoost Classifier Training
- **Phase 5**: Prophet Hyperparameter Tuning
- **Phase 6**: Isolation Forest Database Integration
- **Phase 7**: Ensemble Forecasting + Impact Analysis + Transfer Learning

## Phase 4: XGBoost Classifier Training

### Current State
- `CampaignClassifier.train()` method exists and works
- `create_training_labels()` provides heuristic auto-labeling
- Celery task `train_campaign_classifier` is a stub

### Implementation

#### 4.1 ClassificationService.train_classifier()
```python
# projects/ml/services/classification_service.py

async def train_classifier(
    self,
    config_id: int,
    min_samples: int = 30,
    entity_type: str = "campaign",
) -> dict:
    """
    Train XGBoost classifier for a config.

    Returns:
        Dict with model_id, metrics, samples_used
    """
    # 1. Get historical features (30-90 days)
    features_list = await self.data_service.get_all_entity_features(
        config_id=config_id,
        entity_type=entity_type,
        days=90,
    )

    if len(features_list) < min_samples:
        raise InsufficientDataError(f"Need {min_samples} samples, got {len(features_list)}")

    # 2. Get reference metrics
    avg_metrics = await self.data_service.get_aggregated_metrics(config_id)

    # 3. Create auto-labels from heuristics
    labels = create_training_labels(features_list, avg_metrics.avg_cpl)

    # 4. Train model
    classifier = CampaignClassifier()
    metrics = classifier.train(
        features_list=features_list,
        labels=labels,
        avg_cpl=avg_metrics.avg_cpl,
        avg_ctr=avg_metrics.avg_ctr,
    )

    # 5. Save model to filesystem
    model_path = self._get_model_path(config_id, entity_type)
    classifier.save(model_path)

    # 6. Register in ml_trained_models
    model_record = await self.ml_repo.create_trained_model(
        name=f"classifier_{entity_type}_{config_id}",
        model_type=ModelType.CLASSIFIER,
        config_id=config_id,
        model_path=str(model_path),
        training_metrics=metrics,
        feature_columns=CampaignClassifier.FEATURE_COLUMNS,
        samples_count=len(features_list),
    )

    return {
        "model_id": model_record.id,
        "metrics": metrics,
        "samples_used": len(features_list),
    }
```

#### 4.2 Celery Task Implementation
```python
# projects/ml/jobs/training_tasks.py

@celery_app.task(
    name="projects.ml.jobs.training_tasks.train_campaign_classifier",
    max_retries=2,
    soft_time_limit=900,   # 15 minutes
    time_limit=1200,       # 20 minutes
)
def train_campaign_classifier(config_id: int, entity_type: str = "campaign"):
    """Train XGBoost classifier for entity classification."""
    import asyncio
    from shared.db.session import create_isolated_async_session_maker

    logger.info("Starting classifier training", config_id=config_id, entity_type=entity_type)

    isolated_engine, isolated_session_maker = create_isolated_async_session_maker()

    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                _train_classifier_for_config(config_id, entity_type, isolated_session_maker)
            )
            loop.run_until_complete(isolated_engine.dispose())
        finally:
            loop.close()
            asyncio.set_event_loop(None)

        return result
    except Exception as e:
        logger.error("Classifier training failed", config_id=config_id, error=str(e))
        raise
```

#### 4.3 Celery Beat Schedule
```python
# Add to celery beat schedule
'train-classifiers-daily': {
    'task': 'projects.ml.jobs.training_tasks.train_classifiers_all',
    'schedule': crontab(hour=3, minute=0),  # 03:00 AM
}
```

---

## Phase 5: Prophet Hyperparameter Tuning

### Current State
- `TimeSeriesForecaster` uses Prophet with default params
- No cross-validation or param optimization
- Tasks `train_cpl_forecaster` and `train_leads_forecaster` are stubs

### Implementation

#### 5.1 ProphetTuner Class
```python
# projects/ml/algorithms/models/timeseries/prophet_tuner.py

@dataclass
class TuningResult:
    best_params: dict
    cv_metrics: dict  # mape, rmse, mae per fold
    mean_mape: float
    mean_rmse: float
    training_samples: int

class ProphetTuner:
    """
    Hyperparameter tuning for Prophet models using cross-validation.
    """

    PARAM_GRID = {
        'changepoint_prior_scale': [0.001, 0.01, 0.05, 0.1, 0.5],
        'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
        'seasonality_mode': ['additive', 'multiplicative'],
    }

    def __init__(self, metric: str = 'cpl'):
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
        """
        from prophet.diagnostics import cross_validation, performance_metrics
        from itertools import product

        best_mape = float('inf')
        best_params = {}
        all_results = []

        # Grid search
        param_combinations = [
            dict(zip(self.PARAM_GRID.keys(), v))
            for v in product(*self.PARAM_GRID.values())
        ]

        for params in param_combinations:
            model = Prophet(
                yearly_seasonality=len(df) >= 365,
                weekly_seasonality=True,
                daily_seasonality=False,
                **params
            )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(df)

                try:
                    df_cv = cross_validation(
                        model,
                        initial=initial,
                        period=period,
                        horizon=horizon
                    )
                    df_metrics = performance_metrics(df_cv)

                    mape = df_metrics['mape'].mean()
                    if mape < best_mape:
                        best_mape = mape
                        best_params = params

                    all_results.append({
                        'params': params,
                        'mape': mape,
                        'rmse': df_metrics['rmse'].mean(),
                        'mae': df_metrics['mae'].mean(),
                    })
                except Exception as e:
                    logger.warning(f"CV failed for params {params}: {e}")
                    continue

        self.best_params = best_params

        return TuningResult(
            best_params=best_params,
            cv_metrics=all_results,
            mean_mape=best_mape,
            mean_rmse=min(r['rmse'] for r in all_results) if all_results else 0,
            training_samples=len(df),
        )

    def save_params(self, path: Path) -> None:
        """Save tuned parameters to file."""
        import json
        with open(path, 'w') as f:
            json.dump({
                'best_params': self.best_params,
                'metric': self.metric,
                'tuned_at': datetime.utcnow().isoformat(),
            }, f, indent=2)

    @classmethod
    def load_params(cls, path: Path) -> Optional[dict]:
        """Load tuned parameters from file."""
        if not path.exists():
            return None
        with open(path) as f:
            data = json.load(f)
            return data.get('best_params')
```

#### 5.2 Updated Forecaster with Tuned Params
```python
# Update TimeSeriesForecaster._forecast_prophet()

def _forecast_prophet(
    self,
    df: pd.DataFrame,
    metric: str,
    entity_type: str,
    entity_id: str,
    horizon_days: int,
    tuned_params: Optional[dict] = None,  # NEW
) -> list[ForecastResult]:
    """Previsão usando Prophet with optional tuned params."""
    prophet_df = df.rename(columns={'date': 'ds', metric: 'y'})

    # Base config
    config = {
        'yearly_seasonality': len(df) >= 365,
        'weekly_seasonality': True,
        'daily_seasonality': False,
        'interval_width': self.confidence_level,
    }

    # Apply tuned params if available
    if tuned_params:
        config.update(tuned_params)
    else:
        config['changepoint_prior_scale'] = 0.05

    model = Prophet(**config)
    # ... rest of implementation
```

#### 5.3 Celery Tasks
```python
@celery_app.task(
    name="projects.ml.jobs.training_tasks.tune_prophet_for_config",
    max_retries=2,
    soft_time_limit=3600,  # 1 hour (tuning is slow)
    time_limit=4200,       # 1.2 hours
)
def tune_prophet_for_config(config_id: int, metric: str = 'cpl'):
    """Tune Prophet hyperparameters for a config."""
    # Implementation similar to train_anomaly_detector
    # Tunes for each entity with sufficient data
    pass
```

---

## Phase 6: Isolation Forest Database Integration

### Current State
- Training and model saving work (filesystem)
- Not integrated with `ml_trained_models` table
- No metrics persistence

### Implementation

#### 6.1 Integration with ml_trained_models
```python
# Update _train_isolation_forest_for_entity()

async def _train_isolation_forest_for_entity(...) -> dict:
    # ... existing training code ...

    if success:
        # Save to filesystem (existing)
        saved = detector.save_model(config_id, entity_type, entity_id)

        # NEW: Register in database
        model_record = await ml_repo.create_trained_model(
            name=f"isolation_forest_{entity_type}_{entity_id}",
            model_type=ModelType.ANOMALY_DETECTOR,
            config_id=config_id,
            model_path=str(model_path),
            parameters={
                'contamination': contamination,
                'n_estimators': 100,
            },
            feature_columns=detector.isolation_forest_features,
            training_metrics={
                'samples': len(df),
                'features_used': detector.isolation_forest_features,
            },
            samples_count=len(df),
        )

        return {
            "status": "success",
            "model_id": model_record.id,
            "samples": len(df),
            "features": detector.isolation_forest_features,
        }
```

#### 6.2 Model Loading Enhancement
```python
# Update AnomalyDetector.load_model()

def load_model(self, config_id: int, entity_type: str, entity_id: str) -> bool:
    """
    Load model with fallback:
    1. Check in-memory cache
    2. Check filesystem
    3. Check database for model path
    """
    # ... existing cache/filesystem logic ...

    # NEW: If not found, query database for model path
    if not model_path.exists():
        model_record = self._query_model_from_db(config_id, entity_type, entity_id)
        if model_record and Path(model_record.model_path).exists():
            model_path = Path(model_record.model_path)
        else:
            return False
```

---

## Phase 7: Ensemble + Impact Analysis + Transfer Learning

### 7.1 Ensemble Forecaster

```python
# projects/ml/algorithms/models/timeseries/ensemble_forecaster.py

@dataclass
class EnsembleResult:
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

    def __init__(self):
        self.forecasters = {
            'prophet': TimeSeriesForecaster(method='prophet'),
            'ema': TimeSeriesForecaster(method='ema'),
            'linear': TimeSeriesForecaster(method='linear'),
        }
        self.weights: dict[str, float] = {}

    def calibrate_weights(
        self,
        historical_data: pd.DataFrame,
        metric: str,
        validation_days: int = 14,
    ) -> dict[str, float]:
        """
        Calibrate ensemble weights based on historical validation.

        Args:
            historical_data: Full historical data
            metric: Metric to forecast
            validation_days: Days to use for validation

        Returns:
            Dict of method -> weight
        """
        # Split data: train on all but last N days, validate on last N
        train_data = historical_data.iloc[:-validation_days]
        val_data = historical_data.iloc[-validation_days:]

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
                mapes[method_name] = validation.mape if validation.mape < float('inf') else 100.0
            except Exception as e:
                logger.warning(f"Calibration failed for {method_name}: {e}")
                mapes[method_name] = 100.0  # High MAPE = low weight

        # Convert MAPE to weights (inverse, normalized)
        # Lower MAPE = higher weight
        inverse_mapes = {k: 1.0 / (v + 1.0) for k, v in mapes.items()}  # +1 to avoid div by 0
        total = sum(inverse_mapes.values())

        self.weights = {k: v / total for k, v in inverse_mapes.items()}
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
        Generate ensemble forecast.
        """
        # Get predictions from each method
        predictions = {}
        for method_name, forecaster in self.forecasters.items():
            try:
                result = forecaster.forecast(df, metric, entity_type, entity_id, horizon_days)
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
            weighted_sum = 0
            lower_sum = 0
            upper_sum = 0
            total_weight = 0

            for method_name, forecast_series in predictions.items():
                if i < len(forecast_series.forecasts):
                    weight = self.weights.get(method_name, 0)
                    f = forecast_series.forecasts[i]
                    weighted_sum += f.predicted_value * weight
                    lower_sum += f.confidence_lower * weight
                    upper_sum += f.confidence_upper * weight
                    total_weight += weight

            if total_weight > 0:
                ensemble_forecasts.append(ForecastResult(
                    entity_type=entity_type,
                    entity_id=entity_id,
                    metric=metric,
                    forecast_date=predictions[list(predictions.keys())[0]].forecasts[i].forecast_date,
                    predicted_value=weighted_sum / total_weight,
                    confidence_lower=lower_sum / total_weight,
                    confidence_upper=upper_sum / total_weight,
                    confidence_level=0.95,
                    method='ensemble',
                    model_version='1.0.0',
                ))

        return ForecastSeries(
            entity_type=entity_type,
            entity_id=entity_id,
            metric=metric,
            forecasts=ensemble_forecasts,
            historical=df.tail(14).to_dict('records'),
            method='ensemble',
            created_at=datetime.utcnow(),
        )
```

### 7.2 Impact Analysis (Causal Inference)

```python
# projects/ml/algorithms/models/causal/impact_analyzer.py

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Literal
from scipy import stats
import numpy as np
import pandas as pd

@dataclass
class ImpactResult:
    """Result of impact analysis."""
    entity_type: str
    entity_id: str
    change_date: datetime
    change_type: str  # budget_change, creative_change, audience_change, pause, reactivate

    # Metric changes (% change)
    metric_changes: dict[str, float]  # {'cpl': -15.2, 'leads': +20.5, 'ctr': +5.1}

    # Statistical significance
    significance: dict[str, float]  # {'cpl': 0.95, 'leads': 0.87}
    is_significant: dict[str, bool]  # {'cpl': True, 'leads': False}

    # Effect sizes
    effect_sizes: dict[str, float]  # Cohen's d

    # Verdict
    overall_impact: Literal['positive', 'negative', 'neutral', 'inconclusive']
    recommendation: str

    # Analysis metadata
    window_before: int
    window_after: int
    analyzed_at: datetime


class ImpactAnalyzer:
    """
    Analyzes the causal impact of changes on campaign performance.

    Uses before/after comparison with statistical testing and
    synthetic control (other entities as control group).
    """

    def __init__(
        self,
        significance_threshold: float = 0.05,
        min_effect_size: float = 0.2,  # Small effect (Cohen's d)
    ):
        self.significance_threshold = significance_threshold
        self.min_effect_size = min_effect_size

    async def analyze_impact(
        self,
        entity_type: str,
        entity_id: str,
        change_date: datetime,
        change_type: str,
        data_service,  # DataService instance
        config_id: int,
        window_before: int = 7,
        window_after: int = 7,
        control_entities: Optional[list[str]] = None,
    ) -> ImpactResult:
        """
        Analyze the impact of a change on entity performance.

        Args:
            entity_type: campaign, adset, or ad
            entity_id: ID of the entity that changed
            change_date: When the change occurred
            change_type: Type of change
            data_service: DataService for fetching data
            config_id: Config ID
            window_before: Days before change to analyze
            window_after: Days after change to analyze
            control_entities: Optional list of entity IDs to use as control

        Returns:
            ImpactResult with detailed analysis
        """
        # Get data for the entity
        start_date = change_date - timedelta(days=window_before)
        end_date = change_date + timedelta(days=window_after)

        df = await data_service.get_entity_daily_data(
            config_id=config_id,
            entity_type=entity_type,
            entity_id=entity_id,
            start_date=start_date,
            end_date=end_date,
        )

        if df.empty:
            raise ValueError(f"No data for entity {entity_id}")

        # Split into before and after periods
        df['date'] = pd.to_datetime(df['date'])
        before_df = df[df['date'] < change_date]
        after_df = df[df['date'] >= change_date]

        if len(before_df) < 3 or len(after_df) < 3:
            raise ValueError("Insufficient data for before/after comparison")

        # Calculate metrics for each period
        metrics_to_analyze = ['cpl', 'leads', 'ctr', 'spend']
        metric_changes = {}
        significance = {}
        is_significant = {}
        effect_sizes = {}

        for metric in metrics_to_analyze:
            if metric not in df.columns:
                continue

            before_values = before_df[metric].dropna()
            after_values = after_df[metric].dropna()

            if len(before_values) < 2 or len(after_values) < 2:
                continue

            before_mean = before_values.mean()
            after_mean = after_values.mean()

            # Percentage change
            if before_mean != 0:
                pct_change = ((after_mean - before_mean) / before_mean) * 100
            else:
                pct_change = float('inf') if after_mean > 0 else 0

            metric_changes[metric] = round(pct_change, 2)

            # T-test for significance
            t_stat, p_value = stats.ttest_ind(before_values, after_values)
            significance[metric] = round(1 - p_value, 3)
            is_significant[metric] = p_value < self.significance_threshold

            # Cohen's d effect size
            pooled_std = np.sqrt(
                ((len(before_values) - 1) * before_values.std() ** 2 +
                 (len(after_values) - 1) * after_values.std() ** 2) /
                (len(before_values) + len(after_values) - 2)
            )
            if pooled_std > 0:
                cohens_d = (after_mean - before_mean) / pooled_std
            else:
                cohens_d = 0
            effect_sizes[metric] = round(cohens_d, 3)

        # Determine overall impact
        overall_impact = self._determine_overall_impact(
            metric_changes, is_significant, effect_sizes, change_type
        )

        # Generate recommendation
        recommendation = self._generate_recommendation(
            metric_changes, is_significant, change_type, overall_impact
        )

        return ImpactResult(
            entity_type=entity_type,
            entity_id=entity_id,
            change_date=change_date,
            change_type=change_type,
            metric_changes=metric_changes,
            significance=significance,
            is_significant=is_significant,
            effect_sizes=effect_sizes,
            overall_impact=overall_impact,
            recommendation=recommendation,
            window_before=window_before,
            window_after=window_after,
            analyzed_at=datetime.utcnow(),
        )

    def _determine_overall_impact(
        self,
        metric_changes: dict,
        is_significant: dict,
        effect_sizes: dict,
        change_type: str,
    ) -> Literal['positive', 'negative', 'neutral', 'inconclusive']:
        """Determine overall impact based on metrics."""

        # Key metrics by change type
        primary_metrics = {
            'budget_change': ['cpl', 'leads'],
            'creative_change': ['ctr', 'cpl'],
            'audience_change': ['cpl', 'leads', 'ctr'],
            'pause': ['spend'],
            'reactivate': ['leads', 'cpl'],
        }

        key_metrics = primary_metrics.get(change_type, ['cpl', 'leads'])

        positive_signals = 0
        negative_signals = 0

        for metric in key_metrics:
            if metric not in metric_changes:
                continue

            change = metric_changes[metric]
            significant = is_significant.get(metric, False)
            effect = abs(effect_sizes.get(metric, 0))

            # For CPL: negative change is positive
            # For leads/CTR: positive change is positive
            is_positive_change = (
                (metric == 'cpl' and change < 0) or
                (metric != 'cpl' and change > 0)
            )

            if significant and effect >= self.min_effect_size:
                if is_positive_change:
                    positive_signals += 1
                else:
                    negative_signals += 1

        if positive_signals > negative_signals and positive_signals > 0:
            return 'positive'
        elif negative_signals > positive_signals and negative_signals > 0:
            return 'negative'
        elif positive_signals == 0 and negative_signals == 0:
            return 'inconclusive'
        else:
            return 'neutral'

    def _generate_recommendation(
        self,
        metric_changes: dict,
        is_significant: dict,
        change_type: str,
        overall_impact: str,
    ) -> str:
        """Generate actionable recommendation based on analysis."""

        if overall_impact == 'positive':
            if change_type == 'budget_change':
                return "Budget change was beneficial. Consider further scaling if metrics remain stable."
            elif change_type == 'creative_change':
                return "New creative is performing better. Continue with this direction."
            elif change_type == 'audience_change':
                return "Audience change improved performance. Consider similar audience expansions."
            else:
                return "Change had a positive impact. Continue monitoring."

        elif overall_impact == 'negative':
            if change_type == 'budget_change':
                return "Budget change hurt performance. Consider reverting or adjusting."
            elif change_type == 'creative_change':
                return "New creative underperforms previous. Consider reverting or A/B testing alternatives."
            elif change_type == 'audience_change':
                return "Audience change degraded performance. Consider reverting or narrowing targeting."
            else:
                return "Change had a negative impact. Consider reverting."

        elif overall_impact == 'inconclusive':
            return "Not enough data to determine impact. Continue monitoring for 3-7 more days."

        else:  # neutral
            return "Change had minimal impact. No action required."
```

### 7.3 Transfer Learning (Between Levels)

```python
# projects/ml/algorithms/models/transfer/level_transfer.py

class LevelTransferLearning:
    """
    Transfer learning between entity levels:
    - Train on campaigns -> apply to new adsets/ads
    - Learn global patterns -> adapt to specific entities
    """

    def __init__(self):
        self.global_classifier: Optional[CampaignClassifier] = None
        self.global_features_mean: Optional[dict] = None
        self.global_features_std: Optional[dict] = None

    async def train_global_model(
        self,
        config_id: int,
        data_service,
        ml_repo,
    ) -> dict:
        """
        Train a global model on all campaigns.
        This model learns general patterns that apply across all levels.
        """
        # Get features from all campaigns
        features_list = await data_service.get_all_entity_features(
            config_id=config_id,
            entity_type='campaign',
            days=90,
        )

        if len(features_list) < 10:
            raise ValueError("Need at least 10 campaigns for global model")

        # Calculate global statistics
        self._calculate_global_stats(features_list)

        # Get reference metrics
        avg_metrics = await data_service.get_aggregated_metrics(config_id)

        # Create labels
        labels = create_training_labels(features_list, avg_metrics.avg_cpl)

        # Train global classifier
        self.global_classifier = CampaignClassifier()
        metrics = self.global_classifier.train(
            features_list=features_list,
            labels=labels,
            avg_cpl=avg_metrics.avg_cpl,
            avg_ctr=avg_metrics.avg_ctr,
        )

        # Save as global model
        model_path = Path(settings.models_storage_path) / "transfer" / f"global_{config_id}.joblib"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        self.global_classifier.save(str(model_path))

        # Register in database
        await ml_repo.create_trained_model(
            name=f"global_classifier_{config_id}",
            model_type=ModelType.CLASSIFIER,
            config_id=config_id,
            model_path=str(model_path),
            training_metrics=metrics,
            parameters={'type': 'global_transfer'},
            samples_count=len(features_list),
        )

        return {
            "model_type": "global_transfer",
            "samples": len(features_list),
            "metrics": metrics,
        }

    def classify_new_entity(
        self,
        entity_features: EntityFeatures,
        avg_cpl: float,
        avg_ctr: float,
    ) -> ClassificationResult:
        """
        Classify a new entity (adset/ad) using the global model.
        Useful when entity doesn't have enough history for its own classification.
        """
        if self.global_classifier is None or not self.global_classifier.is_fitted:
            raise ValueError("Global model not trained")

        # Convert EntityFeatures to CampaignFeatures format
        campaign_features = self._convert_to_campaign_features(entity_features)

        # Classify using global model
        result = self.global_classifier.classify(
            campaign_features=campaign_features,
            avg_cpl=avg_cpl,
            avg_ctr=avg_ctr,
        )

        # Adjust confidence (transfer learning has more uncertainty)
        result.confidence_score *= 0.85  # 15% confidence reduction

        return result

    def _calculate_global_stats(self, features_list: list) -> None:
        """Calculate global feature statistics for normalization."""
        df = pd.DataFrame([vars(f) for f in features_list])
        self.global_features_mean = df.mean().to_dict()
        self.global_features_std = df.std().to_dict()

    def _convert_to_campaign_features(self, entity_features: EntityFeatures) -> CampaignFeatures:
        """Convert EntityFeatures to CampaignFeatures format."""
        return CampaignFeatures(
            campaign_id=entity_features.entity_id,
            config_id=entity_features.config_id,
            spend_7d=entity_features.spend_7d,
            impressions_7d=entity_features.impressions_7d,
            clicks_7d=entity_features.clicks_7d,
            leads_7d=entity_features.leads_7d,
            cpl_7d=entity_features.cpl_7d,
            ctr_7d=entity_features.ctr_7d,
            cpl_trend=entity_features.cpl_trend,
            leads_trend=entity_features.leads_trend,
            cpl_std_7d=entity_features.cpl_std_7d,
            conversion_rate_7d=entity_features.conversion_rate_7d,
            days_with_leads_7d=entity_features.days_with_leads_7d,
            frequency_7d=entity_features.frequency_7d,
        )
```

---

## API Endpoints

### New Endpoints for Phase 7

```python
# projects/ml/api/impact.py

@router.post("/impact/analyze")
async def analyze_impact(
    request: ImpactAnalysisRequest,
    session: AsyncSession = Depends(get_async_session),
) -> ImpactAnalysisResponse:
    """
    Analyze the impact of a change on entity performance.

    Request body:
    {
        "config_id": 1,
        "entity_type": "campaign",
        "entity_id": "123456",
        "change_date": "2026-02-01",
        "change_type": "budget_change",
        "window_before": 7,
        "window_after": 7
    }
    """
    pass

@router.get("/forecasts/ensemble/{entity_id}")
async def get_ensemble_forecast(
    entity_id: str,
    config_id: int,
    entity_type: str = "campaign",
    metric: str = "cpl",
    horizon_days: int = 7,
    session: AsyncSession = Depends(get_async_session),
) -> EnsembleForecastResponse:
    """Get ensemble forecast combining multiple methods."""
    pass
```

---

## Celery Beat Schedule (Complete)

```python
# app/celery.py - beat_schedule additions

'train-classifiers-daily': {
    'task': 'projects.ml.jobs.training_tasks.train_classifiers_all',
    'schedule': crontab(hour=3, minute=0),
},
'tune-prophet-weekly': {
    'task': 'projects.ml.jobs.training_tasks.tune_prophet_all',
    'schedule': crontab(day_of_week=0, hour=2, minute=0),  # Sunday 2 AM
},
'train-if-daily': {
    'task': 'projects.ml.jobs.training_tasks.train_anomaly_detectors_all',
    'schedule': crontab(hour=4, minute=0),
},
'calibrate-ensemble-daily': {
    'task': 'projects.ml.jobs.training_tasks.calibrate_ensemble_all',
    'schedule': crontab(hour=5, minute=0),
},
```

---

## File Structure

```
projects/ml/
├── algorithms/models/
│   ├── timeseries/
│   │   ├── forecaster.py          # Existing
│   │   ├── prophet_tuner.py       # NEW (Phase 5)
│   │   └── ensemble_forecaster.py # NEW (Phase 7)
│   ├── causal/
│   │   └── impact_analyzer.py     # NEW (Phase 7)
│   └── transfer/
│       └── level_transfer.py      # NEW (Phase 7)
├── api/
│   └── impact.py                  # NEW (Phase 7)
├── services/
│   └── classification_service.py  # UPDATE (Phase 4)
└── jobs/
    └── training_tasks.py          # UPDATE (Phases 4-7)
```

---

## Success Metrics

| Phase | Metric | Target |
|-------|--------|--------|
| 4 | XGBoost F1-score | > 0.75 |
| 5 | Prophet MAPE improvement | > 10% reduction |
| 6 | IF models persisted | 100% in DB |
| 7 | Ensemble MAPE vs single | > 5% improvement |
| 7 | Impact analysis coverage | All change types |

---

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Insufficient data for tuning | Fallback to default params |
| Cross-validation too slow | Limit grid search combinations |
| Impact analysis false positives | Require min effect size + significance |
| Transfer learning poor generalization | Reduce confidence on transferred predictions |
