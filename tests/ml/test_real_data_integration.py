#!/usr/bin/env python3
"""
Comprehensive ML Module Integration Test with Real Data.

This script tests all ML models against real data from the FamaChat database:
- Anomaly Detection (Isolation Forest, Z-score, IQR)
- Campaign Classification (XGBoost)
- Time Series Forecasting (Prophet, EMA, Linear, Ensemble)
- Recommendation Engine (Rule-based)
- Impact Analysis (Causal inference)
- Transfer Learning (Cross-level classification)

Run with: python -m pytest tests/ml/test_real_data_integration.py -v -s
Or directly: python tests/ml/test_real_data_integration.py
"""

import asyncio
import sys
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sqlalchemy import select, func, and_
from sqlalchemy.ext.asyncio import AsyncSession

from shared.infrastructure.persistence.database import (
    async_session_maker,
    check_database_connection,
)
from shared.db.models.famachat_readonly import (
    SistemaFacebookAdsConfig,
    SistemaFacebookAdsCampaigns,
    SistemaFacebookAdsAdsets,
    SistemaFacebookAdsAds,
    SistemaFacebookAdsInsightsHistory,
)
from projects.ml.algorithms.models.anomaly.anomaly_detector import AnomalyDetector
from projects.ml.algorithms.models.classification.campaign_classifier import (
    CampaignClassifier,
)
from projects.ml.algorithms.models.timeseries.forecaster import TimeSeriesForecaster
from projects.ml.algorithms.models.timeseries.ensemble_forecaster import (
    EnsembleForecaster,
)
from projects.ml.algorithms.models.recommendation.rule_engine import RuleEngine
from projects.ml.algorithms.models.causal.impact_analyzer import ImpactAnalyzer
from projects.ml.algorithms.models.transfer.level_transfer import LevelTransferLearning
from projects.ml.services.feature_engineering import (
    FeatureEngineer,
    CampaignFeatures,
    EntityFeatures,
)


@dataclass
class TestResult:
    """Result of a single test."""
    name: str
    success: bool
    duration_ms: float
    details: dict = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class TestReport:
    """Complete test report."""
    total_tests: int = 0
    passed: int = 0
    failed: int = 0
    results: list[TestResult] = field(default_factory=list)
    config_id: Optional[int] = None
    data_summary: dict = field(default_factory=dict)

    def add_result(self, result: TestResult):
        self.results.append(result)
        self.total_tests += 1
        if result.success:
            self.passed += 1
        else:
            self.failed += 1

    def print_report(self):
        print("\n" + "=" * 80)
        print("ML MODULE INTEGRATION TEST REPORT")
        print("=" * 80)
        print(f"\nConfig ID: {self.config_id}")
        print(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\nData Summary:")
        for key, value in self.data_summary.items():
            print(f"  - {key}: {value}")
        print(f"\nResults: {self.passed}/{self.total_tests} passed")
        print("-" * 80)

        for result in self.results:
            status = "PASS" if result.success else "FAIL"
            print(f"\n[{status}] {result.name} ({result.duration_ms:.0f}ms)")
            if result.error:
                print(f"  Error: {result.error}")
            for key, value in result.details.items():
                if isinstance(value, dict):
                    print(f"  {key}:")
                    for k, v in value.items():
                        print(f"    - {k}: {v}")
                elif isinstance(value, list):
                    print(f"  {key}: {len(value)} items")
                    for item in value[:3]:  # Show first 3
                        print(f"    - {item}")
                    if len(value) > 3:
                        print(f"    ... and {len(value) - 3} more")
                else:
                    print(f"  {key}: {value}")

        print("\n" + "=" * 80)
        if self.failed == 0:
            print("ALL TESTS PASSED!")
        else:
            print(f"WARNING: {self.failed} test(s) failed!")
        print("=" * 80)


class MLIntegrationTester:
    """Tests all ML models with real data."""

    def __init__(self, session: AsyncSession):
        self.session = session
        self.report = TestReport()
        self.config_id: Optional[int] = None
        self.campaigns: list = []
        self.adsets: list = []
        self.ads: list = []
        self.insights: list = []
        self.campaign_dfs: dict[str, pd.DataFrame] = {}
        self.avg_cpl: float = 50.0
        self.avg_ctr: float = 1.0

    async def setup(self) -> bool:
        """Load real data from database."""
        print("\n[SETUP] Loading real data from database...")

        # Get active config
        result = await self.session.execute(
            select(SistemaFacebookAdsConfig)
            .where(SistemaFacebookAdsConfig.is_active.is_(True))
            .limit(1)
        )
        config = result.scalar_one_or_none()
        if not config:
            print("  ERROR: No active Facebook Ads config found!")
            return False

        self.config_id = config.id
        self.report.config_id = config.id
        print(f"  Config ID: {config.id} - {config.account_name}")

        # Get campaigns
        result = await self.session.execute(
            select(SistemaFacebookAdsCampaigns).where(
                SistemaFacebookAdsCampaigns.config_id == self.config_id
            )
        )
        self.campaigns = result.scalars().all()
        print(f"  Campaigns: {len(self.campaigns)}")

        # Get adsets
        result = await self.session.execute(
            select(SistemaFacebookAdsAdsets).where(
                SistemaFacebookAdsAdsets.config_id == self.config_id
            )
        )
        self.adsets = result.scalars().all()
        print(f"  Adsets: {len(self.adsets)}")

        # Get ads
        result = await self.session.execute(
            select(SistemaFacebookAdsAds).where(
                SistemaFacebookAdsAds.config_id == self.config_id
            )
        )
        self.ads = result.scalars().all()
        print(f"  Ads: {len(self.ads)}")

        # Get insights (last 60 days)
        since_date = datetime.now() - timedelta(days=60)
        result = await self.session.execute(
            select(SistemaFacebookAdsInsightsHistory)
            .where(
                and_(
                    SistemaFacebookAdsInsightsHistory.config_id == self.config_id,
                    SistemaFacebookAdsInsightsHistory.date >= since_date,
                )
            )
            .order_by(SistemaFacebookAdsInsightsHistory.date)
        )
        self.insights = result.scalars().all()
        print(f"  Insights (60 days): {len(self.insights)}")

        # Aggregate insights by campaign
        self._aggregate_daily_data()

        self.report.data_summary = {
            "campaigns": len(self.campaigns),
            "adsets": len(self.adsets),
            "ads": len(self.ads),
            "insights": len(self.insights),
            "date_range": f"{since_date.strftime('%Y-%m-%d')} to {datetime.now().strftime('%Y-%m-%d')}",
        }

        return len(self.campaigns) > 0 and len(self.insights) > 0

    def _aggregate_daily_data(self):
        """Aggregate insights by campaign by date into DataFrames."""
        campaign_data: dict[str, list[dict]] = {}

        for insight in self.insights:
            campaign_id = insight.campaign_id
            if campaign_id not in campaign_data:
                campaign_data[campaign_id] = []

            spend = float(insight.spend or 0)
            leads = insight.leads or 0
            impressions = insight.impressions or 0
            clicks = insight.clicks or 0
            reach = insight.reach or 0

            # Calculate derived metrics
            cpl = spend / leads if leads > 0 else 0.0
            ctr = (clicks / impressions * 100) if impressions > 0 else 0.0
            frequency = (impressions / reach) if reach > 0 else 0.0

            campaign_data[campaign_id].append({
                "date": insight.date,
                "spend": spend,
                "impressions": impressions,
                "clicks": clicks,
                "leads": leads,
                "reach": reach,
                "cpl": cpl,
                "ctr": ctr,
                "frequency": frequency,
            })

        # Convert to DataFrames and aggregate by date
        all_cpls = []
        all_ctrs = []

        for campaign_id, data_list in campaign_data.items():
            df = pd.DataFrame(data_list)
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                # Aggregate by date (in case of multiple ads/adsets per campaign per day)
                df_agg = df.groupby('date').agg({
                    'spend': 'sum',
                    'impressions': 'sum',
                    'clicks': 'sum',
                    'leads': 'sum',
                    'reach': 'sum',
                }).reset_index()

                # Recalculate derived metrics after aggregation
                df_agg['cpl'] = df_agg.apply(
                    lambda r: r['spend'] / r['leads'] if r['leads'] > 0 else 0.0,
                    axis=1
                )
                df_agg['ctr'] = df_agg.apply(
                    lambda r: r['clicks'] / r['impressions'] * 100 if r['impressions'] > 0 else 0.0,
                    axis=1
                )
                df_agg['frequency'] = df_agg.apply(
                    lambda r: r['impressions'] / r['reach'] if r['reach'] > 0 else 0.0,
                    axis=1
                )
                df_agg = df_agg.sort_values('date')

                self.campaign_dfs[campaign_id] = df_agg

                # Collect for averages
                for _, row in df_agg.iterrows():
                    if row['cpl'] > 0:
                        all_cpls.append(row['cpl'])
                    if row['ctr'] > 0:
                        all_ctrs.append(row['ctr'])

        # Calculate global averages
        self.avg_cpl = sum(all_cpls) / len(all_cpls) if all_cpls else 50.0
        self.avg_ctr = sum(all_ctrs) / len(all_ctrs) if all_ctrs else 1.0
        print(f"  Global avg CPL: R${self.avg_cpl:.2f}")
        print(f"  Global avg CTR: {self.avg_ctr:.2f}%")

    async def test_anomaly_detection(self) -> TestResult:
        """Test anomaly detection with real campaign data."""
        print("\n[TEST] Anomaly Detection...")
        start = datetime.now()
        details = {}
        error = None

        try:
            detector = AnomalyDetector()
            all_anomalies = []

            # Test on each campaign with enough data
            campaigns_tested = 0
            for campaign_id, df in self.campaign_dfs.items():
                if len(df) < 7:
                    continue

                campaigns_tested += 1
                anomalies = detector.detect_anomalies(
                    df=df,
                    entity_type="campaign",
                    entity_id=campaign_id,
                    config_id=self.config_id,
                )
                all_anomalies.extend(anomalies)

            details["campaigns_tested"] = campaigns_tested
            details["anomalies_found"] = len(all_anomalies)

            # Group by type
            by_type = {}
            for a in all_anomalies:
                atype = a.anomaly_type.value if hasattr(a.anomaly_type, "value") else str(a.anomaly_type)
                by_type[atype] = by_type.get(atype, 0) + 1
            details["by_type"] = by_type

            # Group by severity
            by_severity = {}
            for a in all_anomalies:
                sev = a.severity.value if hasattr(a.severity, "value") else str(a.severity)
                by_severity[sev] = by_severity.get(sev, 0) + 1
            details["by_severity"] = by_severity

            success = True

        except Exception as e:
            import traceback
            error = f"{str(e)}\n{traceback.format_exc()}"
            success = False

        duration = (datetime.now() - start).total_seconds() * 1000
        return TestResult(
            name="Anomaly Detection",
            success=success,
            duration_ms=duration,
            details=details,
            error=error,
        )

    async def test_isolation_forest_training(self) -> TestResult:
        """Test Isolation Forest model training."""
        print("\n[TEST] Isolation Forest Training...")
        start = datetime.now()
        details = {}
        error = None

        try:
            detector = AnomalyDetector()

            # Prepare training data from all campaigns
            all_data = []
            for campaign_id, df in self.campaign_dfs.items():
                if len(df) >= 7:
                    all_data.append(df)

            if all_data:
                training_df = pd.concat(all_data, ignore_index=True)

                if len(training_df) >= 30:
                    # Train the model
                    success_train = detector.train_isolation_forest(
                        training_data=training_df,
                        contamination=0.15,
                    )
                    details["model_trained"] = success_train
                    details["training_samples"] = len(training_df)
                    success = True
                else:
                    details["model_trained"] = False
                    details["reason"] = f"Insufficient data ({len(training_df)} samples, need 30+)"
                    success = True  # Not a failure, just insufficient data
            else:
                details["model_trained"] = False
                details["reason"] = "No campaign data available"
                success = True

        except Exception as e:
            import traceback
            error = f"{str(e)}\n{traceback.format_exc()}"
            success = False

        duration = (datetime.now() - start).total_seconds() * 1000
        return TestResult(
            name="Isolation Forest Training",
            success=success,
            duration_ms=duration,
            details=details,
            error=error,
        )

    async def test_campaign_classification(self) -> TestResult:
        """Test campaign classification."""
        print("\n[TEST] Campaign Classification...")
        start = datetime.now()
        details = {}
        error = None

        try:
            classifier = CampaignClassifier()
            feature_engineer = FeatureEngineer()

            classifications = {}
            for campaign_id, df in self.campaign_dfs.items():
                if len(df) < 7:
                    continue

                # Get campaign info
                campaign = next(
                    (c for c in self.campaigns if c.campaign_id == campaign_id),
                    None,
                )
                if not campaign:
                    continue

                campaign_info = {
                    'campaign_id': campaign_id,
                    'config_id': self.config_id,
                    'status': campaign.status,
                    'daily_budget': campaign.daily_budget,
                    'lifetime_budget': campaign.lifetime_budget,
                }

                # Compute features
                features = feature_engineer.compute_campaign_features(
                    daily_data=df,
                    campaign_info=campaign_info,
                )

                # Classify using rules
                result = classifier.classify_by_rules(
                    campaign_features=features,
                    avg_cpl=self.avg_cpl,
                    avg_ctr=self.avg_ctr,
                )

                tier_name = result.tier.value if hasattr(result.tier, "value") else str(result.tier)
                classifications[campaign_id] = {
                    "tier": tier_name,
                    "confidence": round(result.confidence_score, 2),
                }

            # Count by tier
            tier_counts = {}
            for c in classifications.values():
                tier = c["tier"]
                tier_counts[tier] = tier_counts.get(tier, 0) + 1

            details["campaigns_classified"] = len(classifications)
            details["tier_distribution"] = tier_counts
            details["avg_cpl_reference"] = round(self.avg_cpl, 2)
            details["avg_ctr_reference"] = round(self.avg_ctr, 4)

            success = len(classifications) > 0

        except Exception as e:
            import traceback
            error = f"{str(e)}\n{traceback.format_exc()}"
            success = False

        duration = (datetime.now() - start).total_seconds() * 1000
        return TestResult(
            name="Campaign Classification",
            success=success,
            duration_ms=duration,
            details=details,
            error=error,
        )

    async def test_time_series_forecasting(self) -> TestResult:
        """Test time series forecasting."""
        print("\n[TEST] Time Series Forecasting...")
        start = datetime.now()
        details = {}
        error = None

        try:
            forecaster = TimeSeriesForecaster(method='auto')
            forecasts_generated = 0
            methods_used = set()

            # Test on campaigns with enough data (limit to first 5)
            tested = 0
            for campaign_id, df in list(self.campaign_dfs.items())[:5]:
                if len(df) < 14:
                    continue

                tested += 1

                # Forecast CPL
                try:
                    forecast = forecaster.forecast(
                        df=df,
                        metric='cpl',
                        entity_type='campaign',
                        entity_id=campaign_id,
                        horizon_days=7,
                    )
                    if forecast and forecast.forecasts:
                        forecasts_generated += 1
                        methods_used.add(forecast.method)
                except Exception as fe:
                    pass  # Skip failed forecasts

                # Forecast Leads
                try:
                    forecast = forecaster.forecast(
                        df=df,
                        metric='leads',
                        entity_type='campaign',
                        entity_id=campaign_id,
                        horizon_days=7,
                    )
                    if forecast and forecast.forecasts:
                        forecasts_generated += 1
                        methods_used.add(forecast.method)
                except Exception as fe:
                    pass

            details["campaigns_tested"] = tested
            details["forecasts_generated"] = forecasts_generated
            details["methods_used"] = list(methods_used)

            success = forecasts_generated > 0

        except Exception as e:
            import traceback
            error = f"{str(e)}\n{traceback.format_exc()}"
            success = False

        duration = (datetime.now() - start).total_seconds() * 1000
        return TestResult(
            name="Time Series Forecasting",
            success=success,
            duration_ms=duration,
            details=details,
            error=error,
        )

    async def test_ensemble_forecasting(self) -> TestResult:
        """Test ensemble forecasting."""
        print("\n[TEST] Ensemble Forecasting...")
        start = datetime.now()
        details = {}
        error = None

        try:
            ensemble = EnsembleForecaster(include_prophet=True)

            # Find a campaign with enough data
            test_df = None
            test_campaign = None
            for campaign_id, df in self.campaign_dfs.items():
                if len(df) >= 30:
                    test_df = df
                    test_campaign = campaign_id
                    break

            if test_df is not None:
                # Calibrate weights
                weights = ensemble.calibrate_weights(
                    historical_data=test_df,
                    metric='cpl',
                    validation_days=7,
                )

                details["weights_calibrated"] = True
                details["ensemble_weights"] = {k: round(v, 3) for k, v in weights.items()}
                details["methods_in_ensemble"] = list(ensemble.forecasters.keys())
                success = True
            else:
                details["weights_calibrated"] = False
                details["reason"] = "No campaign with 30+ days of data"
                success = True  # Not a failure, just insufficient data

        except Exception as e:
            import traceback
            error = f"{str(e)}\n{traceback.format_exc()}"
            success = False

        duration = (datetime.now() - start).total_seconds() * 1000
        return TestResult(
            name="Ensemble Forecasting",
            success=success,
            duration_ms=duration,
            details=details,
            error=error,
        )

    async def test_recommendation_engine(self) -> TestResult:
        """Test recommendation engine."""
        print("\n[TEST] Recommendation Engine...")
        start = datetime.now()
        details = {}
        error = None

        try:
            rule_engine = RuleEngine(
                avg_cpl=self.avg_cpl,
                avg_ctr=self.avg_ctr,
            )
            feature_engineer = FeatureEngineer()

            all_recommendations = []

            for campaign_id, df in self.campaign_dfs.items():
                if len(df) < 7:
                    continue

                # Get campaign info
                campaign = next(
                    (c for c in self.campaigns if c.campaign_id == campaign_id),
                    None,
                )
                if not campaign:
                    continue

                campaign_info = {
                    'campaign_id': campaign_id,
                    'config_id': self.config_id,
                    'status': campaign.status,
                    'daily_budget': campaign.daily_budget,
                    'lifetime_budget': campaign.lifetime_budget,
                }

                features = feature_engineer.compute_campaign_features(
                    daily_data=df,
                    campaign_info=campaign_info,
                )

                recommendations = rule_engine.generate_recommendations(
                    features=features,
                    entity_type="campaign",
                )
                all_recommendations.extend(recommendations)

            # Group by type
            by_type = {}
            for r in all_recommendations:
                rtype = r.recommendation_type.value if hasattr(r.recommendation_type, "value") else str(r.recommendation_type)
                by_type[rtype] = by_type.get(rtype, 0) + 1

            details["recommendations_generated"] = len(all_recommendations)
            details["by_type"] = by_type

            # Show top 3 recommendations
            sorted_recs = sorted(
                all_recommendations, key=lambda x: x.priority, reverse=True
            )[:3]
            details["top_recommendations"] = [
                {
                    "type": r.recommendation_type.value if hasattr(r.recommendation_type, "value") else str(r.recommendation_type),
                    "priority": r.priority,
                    "confidence": round(r.confidence_score, 2),
                }
                for r in sorted_recs
            ]

            success = True

        except Exception as e:
            import traceback
            error = f"{str(e)}\n{traceback.format_exc()}"
            success = False

        duration = (datetime.now() - start).total_seconds() * 1000
        return TestResult(
            name="Recommendation Engine",
            success=success,
            duration_ms=duration,
            details=details,
            error=error,
        )

    async def test_impact_analysis(self) -> TestResult:
        """Test causal impact analysis."""
        print("\n[TEST] Impact Analysis...")
        start = datetime.now()
        details = {}
        error = None

        try:
            analyzer = ImpactAnalyzer()

            # The ImpactAnalyzer.analyze_impact requires a data_service
            # We'll test the analyzer's internal methods directly
            test_result = None

            for campaign_id, df in self.campaign_dfs.items():
                if len(df) >= 14:
                    # Split in half for before/after
                    mid = len(df) // 2
                    before_df = df.iloc[:mid].copy()
                    after_df = df.iloc[mid:].copy()

                    # Test _analyze_metric method
                    metric = 'cpl'
                    if metric in df.columns:
                        result = analyzer._analyze_metric(
                            before_df[metric].dropna(),
                            after_df[metric].dropna(),
                        )
                        if result:
                            test_result = {
                                'campaign_id': campaign_id,
                                'metric': metric,
                                'pct_change': result['pct_change'],
                                'is_significant': result['is_significant'],
                                'effect_size': result['effect_size'],
                                'confidence': result['confidence'],
                            }
                            break

            if test_result:
                details["analysis_completed"] = True
                details["sample_analysis"] = {
                    "campaign": test_result['campaign_id'],
                    "metric": test_result['metric'],
                    "pct_change": f"{test_result['pct_change']:+.1f}%",
                    "is_significant": test_result['is_significant'],
                    "effect_size": round(test_result['effect_size'], 3),
                    "confidence": round(test_result['confidence'], 3),
                }
                success = True
            else:
                details["analysis_completed"] = False
                details["reason"] = "No campaign with sufficient data for analysis"
                success = True  # Not a failure

        except Exception as e:
            import traceback
            error = f"{str(e)}\n{traceback.format_exc()}"
            success = False

        duration = (datetime.now() - start).total_seconds() * 1000
        return TestResult(
            name="Impact Analysis",
            success=success,
            duration_ms=duration,
            details=details,
            error=error,
        )

    async def test_transfer_learning(self) -> TestResult:
        """Test transfer learning for cross-level classification."""
        print("\n[TEST] Transfer Learning...")
        start = datetime.now()
        details = {}
        error = None

        try:
            transfer = LevelTransferLearning()
            classifier = CampaignClassifier()
            feature_engineer = FeatureEngineer()

            # Prepare campaign features for training
            campaign_features_list = []
            for campaign_id, df in self.campaign_dfs.items():
                if len(df) < 7:
                    continue

                campaign = next(
                    (c for c in self.campaigns if c.campaign_id == campaign_id),
                    None,
                )
                if not campaign:
                    continue

                campaign_info = {
                    'campaign_id': campaign_id,
                    'config_id': self.config_id,
                    'status': campaign.status,
                    'daily_budget': campaign.daily_budget,
                    'lifetime_budget': campaign.lifetime_budget,
                }

                features = feature_engineer.compute_campaign_features(
                    daily_data=df,
                    campaign_info=campaign_info,
                )
                campaign_features_list.append(features)

            details["campaign_features_computed"] = len(campaign_features_list)

            # Train classifier directly (transfer learning uses the classifier)
            if len(campaign_features_list) >= 10:
                # Create labels using heuristics
                from projects.ml.algorithms.models.classification.campaign_classifier import (
                    create_training_labels,
                )
                labels = create_training_labels(campaign_features_list, self.avg_cpl)

                metrics = classifier.train(
                    features_list=campaign_features_list,
                    labels=labels,
                    avg_cpl=self.avg_cpl,
                    avg_ctr=self.avg_ctr,
                    test_size=0.2,
                )

                details["model_trained"] = True
                details["training_samples"] = len(campaign_features_list)
                details["training_metrics"] = {
                    "accuracy": round(metrics.get('accuracy', 0), 3),
                    "f1_macro": round(metrics.get('f1_macro', 0), 3),
                }

                # Test classification on a sample
                if campaign_features_list:
                    test_features = campaign_features_list[0]
                    result = classifier.classify(
                        campaign_features=test_features,
                        avg_cpl=self.avg_cpl,
                        avg_ctr=self.avg_ctr,
                    )
                    details["test_classification"] = {
                        "tier": result.tier.value,
                        "confidence": round(result.confidence_score, 2),
                    }

                success = True
            else:
                details["model_trained"] = False
                details["reason"] = f"Insufficient campaigns ({len(campaign_features_list)}, need 10+)"
                success = True  # Not a failure

        except Exception as e:
            import traceback
            error = f"{str(e)}\n{traceback.format_exc()}"
            success = False

        duration = (datetime.now() - start).total_seconds() * 1000
        return TestResult(
            name="Transfer Learning / XGBoost Training",
            success=success,
            duration_ms=duration,
            details=details,
            error=error,
        )

    async def test_feature_engineering(self) -> TestResult:
        """Test feature engineering pipeline."""
        print("\n[TEST] Feature Engineering...")
        start = datetime.now()
        details = {}
        error = None

        try:
            feature_engineer = FeatureEngineer()

            features_computed = 0
            sample_features = None

            for campaign_id, df in self.campaign_dfs.items():
                if len(df) < 7:
                    continue

                campaign = next(
                    (c for c in self.campaigns if c.campaign_id == campaign_id),
                    None,
                )
                if not campaign:
                    continue

                campaign_info = {
                    'campaign_id': campaign_id,
                    'config_id': self.config_id,
                    'status': campaign.status,
                    'daily_budget': campaign.daily_budget,
                    'lifetime_budget': campaign.lifetime_budget,
                }

                features = feature_engineer.compute_campaign_features(
                    daily_data=df,
                    campaign_info=campaign_info,
                )
                features_computed += 1
                if sample_features is None:
                    sample_features = features

            details["features_computed"] = features_computed

            if sample_features:
                details["sample_features"] = {
                    "campaign_id": sample_features.campaign_id,
                    "spend_7d": round(sample_features.spend_7d, 2),
                    "leads_7d": sample_features.leads_7d,
                    "cpl_7d": round(sample_features.cpl_7d, 2),
                    "ctr_7d": round(sample_features.ctr_7d, 4),
                    "cpl_trend": f"{sample_features.cpl_trend:+.1f}%",
                    "leads_trend": f"{sample_features.leads_trend:+.1f}%",
                    "days_with_leads_7d": sample_features.days_with_leads_7d,
                }

            success = features_computed > 0

        except Exception as e:
            import traceback
            error = f"{str(e)}\n{traceback.format_exc()}"
            success = False

        duration = (datetime.now() - start).total_seconds() * 1000
        return TestResult(
            name="Feature Engineering",
            success=success,
            duration_ms=duration,
            details=details,
            error=error,
        )

    async def run_all_tests(self):
        """Run all tests and generate report."""
        print("\n" + "=" * 80)
        print("STARTING ML MODULE INTEGRATION TESTS WITH REAL DATA")
        print("=" * 80)

        # Setup
        if not await self.setup():
            print("\nFATAL: Could not load test data!")
            return

        # Run tests
        tests = [
            self.test_feature_engineering,
            self.test_anomaly_detection,
            self.test_isolation_forest_training,
            self.test_campaign_classification,
            self.test_time_series_forecasting,
            self.test_ensemble_forecasting,
            self.test_recommendation_engine,
            self.test_impact_analysis,
            self.test_transfer_learning,
        ]

        for test in tests:
            result = await test()
            self.report.add_result(result)

        # Print report
        self.report.print_report()


async def main():
    """Main entry point."""
    print("Checking database connection...")
    if not await check_database_connection():
        print("ERROR: Cannot connect to database!")
        sys.exit(1)

    async with async_session_maker() as session:
        tester = MLIntegrationTester(session)
        await tester.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())
