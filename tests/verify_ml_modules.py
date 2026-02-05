
import sys
import os
import asyncio
from typing import Optional, List
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Add the project root to the python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# FIX: Unset extra env vars that might cause Pydantic validation errors if config doesn't allow extras
os.environ.pop("flower_user", None)
# FIX: Unset extra env vars that might cause Pydantic validation errors if config doesn't allow extras
os.environ.pop("flower_user", None)
os.environ.pop("flower_password", None)

from projects.ml.algorithms.models.recommendation.rule_engine import RuleEngine, Recommendation, create_rule_engine
from projects.ml.services.feature_engineering import CampaignFeatures
from projects.ml.db.models import RecommendationType, CampaignTier, ModelType
from projects.ml.algorithms.models.classification.campaign_classifier import CampaignClassifier, classifier
from projects.ml.algorithms.models.timeseries.forecaster import TimeSeriesForecaster, forecaster

# --- Utility: Mock Data Generators ---

def create_mock_campaign_features(
    cpl_ratio: float, 
    leads: int, 
    spend: float, 
    cpl_trend: float,
    frequency: float = 1.0, 
    ctr_trend: float = 0.0,
    leads_trend: float = 0.0
) -> CampaignFeatures:
    """Helper to create CampaignFeatures with specific characteristics."""
    avg_cpl = 50.0  # Reference
    cpl_7d = avg_cpl * cpl_ratio
    
    return CampaignFeatures(
        campaign_id="test_camp_1",
        config_id=1,
        # Basic metrics
        spend_7d=spend,
        leads_7d=leads,
        impressions_7d=int(spend * 20), # dummy
        clicks_7d=int(spend * 0.2), # dummy
        
        # Calculated
        cpl_7d=cpl_7d,
        ctr_7d=1.5,
        cpc_7d=spend/max(1, int(spend*0.2)),
        conversion_rate_7d=5.0,
        
        # Trends
        cpl_trend=cpl_trend,
        leads_trend=leads_trend,
        spend_trend=0.0,
        ctr_trend=ctr_trend,
        
        # 14d/30d context (simplified)
        cpl_14d=cpl_7d,
        leads_14d=leads * 2,
        cpl_30d=cpl_7d,
        leads_30d=leads * 4,
        avg_daily_spend_30d=spend/7,
        
        # Volatility
        cpl_std_7d=cpl_7d * 0.1,
        leads_std_7d=1.0,
        
        # Misc
        best_day_of_week=0,
        worst_day_of_week=1,
        frequency_7d=frequency,
        reach_7d=1000,
        days_with_leads_7d=5 if leads > 0 else 0,
        days_active=30,
        is_active=True,
        has_budget=True,
        computed_at=datetime.utcnow()
    )

def create_mock_timeseries_data(n_days=30, trend=0.5):
    """Create synthetic daily data for forecasting."""
    dates = [datetime.now() - timedelta(days=x) for x in range(n_days)]
    dates.reverse()
    
    data = []
    for i, date in enumerate(dates):
        # Linear trend + noise
        value = 10 + (i * trend) + (np.random.normal(0, 2))
        data.append({
            'date': date,
            'leads': max(0, int(value)),
            'spend': max(0, value * 50),
            'impressions': max(0, value * 5000)
        })
    return pd.DataFrame(data)

# --- Verification Tests ---

def test_rule_engine():
    print("\n--- Testing Rule Engine ---")
    
    avg_metrics = {'avg_cpl': 50.0, 'avg_ctr': 1.0, 'avg_conversion_rate': 5.0}
    engine = create_rule_engine(avg_metrics)
    
    # Case 1: Scale Up (High Performance)
    # CPL 0.4x avg (20.0), 10 leads, stable trend
    feat_scale = create_mock_campaign_features(cpl_ratio=0.4, leads=10, spend=200, cpl_trend=-5.0)
    recs_scale = engine.generate_recommendations(feat_scale)
    
    found_scale = any(r.recommendation_type == RecommendationType.SCALE_UP for r in recs_scale)
    print(f"CASE 1 (Scale Up): {'✅ PASS' if found_scale else '❌ FAIL'}")
    if found_scale:
        print(f"  - Generated: {recs_scale[0].title}")

    # Case 2: Pause Campaign (Underperformer)
    # CPL 3.0x avg (150.0), 1 lead, trend rising
    feat_pause = create_mock_campaign_features(cpl_ratio=3.0, leads=1, spend=150, cpl_trend=25.0)
    # Trick: set days active > 7
    feat_pause.days_active = 10 
    
    recs_pause = engine.generate_recommendations(feat_pause)
    
    # Note: Logic for PAUSE might require strict conditions (e.g. no leads or tier).
    # Let's check logic: very_high_cpl = cpl_ratio >= 2.5 and days_active >= 7
    found_pause = any(r.recommendation_type == RecommendationType.PAUSE_CAMPAIGN for r in recs_pause)
    print(f"CASE 2 (Pause): {'✅ PASS' if found_pause else '❌ FAIL'}")
    if found_pause:
        print(f"  - Generated: {recs_pause[0].title}")

def test_campaign_classifier():
    print("\n--- Testing Campaign Classifier ---")
    
    # 1. Test Rule-Based Classification (fallback)
    features = create_mock_campaign_features(cpl_ratio=0.5, leads=5, spend=100, cpl_trend=0)
    # Trick: days_with_leads_7d >= 4 for high performer
    features.days_with_leads_7d = 5
    
    result = classifier.classify_by_rules(features, avg_cpl=50.0, avg_ctr=1.0)
    
    print(f"Rule Classification Result: {result.tier.value}")
    is_correct = result.tier == CampaignTier.HIGH_PERFORMER
    print(f"Rule-Based Test: {'✅ PASS' if is_correct else '❌ FAIL'}")
    
    # 2. Test Training Flow (Smoke Test)
    print("Training Smoke Test...")
    train_features = []
    train_labels = []
    
    # Generate synthetic dataset
    for _ in range(25):
        # High Performer
        f = create_mock_campaign_features(cpl_ratio=0.5, leads=5, spend=100, cpl_trend=0)
        f.days_with_leads_7d = 5
        train_features.append(f)
        train_labels.append(CampaignTier.HIGH_PERFORMER)
        
        # Moderate
        f = create_mock_campaign_features(cpl_ratio=0.9, leads=3, spend=100, cpl_trend=5)
        f.days_with_leads_7d = 3
        train_features.append(f)
        train_labels.append(CampaignTier.MODERATE)
        
        # Low
        f = create_mock_campaign_features(cpl_ratio=1.4, leads=1, spend=100, cpl_trend=10)
        f.days_with_leads_7d = 1
        train_features.append(f)
        train_labels.append(CampaignTier.LOW)
        
        # Underperformer
        f = create_mock_campaign_features(cpl_ratio=2.0, leads=0, spend=100, cpl_trend=20)
        f.days_with_leads_7d = 0
        train_features.append(f)
        train_labels.append(CampaignTier.UNDERPERFORMER)

    try:
        metrics = classifier.train(
            train_features, 
            train_labels, 
            avg_cpl=50.0, 
            avg_ctr=1.0,
            test_size=0.2
        )
        print(f"Training Metrics: Accuracy={metrics['accuracy']:.2f}")
        print(f"Training Test: {'✅ PASS' if metrics['accuracy'] > 0 else '❌ FAIL'}")
        
        # 3. Test Prediction with Trained Model
        pred_result = classifier.classify(features, avg_cpl=50.0, avg_ctr=1.0)
        print(f"Model Classification Result: {pred_result.tier.value} (Confidence: {pred_result.confidence_score:.2f})")
        print(f"Prediction Test: {'✅ PASS' if pred_result.confidence_score > 0 else '❌ FAIL'}")

    except Exception as e:
        print(f"Training Failed: {str(e)}")
        import traceback
        traceback.print_exc()

def test_forecaster():
    print("\n--- Testing Time Series Forecaster ---")
    
    df = create_mock_timeseries_data(n_days=60, trend=0.5) # 60 days of data
    
    # Test CPL Forecast
    try:
        # Force EMA to avoid Prophet dependency issues in test environment if missing, 
        # though code handles it. Let's use 'auto'.
        series = forecaster.forecast_leads(
            df, 
            entity_type="campaign", 
            entity_id="test_1", 
            horizon_days=7
        )
        
        print(f"Method Used: {series.method}")
        print(f"Forecasts Generated: {len(series.forecasts)}")
        if len(series.forecasts) == 7:
            first_val = series.forecasts[0].predicted_value
            last_val = series.forecasts[-1].predicted_value
            print(f"Trend: {first_val:.2f} -> {last_val:.2f}")
            print(f"Forecasting Test: {'✅ PASS'}")
        else:
            print(f"Forecasting Test: ❌ FAIL (Expected 7 days, got {len(series.forecasts)})")
            
    except Exception as e:
        print(f"Forecasting Failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("=== STARTING ML MODULE VERIFICATION ===")
    test_rule_engine()
    test_campaign_classifier()
    test_forecaster()
    print("\n=== VERIFICATION COMPLETE ===")
