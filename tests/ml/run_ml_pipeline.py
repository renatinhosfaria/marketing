#!/usr/bin/env python3
"""
ML Pipeline Runner - Executes all ML services to populate data with entity names.

This script runs the ML pipeline services to create new records with entity names
(campaign_name, adset_name, ad_name) populated from the Facebook Ads tables.

Run with: python tests/ml/run_ml_pipeline.py
"""

import asyncio
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from shared.infrastructure.persistence.database import (
    async_session_maker,
    check_database_connection,
)
from shared.db.models.famachat_readonly import SistemaFacebookAdsConfig
from shared.core.logging import get_logger

logger = get_logger(__name__)


async def run_pipeline_for_config(session: AsyncSession, config_id: int, config_name: str):
    """Run the full ML pipeline for a single config."""
    print(f"\n{'='*60}")
    print(f"Processing config: {config_id} - {config_name}")
    print('='*60)

    # 1. Run Classification
    print("\n[1/4] Running Classification Service...")
    try:
        from projects.ml.services.classification_service import ClassificationService
        classification_service = ClassificationService(session)

        classifications = await classification_service.classify_entities(
            config_id=config_id,
            entity_type="campaign",
            force_reclassify=True  # Force to get new records with names
        )
        await session.commit()

        print(f"  Classifications created: {len(classifications)}")

        # Show sample with name
        if classifications:
            sample = classifications[0]
            print(f"  Sample: entity_id={sample.get('entity_id')}, tier={sample.get('tier')}")
    except Exception as e:
        print(f"  Error: {e}")
        await session.rollback()

    # 2. Run Recommendations
    print("\n[2/4] Running Recommendation Service...")
    try:
        from projects.ml.services.recommendation_service import RecommendationService
        recommendation_service = RecommendationService(session)

        recommendations = await recommendation_service.generate_entity_recommendations(
            config_id=config_id,
            entity_type="campaign",
            force_refresh=True
        )
        await session.commit()

        print(f"  Recommendations created: {len(recommendations)}")
    except Exception as e:
        print(f"  Error: {e}")
        await session.rollback()

    # 3. Run Anomaly Detection
    print("\n[3/4] Running Anomaly Detection...")
    try:
        from projects.ml.services.anomaly_service import AnomalyService
        anomaly_service = AnomalyService(session)

        result = await anomaly_service.detect_anomalies(
            config_id=config_id,
            entity_type="campaign",
            days_to_analyze=7,
            history_days=30,
        )
        await session.commit()

        print(f"  Entities analyzed: {result.entities_analyzed}")
        print(f"  Anomalies detected: {result.anomalies_detected}")
    except Exception as e:
        print(f"  Error: {e}")
        await session.rollback()

    # 4. Run Features/Forecasts via scheduled task helpers
    print("\n[4/4] Running Features & Forecasts...")
    try:
        from projects.ml.jobs.scheduled_tasks import (
            _run_compute_features_for_config,
            _run_forecasts_for_config,
        )
        from shared.db.session import create_isolated_async_session_maker

        # Create isolated session for the async helpers
        isolated_engine, isolated_session_maker = create_isolated_async_session_maker()

        try:
            features_result = await _run_compute_features_for_config(
                config_id, window_days=30, session_maker=isolated_session_maker
            )
            print(f"  Features: processed={features_result['processed']}, inserted={features_result['inserted']}")

            forecasts_result = await _run_forecasts_for_config(
                config_id, window_days=30, session_maker=isolated_session_maker
            )
            print(f"  Forecasts: generated={forecasts_result['generated']}, skipped={forecasts_result['skipped']}")
        finally:
            await isolated_engine.dispose()
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()


async def verify_names_populated(session: AsyncSession, config_id: int):
    """Verify that names have been populated in the ML tables."""
    print(f"\n{'='*60}")
    print("Verifying entity names in ML tables...")
    print('='*60)

    from sqlalchemy import func
    from projects.ml.db.models import (
        MLClassification,
        MLRecommendation,
        MLAnomaly,
        MLFeature,
        MLForecast,
    )

    tables = [
        ("ml_classifications", MLClassification),
        ("ml_recommendations", MLRecommendation),
        ("ml_anomalies", MLAnomaly),
        ("ml_features", MLFeature),
        ("ml_forecasts", MLForecast),
    ]

    for table_name, model in tables:
        # Count total records
        total_result = await session.execute(
            select(func.count(model.id)).where(model.config_id == config_id)
        )
        total = total_result.scalar() or 0

        # Count records with campaign_name populated
        with_name_result = await session.execute(
            select(func.count(model.id)).where(
                model.config_id == config_id,
                model.campaign_name.isnot(None)
            )
        )
        with_name = with_name_result.scalar() or 0

        pct = (with_name / total * 100) if total > 0 else 0
        print(f"  {table_name}: {with_name}/{total} records with campaign_name ({pct:.1f}%)")


async def main():
    """Main entry point."""
    print("\n" + "=" * 60)
    print("ML PIPELINE RUNNER - Populating Entity Names")
    print("=" * 60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    print("\nChecking database connection...")
    if not await check_database_connection():
        print("ERROR: Cannot connect to database!")
        sys.exit(1)
    print("Database connection OK")

    async with async_session_maker() as session:
        # Get active configs
        result = await session.execute(
            select(SistemaFacebookAdsConfig)
            .where(SistemaFacebookAdsConfig.is_active.is_(True))
        )
        configs = result.scalars().all()

        if not configs:
            print("\nNo active Facebook Ads configs found!")
            sys.exit(1)

        print(f"\nFound {len(configs)} active config(s)")

        # Run pipeline for each config
        for config in configs:
            await run_pipeline_for_config(session, config.id, config.account_name or config.name)

        # Verify names were populated
        for config in configs:
            await verify_names_populated(session, config.id)

    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Pipeline complete!")


if __name__ == "__main__":
    asyncio.run(main())
