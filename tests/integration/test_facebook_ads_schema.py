import os

import pytest
from sqlalchemy import create_engine, text


REQUIRED_TABLES = [
    "sistema_facebook_ads_config",
    "sistema_facebook_ads_campaigns",
    "sistema_facebook_ads_adsets",
    "sistema_facebook_ads_ads",
    "sistema_facebook_ads_insights_history",
    "sistema_facebook_ads_insights_today",
    "sistema_facebook_ads_sync_history",
    "ml_facebook_ads_management_log",
    "ml_facebook_ads_rate_limit_log",
]


def _database_url() -> str:
    url = os.getenv("DATABASE_URL")
    if not url:
        pytest.skip("DATABASE_URL not set for schema checks.")
    return url


def test_facebook_ads_tables_exist() -> None:
    url = _database_url()
    engine = create_engine(url)
    missing = []

    try:
        with engine.connect() as conn:
            for table in REQUIRED_TABLES:
                result = conn.execute(
                    text("select to_regclass(:table_name)"),
                    {"table_name": f"public.{table}"},
                )
                if result.scalar() is None:
                    missing.append(table)
    finally:
        engine.dispose()

    assert not missing, f"Missing tables: {', '.join(missing)}"
