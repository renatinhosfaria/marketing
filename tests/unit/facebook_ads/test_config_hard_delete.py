import pytest
from sqlalchemy.sql.dml import Delete

from projects.facebook_ads.api.config_endpoints import delete_config
from projects.facebook_ads.services.config_deletion import (
    build_hard_delete_statements,
    hard_delete_config,
)


def _table_names(statements):
    return [stmt.table.name for stmt in statements]


def test_build_hard_delete_statements_includes_tables_in_order():
    statements = build_hard_delete_statements(123)

    assert _table_names(statements) == [
        "ml_predictions",
        "ml_campaign_classifications",
        "ml_recommendations",
        "ml_anomalies",
        "ml_features",
        "ml_forecasts",
        "ml_training_jobs",
        "ml_trained_models",
        "ml_facebook_ads_management_log",
        "ml_facebook_ads_rate_limit_log",
        "sistema_facebook_ads_sync_history",
        "sistema_facebook_ads_insights_today",
        "sistema_facebook_ads_insights_history",
        "sistema_facebook_ads_ads",
        "sistema_facebook_ads_adsets",
        "sistema_facebook_ads_campaigns",
        "sistema_facebook_ads_config",
    ]


def test_build_hard_delete_statements_includes_config_table():
    statements = build_hard_delete_statements(123)
    assert "sistema_facebook_ads_config" in _table_names(statements)


class _FakeResult:
    def __init__(self, values=None):
        self._values = values or []

    def scalars(self):
        return self

    def all(self):
        return self._values

    def scalar_one_or_none(self):
        if not self._values:
            return None
        return self._values[0]

    def scalar(self):
        if not self._values:
            return None
        return self._values[0]


class _FakeSession:
    def __init__(self):
        self.executed = []

    async def execute(self, stmt):
        self.executed.append(stmt)
        if stmt.__class__.__name__ == "TextClause":
            return _FakeResult(["table_exists"])
        return _FakeResult()


class _EndpointSession:
    def __init__(self, config):
        self.config = config
        self.executed = []
        self.committed = False
        self._select_calls = 0

    async def execute(self, stmt):
        self.executed.append(stmt)
        if stmt.__class__.__name__ == "Select":
            self._select_calls += 1
            if self._select_calls == 1:
                return _FakeResult([self.config])
            return _FakeResult()
        if stmt.__class__.__name__ == "TextClause":
            return _FakeResult(["table_exists"])
        return _FakeResult()

    async def commit(self):
        self.committed = True


@pytest.mark.asyncio
async def test_hard_delete_config_executes_expected_deletes():
    session = _FakeSession()

    await hard_delete_config(session, 99)

    delete_tables = [
        stmt.table.name for stmt in session.executed if isinstance(stmt, Delete)
    ]

    assert delete_tables == [
        "ml_predictions",
        "ml_campaign_classifications",
        "ml_recommendations",
        "ml_anomalies",
        "ml_features",
        "ml_forecasts",
        "ml_training_jobs",
        "ml_trained_models",
        "ml_facebook_ads_management_log",
        "ml_facebook_ads_rate_limit_log",
        "sistema_facebook_ads_sync_history",
        "sistema_facebook_ads_insights_today",
        "sistema_facebook_ads_insights_history",
        "sistema_facebook_ads_ads",
        "sistema_facebook_ads_adsets",
        "sistema_facebook_ads_campaigns",
        "sistema_facebook_ads_config",
    ]


@pytest.mark.asyncio
async def test_delete_config_hard_delete_executes_deletes():
    session = _EndpointSession(object())

    response = await delete_config(
        99,
        True,
        session,
    )

    delete_tables = [
        stmt.table.name for stmt in session.executed if isinstance(stmt, Delete)
    ]

    assert "sistema_facebook_ads_config" in delete_tables
    assert response["message"] == "Configuração excluída permanentemente"
