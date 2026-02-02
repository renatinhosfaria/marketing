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
    statements = build_hard_delete_statements(123, ["thread-1", "thread-2"])

    assert _table_names(statements) == [
        "agent_checkpoints",
        "agent_writes",
        "agent_conversations",
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


def test_build_hard_delete_statements_skips_thread_dependent_tables_when_empty():
    statements = build_hard_delete_statements(123, [])
    table_names = _table_names(statements)

    assert "agent_checkpoints" not in table_names
    assert "agent_writes" not in table_names
    assert "agent_conversations" in table_names


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


class _FakeSession:
    def __init__(self, thread_ids):
        self.thread_ids = thread_ids
        self.executed = []

    async def execute(self, stmt):
        self.executed.append(stmt)
        if stmt.__class__.__name__ == "Select":
            return _FakeResult(self.thread_ids)
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
            return _FakeResult(["thread-1"])
        return _FakeResult()

    async def commit(self):
        self.committed = True


@pytest.mark.asyncio
async def test_hard_delete_config_executes_expected_deletes():
    session = _FakeSession(["thread-1"])

    await hard_delete_config(session, 99)

    delete_tables = [
        stmt.table.name for stmt in session.executed if isinstance(stmt, Delete)
    ]

    assert delete_tables == [
        "agent_checkpoints",
        "agent_writes",
        "agent_conversations",
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
        current_user={"id": 1},
    )

    delete_tables = [
        stmt.table.name for stmt in session.executed if isinstance(stmt, Delete)
    ]

    assert "sistema_facebook_ads_config" in delete_tables
    assert response["message"] == "Configuração excluída permanentemente"
