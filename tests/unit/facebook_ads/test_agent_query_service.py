import pytest
from unittest.mock import AsyncMock

from projects.facebook_ads.services.agent_query_service import AgentQueryService
from projects.facebook_ads.services.agent_sql_guard import SQLGuardError


class _FakeRow:
    def __init__(self, mapping):
        self._mapping = mapping


class _FakeResult:
    def __init__(self, rows=None, rowcount=0):
        self._rows = rows or []
        self.rowcount = rowcount

    def fetchall(self):
        return self._rows


class _FakeSession:
    def __init__(self):
        self.execute = AsyncMock()


@pytest.mark.asyncio
async def test_service_executes_select_and_audits() -> None:
    db = _FakeSession()
    db.execute.side_effect = [
        _FakeResult(rows=[_FakeRow({"campaign_id": "123", "spend": 100.0})], rowcount=1),
        _FakeResult(rowcount=1),
    ]

    service = AgentQueryService(db)
    result = await service.execute_sql(prompt="listar campanhas", sql="SELECT 1")

    assert result["operationType"] == "SELECT"
    assert result["rowsAffected"] == 1
    assert result["rows"][0]["campaign_id"] == "123"
    assert db.execute.await_count == 2


@pytest.mark.asyncio
async def test_service_blocks_unsafe_sql() -> None:
    db = _FakeSession()
    service = AgentQueryService(db)

    with pytest.raises(SQLGuardError):
        await service.execute_sql(prompt="apagar tudo", sql="DELETE FROM sistema_facebook_ads_ads")
