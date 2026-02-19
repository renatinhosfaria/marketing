from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from projects.facebook_ads.services.sync_service import SyncService


class _FakeResult:
    def __init__(self, value):
        self._value = value

    def scalar_one_or_none(self):
        return self._value


class _FakeSession:
    def __init__(self, sync_history):
        self.sync_history = sync_history
        self.executed = []
        self.flush_count = 0

    async def execute(self, stmt):
        self.executed.append(stmt)
        if stmt.__class__.__name__ == "Select":
            return _FakeResult(self.sync_history)
        return _FakeResult(None)

    async def flush(self):
        self.flush_count += 1


def _make_sync_history(sync_type: str = "today_only"):
    return SimpleNamespace(
        id=1,
        config_id=14,
        status="pending",
        sync_type=sync_type,
        entities_synced=0,
        campaigns_synced=0,
        adsets_synced=0,
        ads_synced=0,
        insights_synced=0,
        date_range_start=None,
        date_range_end=None,
        error_message=None,
        error_details=None,
        started_at=None,
        completed_at=None,
        duration_ms=None,
    )


@pytest.mark.asyncio
async def test_execute_sync_today_only_reconsolidates_recent_days():
    sync_history = _make_sync_history("today_only")
    config = SimpleNamespace(id=14)
    session = _FakeSession(sync_history)
    service = SyncService(session)

    service.get_config = AsyncMock(return_value=config)
    service._insights_service.sync_today = AsyncMock(
        return_value={"synced": 5, "inserted": 5, "updated": 0, "errors": 0}
    )
    service._insights_service.sync_recent_days = AsyncMock(
        return_value={"synced": 8, "inserted": 2, "updated": 6, "errors": 0}
    )

    result = await service.execute_sync(sync_id=1, sync_type="today_only")

    service._insights_service.sync_today.assert_awaited_once_with(config)
    service._insights_service.sync_recent_days.assert_awaited_once_with(
        config, days_back=3
    )
    assert result["insights"]["synced"] == 13
    assert result["insights"]["updated"] == 6
    assert sync_history.insights_synced == 13


@pytest.mark.asyncio
async def test_execute_sync_incremental_reconsolidates_recent_days():
    sync_history = _make_sync_history("incremental")
    config = SimpleNamespace(id=14)
    session = _FakeSession(sync_history)
    service = SyncService(session)

    service.get_config = AsyncMock(return_value=config)
    service._campaigns_service.sync = AsyncMock(
        return_value={"synced": 2, "created": 0, "updated": 2, "errors": 0}
    )
    service._adsets_ads_service.sync_adsets = AsyncMock(
        return_value={"synced": 3, "created": 0, "updated": 3, "errors": 0}
    )
    service._adsets_ads_service.sync_ads = AsyncMock(
        return_value={"synced": 4, "created": 0, "updated": 4, "errors": 0}
    )
    service._insights_service.sync_today = AsyncMock(
        return_value={"synced": 5, "inserted": 5, "updated": 0, "errors": 0}
    )
    service._insights_service.sync_recent_days = AsyncMock(
        return_value={"synced": 8, "inserted": 2, "updated": 6, "errors": 0}
    )

    result = await service.execute_sync(sync_id=1, sync_type="incremental")

    service._insights_service.sync_today.assert_awaited_once_with(config)
    service._insights_service.sync_recent_days.assert_awaited_once_with(
        config, days_back=3
    )
    assert result["insights"]["synced"] == 13
    assert sync_history.insights_synced == 13
    assert sync_history.entities_synced == 22
