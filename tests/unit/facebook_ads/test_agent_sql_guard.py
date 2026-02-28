import pytest

from projects.facebook_ads.services.agent_sql_guard import SQLGuard, SQLGuardError


def test_blocks_drop() -> None:
    with pytest.raises(SQLGuardError):
        SQLGuard.validate("DROP TABLE sistema_facebook_ads_ads")


def test_blocks_truncate() -> None:
    with pytest.raises(SQLGuardError):
        SQLGuard.validate("TRUNCATE TABLE sistema_facebook_ads_ads")


def test_blocks_delete_without_where() -> None:
    with pytest.raises(SQLGuardError):
        SQLGuard.validate("DELETE FROM sistema_facebook_ads_ads")


def test_blocks_update_without_where() -> None:
    with pytest.raises(SQLGuardError):
        SQLGuard.validate("UPDATE sistema_facebook_ads_ads SET status = PAUSED")


def test_allows_safe_update_with_where() -> None:
    result = SQLGuard.validate("UPDATE sistema_facebook_ads_ads SET status=PAUSED WHERE id=1")
    assert result.operation_type == "UPDATE"
