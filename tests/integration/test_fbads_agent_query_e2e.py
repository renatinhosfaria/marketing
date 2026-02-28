import os
from uuid import uuid4

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from projects.facebook_ads.services.agent_query_service import AgentQueryService
from projects.facebook_ads.services.agent_sql_guard import SQLGuardError


def _database_url() -> str:
    url = os.getenv("DATABASE_URL")
    if not url:
        pytest.skip("DATABASE_URL not set for e2e checks.")
    return url


def _async_database_url(url: str) -> str:
    # Remove sslmode para compatibilidade com asyncpg
    if "?" in url:
        base, params = url.split("?", 1)
        filtered = [p for p in params.split("&") if not p.startswith("sslmode=")]
        url = f"{base}?{'&'.join(filtered)}" if filtered else base

    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+asyncpg://", 1)
    if url.startswith("postgres://"):
        return url.replace("postgres://", "postgresql+asyncpg://", 1)
    return url


@pytest.mark.asyncio
async def test_prompt_nl_to_sql_executes_and_writes_audit() -> None:
    requested_by = f"fbads-e2e-{uuid4().hex[:8]}"
    prompt = "listar top 5 campanhas por spend da config 1"

    engine = create_async_engine(_async_database_url(_database_url()))
    session_maker = async_sessionmaker(engine, expire_on_commit=False)

    try:
        async with session_maker() as session:
            service = AgentQueryService(session)
            result = await service.execute_sql(prompt=prompt, sql=None, requested_by=requested_by)
            await session.commit()

            assert result["operationType"] == "SELECT"
            assert "sistema_facebook_ads_insights_history" in result["sqlExecuted"]

            audit_result = await session.execute(
                text(
                    """
                    SELECT execution_status, operation_type, generated_sql
                    FROM fbads_agent_query_audit
                    WHERE requested_by = :requested_by
                    ORDER BY id DESC
                    LIMIT 1
                    """
                ),
                {"requested_by": requested_by},
            )
            audit_row = audit_result.mappings().first()

            assert audit_row is not None
            assert audit_row["execution_status"] == "success"
            assert audit_row["operation_type"] == "SELECT"
            assert "sistema_facebook_ads_insights_history" in audit_row["generated_sql"]
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_destructive_sql_without_where_is_blocked() -> None:
    engine = create_async_engine(_async_database_url(_database_url()))
    session_maker = async_sessionmaker(engine, expire_on_commit=False)

    try:
        async with session_maker() as session:
            service = AgentQueryService(session)
            with pytest.raises(SQLGuardError, match="DELETE sem WHERE bloqueado"):
                await service.execute_sql(
                    prompt="apagar tudo",
                    sql="DELETE FROM sistema_facebook_ads_ads",
                    requested_by="fbads-e2e-block",
                )
    finally:
        await engine.dispose()
