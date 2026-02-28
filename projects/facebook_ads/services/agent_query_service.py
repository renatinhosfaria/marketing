"""Servico de execucao SQL do agente FB Ads com auditoria."""

from datetime import datetime
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from projects.facebook_ads.services.agent_sql_guard import SQLGuard


class AgentQueryService:
    """Executa SQL validado e registra auditoria."""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def execute_sql(self, prompt: str, sql: str, requested_by: str = "fbads") -> dict[str, Any]:
        validation = SQLGuard.validate(sql)

        started_at = datetime.utcnow()
        result = await self.db.execute(text(sql))

        rows: list[dict[str, Any]] = []
        if validation.operation_type == "SELECT":
            rows = [dict(row._mapping) for row in result.fetchall()]
            rows_affected = len(rows)
        else:
            rows_affected = int(result.rowcount or 0)

        duration_ms = int((datetime.utcnow() - started_at).total_seconds() * 1000)

        await self.db.execute(
            text(
                """
                INSERT INTO fbads_agent_query_audit
                (
                    requested_at,
                    requested_by,
                    prompt,
                    generated_sql,
                    operation_type,
                    execution_status,
                    rows_affected,
                    duration_ms
                )
                VALUES
                (
                    now(),
                    :requested_by,
                    :prompt,
                    :generated_sql,
                    :operation_type,
                    'success',
                    :rows_affected,
                    :duration_ms
                )
                """
            ),
            {
                "requested_by": requested_by,
                "prompt": prompt,
                "generated_sql": sql,
                "operation_type": validation.operation_type,
                "rows_affected": rows_affected,
                "duration_ms": duration_ms,
            },
        )

        return {
            "operationType": validation.operation_type,
            "sqlExecuted": sql,
            "rowsAffected": rows_affected,
            "rows": rows,
            "durationMs": duration_ms,
        }
