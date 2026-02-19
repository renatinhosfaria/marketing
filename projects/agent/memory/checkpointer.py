"""
Factory do AsyncPostgresSaver (checkpointer).

create_checkpointer_cm(): retorna async context manager do checkpointer.
O lifespan do FastAPI gerencia o ciclo de vida (open/close).

Usa AsyncConnectionPool (psycopg_pool) para resiliencia contra
conexoes mortas — pool reconecta automaticamente.
"""

from contextlib import asynccontextmanager
from typing import AsyncIterator

from psycopg_pool import AsyncConnectionPool
from psycopg.rows import dict_row
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from shared.config import settings

import structlog

logger = structlog.get_logger()


@asynccontextmanager
async def create_checkpointer_cm() -> AsyncIterator[AsyncPostgresSaver]:
    """Context manager que cria e gerencia o AsyncPostgresSaver.

    Usa pool de conexoes (min=1, max=3) ao inves de conexao unica.
    Isso evita o erro 'the connection is closed' quando a conexao
    fica idle por muito tempo e o PostgreSQL a fecha.
    """
    logger.info("checkpointer.init_start")

    pool = AsyncConnectionPool(
        conninfo=settings.database_url,
        min_size=1,
        max_size=3,
        check=AsyncConnectionPool.check_connection,
        max_idle=60,
        kwargs={
            "autocommit": True,
            "prepare_threshold": 0,
            "row_factory": dict_row,
            "keepalives": 1,
            "keepalives_idle": 30,
            "keepalives_interval": 10,
            "keepalives_count": 5,
        },
    )
    await pool.open()

    try:
        checkpointer = AsyncPostgresSaver(conn=pool)
        await checkpointer.setup()

        # Adicionar coluna created_at que o LangGraph nao cria por padrao.
        # Necessaria para listar conversas ordenadas por data.
        async with pool.connection() as conn:
            await conn.execute(
                "ALTER TABLE checkpoints "
                "ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ NOT NULL DEFAULT now()"
            )

        logger.info("checkpointer.init_complete", pool_min=1, pool_max=3)
        yield checkpointer
    finally:
        await pool.close()
