"""
Factory do PostgresStore com embeddings pgvector.

create_store_cm(): retorna async context manager do Store.
O lifespan do FastAPI gerencia o ciclo de vida (open/close).

Usa AsyncConnectionPool (psycopg_pool) para resiliencia contra
conexoes mortas — pool reconecta automaticamente.
"""

from contextlib import asynccontextmanager
from typing import AsyncIterator

from psycopg_pool import AsyncConnectionPool
from psycopg.rows import dict_row
from langgraph.store.postgres import AsyncPostgresStore

from shared.config import settings
from projects.agent.config import agent_settings

import structlog

logger = structlog.get_logger()


def _get_index_config() -> dict | None:
    """Retorna config de embeddings pgvector se API key estiver configurada."""
    if not agent_settings.openai_api_key:
        logger.warning(
            "store.embeddings_disabled",
            reason="AGENT_OPENAI_API_KEY nao configurada. "
            "Busca semantica desabilitada.",
        )
        return None

    import os
    os.environ.setdefault("OPENAI_API_KEY", agent_settings.openai_api_key)

    from langchain.embeddings import init_embeddings

    logger.info("store.embeddings_enabled")
    return {
        "dims": agent_settings.store_embedding_dims,
        "embed": init_embeddings(agent_settings.store_embedding_model),
        "fields": ["insight_text", "context"],
    }


@asynccontextmanager
async def create_store_cm() -> AsyncIterator[AsyncPostgresStore]:
    """Context manager que cria e gerencia o PostgresStore.

    Usa pool de conexoes (min=1, max=3) ao inves de conexao unica.
    Isso evita o erro 'the connection is closed' quando a conexao
    fica idle por muito tempo e o PostgreSQL a fecha.
    """
    logger.info(
        "store.init_start",
        embedding_model=agent_settings.store_embedding_model,
        dims=agent_settings.store_embedding_dims,
    )

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
        index_config = _get_index_config()
        store_kwargs = {"conn": pool}
        if index_config:
            store_kwargs["index"] = index_config

        store = AsyncPostgresStore(**store_kwargs)
        await store.setup()
        logger.info("store.init_complete", pool_min=1, pool_max=3)
        yield store
    finally:
        await pool.close()
