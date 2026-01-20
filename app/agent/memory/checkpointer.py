"""
Configuração do checkpointer PostgreSQL para persistência de estado do agente.
"""

from typing import Optional
from contextlib import asynccontextmanager

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool

from app.config import settings
from app.core.logging import get_logger


logger = get_logger(__name__)


class AgentCheckpointer:
    """
    Gerenciador do checkpointer PostgreSQL para o agente LangGraph.

    Usa connection pool para eficiência em produção.
    """

    _instance: Optional["AgentCheckpointer"] = None
    _pool: Optional[AsyncConnectionPool] = None
    _checkpointer: Optional[AsyncPostgresSaver] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    async def initialize(cls) -> "AgentCheckpointer":
        """
        Inicializa o checkpointer com connection pool.

        Returns:
            Instância do AgentCheckpointer
        """
        instance = cls()

        if instance._pool is None:
            # Converter URL para formato psycopg3
            database_url = settings.database_url
            if database_url.startswith("postgresql+asyncpg://"):
                database_url = database_url.replace("postgresql+asyncpg://", "postgresql://")

            # Criar pool de conexões
            instance._pool = AsyncConnectionPool(
                conninfo=database_url,
                min_size=2,
                max_size=10,
                open=False
            )
            await instance._pool.open()

            logger.info("Pool de conexões do checkpointer inicializado")

        if instance._checkpointer is None:
            instance._checkpointer = AsyncPostgresSaver(instance._pool)
            await instance._checkpointer.setup()

            logger.info("Checkpointer PostgreSQL configurado")

        return instance

    @classmethod
    async def get_checkpointer(cls) -> AsyncPostgresSaver:
        """
        Obtém o checkpointer inicializado.

        Returns:
            AsyncPostgresSaver pronto para uso
        """
        instance = await cls.initialize()

        if instance._checkpointer is None:
            raise RuntimeError("Checkpointer não inicializado")

        return instance._checkpointer

    @classmethod
    async def close(cls):
        """
        Fecha o pool de conexões.
        """
        instance = cls()

        if instance._pool is not None:
            await instance._pool.close()
            instance._pool = None
            instance._checkpointer = None

            logger.info("Pool de conexões do checkpointer fechado")


@asynccontextmanager
async def get_agent_checkpointer():
    """
    Context manager para obter o checkpointer.

    Uso:
        async with get_agent_checkpointer() as checkpointer:
            agent = graph.compile(checkpointer=checkpointer)
    """
    checkpointer = await AgentCheckpointer.get_checkpointer()
    try:
        yield checkpointer
    finally:
        # O pool é gerenciado globalmente, não fechamos aqui
        pass


async def setup_checkpointer_tables():
    """
    Configura as tabelas do checkpointer no banco.

    Deve ser chamado durante a inicialização da aplicação.
    """
    try:
        checkpointer = await AgentCheckpointer.get_checkpointer()
        await checkpointer.setup()
        logger.info("Tabelas do checkpointer configuradas")
    except Exception as e:
        logger.error(f"Erro ao configurar tabelas do checkpointer: {e}")
        raise


async def cleanup_old_checkpoints(days: int = 30):
    """
    Remove checkpoints antigos para liberar espaço.

    Args:
        days: Número de dias para manter
    """
    try:
        checkpointer = await AgentCheckpointer.get_checkpointer()

        # Query para remover checkpoints antigos
        async with checkpointer._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    DELETE FROM agent_checkpoints
                    WHERE thread_ts < NOW() - INTERVAL '%s days'
                    """,
                    (days,)
                )

                deleted = cur.rowcount
                await conn.commit()

                logger.info(f"Removidos {deleted} checkpoints com mais de {days} dias")

    except Exception as e:
        logger.error(f"Erro ao limpar checkpoints antigos: {e}")
