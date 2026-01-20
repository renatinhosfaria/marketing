"""
Configuração do SQLAlchemy e gerenciamento de sessões.
Suporta operações assíncronas com asyncpg.
"""

from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import text

from app.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def get_async_database_url() -> str:
    """
    Converte DATABASE_URL para formato async (asyncpg).
    postgresql:// -> postgresql+asyncpg://
    Remove parametros incompativeis com asyncpg (sslmode).
    """
    url = settings.database_url

    # Remover sslmode que nao e suportado pelo asyncpg
    if "?" in url:
        base, params = url.split("?", 1)
        param_list = params.split("&")
        # Filtrar parametros incompativeis
        filtered_params = [p for p in param_list if not p.startswith("sslmode=")]
        if filtered_params:
            url = f"{base}?{'&'.join(filtered_params)}"
        else:
            url = base

    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+asyncpg://", 1)
    elif url.startswith("postgres://"):
        return url.replace("postgres://", "postgresql+asyncpg://", 1)
    return url


# Engine assíncrona
engine = create_async_engine(
    get_async_database_url(),
    pool_size=settings.database_pool_size,
    max_overflow=settings.database_max_overflow,
    pool_recycle=settings.database_pool_recycle,
    pool_timeout=settings.database_pool_timeout,
    pool_pre_ping=True,
    echo=settings.debug,
)

# Session factory assíncrona
async_session_maker = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


class Base(DeclarativeBase):
    """Base class para todos os modelos SQLAlchemy."""
    pass


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency injection para obter sessão do banco de dados.
    Uso com FastAPI Depends().

    Yields:
        AsyncSession: Sessão do banco de dados
    """
    async with async_session_maker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def check_database_connection() -> bool:
    """
    Verifica se a conexão com o banco de dados está funcionando.

    Returns:
        bool: True se conectado, False caso contrário
    """
    try:
        async with async_session_maker() as session:
            await session.execute(text("SELECT 1"))
            return True
    except Exception as e:
        logger.error("Erro ao conectar ao banco de dados", error=str(e))
        return False


# Engine síncrona para scripts e migrações
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from contextlib import contextmanager

sync_engine = create_engine(
    settings.database_url,
    pool_size=settings.database_pool_size,
    max_overflow=settings.database_max_overflow,
    pool_pre_ping=True,
)

# Session factory síncrona
sync_session_maker = sessionmaker(
    bind=sync_engine,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


@contextmanager
def get_db_session():
    """
    Context manager síncrono para obter sessão do banco de dados.
    Uso com 'with' statement para tools síncronas.

    Yields:
        Session: Sessão síncrona do banco de dados
    """
    session = sync_session_maker()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def create_isolated_async_session_maker():
    """
    Cria uma engine e session_maker isolados para uso em Celery tasks.

    O problema: Celery usa fork() para criar workers. O asyncpg mantém
    referências ao event loop onde as conexões foram criadas. Quando
    asyncio.run() cria um novo loop, o pool antigo não funciona.

    Solução: Criar uma nova engine/session para cada execução de task,
    garantindo que o pool seja criado no mesmo loop onde será usado.

    Returns:
        tuple: (engine, async_session_maker) - Devem ser fechados após uso
    """
    isolated_engine = create_async_engine(
        get_async_database_url(),
        pool_size=5,  # Pool menor para tasks isoladas
        max_overflow=2,
        pool_pre_ping=True,
        echo=False,
    )

    isolated_session_maker = async_sessionmaker(
        isolated_engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autocommit=False,
        autoflush=False,
    )

    return isolated_engine, isolated_session_maker


async def dispose_isolated_engine(engine):
    """
    Fecha a engine isolada e libera conexões do pool.
    Deve ser chamado após terminar o uso em uma task Celery.
    """
    await engine.dispose()


from contextlib import asynccontextmanager

@asynccontextmanager
async def isolated_async_session():
    """
    Context manager para criar e gerenciar uma session isolada.
    Cria engine nova, executa operações, e fecha tudo ao final.

    Uso:
        async with isolated_async_session() as session:
            # usar session aqui
            await session.execute(...)

    Yields:
        AsyncSession: Sessão isolada do banco de dados
    """
    engine, session_maker = create_isolated_async_session_maker()
    try:
        async with session_maker() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
    finally:
        await engine.dispose()
