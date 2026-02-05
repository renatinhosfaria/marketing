"""Unit of Work interface for transaction management."""
from abc import ABC, abstractmethod
from typing import TypeVar

T = TypeVar('T')


class UnitOfWork(ABC):
    """Interface para Unit of Work pattern.

    Gerencia transações de banco de dados, garantindo
    que múltiplas operações sejam atômicas.
    """

    @abstractmethod
    async def __aenter__(self) -> 'UnitOfWork':
        """Inicia o contexto de transação."""
        pass

    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Finaliza o contexto, fazendo commit ou rollback."""
        pass

    @abstractmethod
    async def commit(self) -> None:
        """Confirma todas as alterações."""
        pass

    @abstractmethod
    async def rollback(self) -> None:
        """Desfaz todas as alterações."""
        pass
