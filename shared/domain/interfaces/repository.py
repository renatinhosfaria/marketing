"""Base repository interface for Clean Architecture."""
from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Optional, Sequence

T = TypeVar('T')
ID = TypeVar('ID')


class Repository(ABC, Generic[T, ID]):
    """Interface base para todos os repositories.

    Define o contrato que todas as implementações de repository
    devem seguir, garantindo consistência entre os projetos.
    """

    @abstractmethod
    async def get_by_id(self, id: ID) -> Optional[T]:
        """Busca uma entidade pelo ID."""
        pass

    @abstractmethod
    async def get_all(self, limit: int = 50, offset: int = 0) -> Sequence[T]:
        """Lista entidades com paginação."""
        pass

    @abstractmethod
    async def add(self, entity: T) -> T:
        """Adiciona uma nova entidade."""
        pass

    @abstractmethod
    async def update(self, entity: T) -> T:
        """Atualiza uma entidade existente."""
        pass

    @abstractmethod
    async def delete(self, id: ID) -> bool:
        """Remove uma entidade pelo ID."""
        pass


class ReadOnlyRepository(ABC, Generic[T, ID]):
    """Interface para repositories somente leitura."""

    @abstractmethod
    async def get_by_id(self, id: ID) -> Optional[T]:
        """Busca uma entidade pelo ID."""
        pass

    @abstractmethod
    async def get_all(self, limit: int = 50, offset: int = 0) -> Sequence[T]:
        """Lista entidades com paginação."""
        pass

    @abstractmethod
    async def count(self) -> int:
        """Conta total de entidades."""
        pass
