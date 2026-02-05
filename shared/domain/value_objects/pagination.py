"""Pagination value objects."""
from dataclasses import dataclass
from typing import TypeVar, Generic, Sequence

T = TypeVar('T')


@dataclass(frozen=True)
class PageRequest:
    """Requisição de paginação."""
    page: int = 1
    page_size: int = 20

    def __post_init__(self):
        if self.page < 1:
            raise ValueError("Page must be >= 1")
        if self.page_size < 1 or self.page_size > 100:
            raise ValueError("Page size must be between 1 and 100")

    @property
    def offset(self) -> int:
        """Calcula offset para a query."""
        return (self.page - 1) * self.page_size

    @property
    def limit(self) -> int:
        """Retorna o limit para a query."""
        return self.page_size


@dataclass(frozen=True)
class PageResponse(Generic[T]):
    """Resposta paginada."""
    items: Sequence[T]
    total: int
    page: int
    page_size: int

    @property
    def total_pages(self) -> int:
        """Calcula número total de páginas."""
        if self.page_size == 0:
            return 0
        return (self.total + self.page_size - 1) // self.page_size

    @property
    def has_next(self) -> bool:
        """Verifica se há próxima página."""
        return self.page < self.total_pages

    @property
    def has_previous(self) -> bool:
        """Verifica se há página anterior."""
        return self.page > 1
