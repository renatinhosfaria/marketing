"""Base DTOs for application layer."""
from datetime import datetime
from typing import TypeVar, Generic, Sequence, Optional
from pydantic import BaseModel, ConfigDict

T = TypeVar('T')


class BaseDTO(BaseModel):
    """Base DTO com configurações padrão."""
    model_config = ConfigDict(
        from_attributes=True,
        populate_by_name=True,
    )


class TimestampedDTO(BaseDTO):
    """DTO com timestamps."""
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class PaginatedDTO(BaseModel, Generic[T]):
    """DTO para respostas paginadas."""
    items: Sequence[T]
    total: int
    page: int
    page_size: int
    total_pages: int
    has_next: bool
    has_previous: bool

    model_config = ConfigDict(
        from_attributes=True,
    )

    @classmethod
    def from_page_response(cls, items: Sequence[T], total: int, page: int, page_size: int) -> "PaginatedDTO[T]":
        """Cria PaginatedDTO a partir de parâmetros."""
        total_pages = (total + page_size - 1) // page_size if page_size > 0 else 0
        return cls(
            items=items,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
            has_next=page < total_pages,
            has_previous=page > 1,
        )
