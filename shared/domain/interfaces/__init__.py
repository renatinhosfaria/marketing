"""Domain interfaces for Clean Architecture."""
from .repository import Repository, ReadOnlyRepository
from .unit_of_work import UnitOfWork

__all__ = [
    "Repository",
    "ReadOnlyRepository",
    "UnitOfWork",
]
