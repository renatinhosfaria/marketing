"""ML persistence repositories.

Re-exports from the original db/repositories/ location.
"""
from projects.ml.db.repositories.ml_repo import MLRepository
from projects.ml.db.repositories.insights_repo import InsightsRepository

__all__ = [
    "MLRepository",
    "InsightsRepository",
]
