"""ML API routes.

Re-exports from the original api/ location.
"""
from projects.ml.api import health
from projects.ml.api import predictions
from projects.ml.api import forecasts
from projects.ml.api import classifications
from projects.ml.api import recommendations
from projects.ml.api import anomalies
from projects.ml.api import models

__all__ = [
    "health",
    "predictions",
    "forecasts",
    "classifications",
    "recommendations",
    "anomalies",
    "models",
]
