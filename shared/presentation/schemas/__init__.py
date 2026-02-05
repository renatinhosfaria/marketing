"""Shared presentation schemas."""
from .health import HealthResponse, DetailedHealthResponse
from .errors import ErrorResponse, ValidationErrorResponse

__all__ = [
    "HealthResponse",
    "DetailedHealthResponse",
    "ErrorResponse",
    "ValidationErrorResponse",
]
