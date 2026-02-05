"""Error response schemas."""
from typing import Optional, Any
from pydantic import BaseModel


class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str
    message: str
    details: Optional[dict[str, Any]] = None


class ValidationErrorResponse(BaseModel):
    """Validation error response."""
    error: str = "validation_error"
    message: str = "Validation failed"
    errors: list[dict[str, Any]]
