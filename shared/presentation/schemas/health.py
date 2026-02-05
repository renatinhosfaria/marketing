"""Health check schemas."""
from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Response for health check endpoint."""
    status: str = "healthy"
    service: str
    version: str


class DetailedHealthResponse(HealthResponse):
    """Detailed health response with component status."""
    database: str = "unknown"
    redis: str = "unknown"
