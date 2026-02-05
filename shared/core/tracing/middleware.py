"""
Backwards compatibility layer.
Use shared.infrastructure.tracing.middleware instead.
"""
from shared.infrastructure.tracing.middleware import TraceMiddleware

__all__ = ["TraceMiddleware"]
