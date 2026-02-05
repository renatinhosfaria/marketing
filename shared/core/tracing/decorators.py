"""
Backwards compatibility layer.
Use shared.infrastructure.tracing.decorators instead.
"""
from shared.infrastructure.tracing.decorators import log_span

__all__ = ["log_span"]
