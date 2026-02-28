"""
Backwards compatibility layer.
Use shared.infrastructure.tracing.events instead.
"""
from shared.infrastructure.tracing.events import (
    log_tool_call,
    log_tool_call_error,
)

__all__ = [
    "log_tool_call",
    "log_tool_call_error",
]
