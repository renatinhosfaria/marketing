"""
Backwards compatibility layer.
Use shared.infrastructure.logging instead.
"""
from shared.infrastructure.logging.structlog_config import setup_logging, get_logger

__all__ = ["setup_logging", "get_logger"]
