"""
Backwards compatibility layer.
Use shared.infrastructure.config instead.
"""
from shared.infrastructure.config.settings import Settings, get_settings, settings

__all__ = ["Settings", "get_settings", "settings"]
