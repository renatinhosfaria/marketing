"""
Helper para logging estruturado do Agent.

Centraliza a criacao do logger structlog para o modulo Agent,
garantindo namespace consistente em todos os eventos.
"""

import structlog
from typing import Any

logger = structlog.get_logger("agent")


def log_agent_event(event: str, **kwargs: Any):
    """Log estruturado para eventos do agent."""
    logger.info(event, **kwargs)


def log_agent_error(event: str, **kwargs: Any):
    """Log estruturado para erros do agent."""
    logger.error(event, **kwargs)
