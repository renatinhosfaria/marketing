"""
Configuração de logging estruturado com structlog.
"""

import logging
import sys
from typing import Optional

import structlog


def setup_logging(log_level: str = "INFO") -> None:
    """
    Configura logging estruturado para a aplicação.

    Args:
        log_level: Nível de log (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    # Configurar nível de log
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    # Configurar logging padrão do Python
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=numeric_level,
    )

    # Suprimir logs verbosos de bibliotecas HTTP (connect_tcp, start_tls, etc.)
    for noisy_logger in ("httpx", "httpcore", "httpcore.http11", "httpcore.connection"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)

    # Configurar structlog
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer()
            if log_level.upper() == "DEBUG"
            else structlog.processors.JSONRenderer()
        ],
        wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: Optional[str] = None) -> structlog.BoundLogger:
    """
    Obtém um logger configurado.

    Args:
        name: Nome do logger (geralmente __name__)

    Returns:
        Logger estruturado
    """
    return structlog.get_logger(name)
