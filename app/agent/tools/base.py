"""
Base para tools do agente.
"""

from typing import Any, Callable
from functools import wraps
from langchain_core.tools import tool

from app.core.logging import get_logger

logger = get_logger(__name__)


def agent_tool(
    name: str | None = None,
    description: str | None = None,
    return_direct: bool = False
):
    """
    Decorator para criar tools do agente com logging automático.

    Args:
        name: Nome da tool (opcional, usa nome da função)
        description: Descrição da tool (opcional, usa docstring)
        return_direct: Se True, retorna resultado diretamente ao usuário

    Returns:
        Decorator que cria a tool
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            tool_name = name or func.__name__
            logger.debug(f"Executando tool: {tool_name}", args=args, kwargs=kwargs)

            try:
                result = await func(*args, **kwargs)
                logger.debug(f"Tool {tool_name} concluída", result_type=type(result).__name__)
                return result
            except Exception as e:
                logger.error(f"Erro na tool {tool_name}", error=str(e))
                return {"error": str(e), "tool": tool_name}

        # Aplica o decorator @tool do LangChain
        tool_decorator = tool(
            name=name,
            description=description or func.__doc__,
            return_direct=return_direct
        )
        return tool_decorator(wrapper)

    return decorator


def format_currency(value: float) -> str:
    """Formata valor como moeda brasileira."""
    return f"R$ {value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def format_percentage(value: float) -> str:
    """Formata valor como porcentagem."""
    return f"{value:.1f}%"


def format_number(value: int | float) -> str:
    """Formata número com separadores de milhar."""
    if isinstance(value, float):
        return f"{value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"{value:,}".replace(",", ".")
