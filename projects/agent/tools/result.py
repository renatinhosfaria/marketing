"""
Contrato de retorno padronizado para todas as tools.

ToolResult: TypedDict com ok, data, error.
tool_success(): helper para retorno de sucesso.
tool_error(): helper para retorno de erro.

Codigos de erro padronizados:
  - TIMEOUT: API nao respondeu no prazo
  - UNAVAILABLE: Servico indisponivel
  - HTTP_ERROR: Resposta HTTP nao-2xx
  - VALIDATION_ERROR: Parametros invalidos
  - OWNERSHIP_ERROR: Entidade nao pertence ao account
  - NOT_FOUND: Recurso nao encontrado
  - DB_ERROR: Erro de banco de dados
  - FB_API_ERROR: Erro da API do Facebook
  - SCHEMA_MISMATCH: Resposta incompativel com schema esperado
"""

from typing import TypedDict, Optional, Any


class ToolResult(TypedDict):
    """Contrato de retorno padronizado para todas as tools."""
    ok: bool                  # True = sucesso, False = falha
    data: Optional[Any]       # Payload de sucesso (dict, list, str)
    error: Optional[dict]     # Detalhes do erro (se ok=False)
    # error schema: {"code": str, "message": str, "retryable": bool}


def tool_success(data: Any) -> ToolResult:
    """Helper para retorno de sucesso."""
    return {"ok": True, "data": data, "error": None}


def tool_error(
    code: str,
    message: str,
    retryable: bool = False,
) -> ToolResult:
    """Helper para retorno de erro.

    Args:
        code: Codigo do erro (TIMEOUT, UNAVAILABLE, HTTP_ERROR, etc.)
        message: Mensagem descritiva do erro.
        retryable: Se o erro pode ser retentado.
    """
    return {
        "ok": False,
        "data": None,
        "error": {
            "code": code,
            "message": message,
            "retryable": retryable,
        },
    }
