"""Guardrails de seguranca para SQL gerado pelo agente FB Ads."""

from dataclasses import dataclass
import re


class SQLGuardError(Exception):
    """Erro de validacao de SQL pelo guardrail."""


@dataclass
class SQLValidationResult:
    """Resultado da validacao de SQL."""

    operation_type: str


class SQLGuard:
    """Valida SQL antes da execucao no banco."""

    FORBIDDEN_OPS = {"DROP", "TRUNCATE"}

    @staticmethod
    def validate(sql: str) -> SQLValidationResult:
        normalized = " ".join(sql.strip().split())
        if not normalized:
            raise SQLGuardError("SQL vazio")

        op = normalized.split(" ", 1)[0].upper()
        upper_sql = normalized.upper()

        if op in SQLGuard.FORBIDDEN_OPS:
            raise SQLGuardError(f"Operacao proibida: {op}")

        # Sem comentarios SQL para reduzir superficie de bypass.
        if re.search(r"/\\*|\\*/|--", normalized):
            raise SQLGuardError("Comentarios SQL nao permitidos")

        # Bloqueia batch de multiplas instrucoes em uma unica chamada.
        if ";" in normalized[:-1]:
            raise SQLGuardError("Multiplas instrucoes nao permitidas")

        if op == "DELETE" and " WHERE " not in f" {upper_sql} ":
            raise SQLGuardError("DELETE sem WHERE bloqueado")

        if op == "UPDATE" and " WHERE " not in f" {upper_sql} ":
            raise SQLGuardError("UPDATE sem WHERE bloqueado")

        return SQLValidationResult(operation_type=op)
