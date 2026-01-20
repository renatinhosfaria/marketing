"""
Exceções customizadas do microserviço ML.
"""

from typing import Any, Optional


class MLServiceException(Exception):
    """Exceção base para erros do serviço ML."""

    def __init__(
        self,
        message: str,
        details: Optional[dict[str, Any]] = None
    ):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class ModelNotFoundException(MLServiceException):
    """Modelo ML não encontrado."""

    def __init__(self, model_id: int):
        super().__init__(
            message=f"Modelo com ID {model_id} não encontrado",
            details={"model_id": model_id}
        )


class ModelNotReadyException(MLServiceException):
    """Modelo ML não está pronto para inferência."""

    def __init__(self, model_id: int, status: str):
        super().__init__(
            message=f"Modelo {model_id} não está pronto. Status atual: {status}",
            details={"model_id": model_id, "status": status}
        )


class InsufficientDataException(MLServiceException):
    """Dados insuficientes para treinamento ou previsão."""

    def __init__(self, required: int, available: int, entity: str = "samples"):
        super().__init__(
            message=f"Dados insuficientes: {available} {entity} disponíveis, mínimo {required} necessários",
            details={"required": required, "available": available, "entity": entity}
        )


class TrainingException(MLServiceException):
    """Erro durante treinamento de modelo."""

    def __init__(self, model_type: str, error: str):
        super().__init__(
            message=f"Erro ao treinar modelo {model_type}: {error}",
            details={"model_type": model_type, "error": error}
        )


class PredictionException(MLServiceException):
    """Erro durante inferência/previsão."""

    def __init__(self, entity_id: str, error: str):
        super().__init__(
            message=f"Erro ao gerar previsão para {entity_id}: {error}",
            details={"entity_id": entity_id, "error": error}
        )


class ConfigNotFoundException(MLServiceException):
    """Configuração do Facebook Ads não encontrada."""

    def __init__(self, config_id: int):
        super().__init__(
            message=f"Configuração de FB Ads com ID {config_id} não encontrada",
            details={"config_id": config_id}
        )


class EntityNotFoundException(MLServiceException):
    """Entidade (campanha, adset, ad) não encontrada."""

    def __init__(self, entity_type: str, entity_id: str):
        super().__init__(
            message=f"{entity_type.capitalize()} com ID {entity_id} não encontrado",
            details={"entity_type": entity_type, "entity_id": entity_id}
        )


class ValidationException(MLServiceException):
    """Erro de validação de dados."""

    def __init__(self, field: str, message: str):
        super().__init__(
            message=f"Erro de validação em '{field}': {message}",
            details={"field": field}
        )
