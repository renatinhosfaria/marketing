"""
Configurações do microserviço ML.
Carrega variáveis de ambiente e define configurações globais.
"""

from functools import lru_cache
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Configurações da aplicação carregadas do ambiente."""

    # Aplicação
    app_name: str = "FamaChat ML"
    app_version: str = "1.0.0"
    debug: bool = False
    log_level: str = "INFO"
    environment: str = "development"

    # Database
    database_url: str = Field(
        default="postgresql://user:pass@localhost:5432/famachat",
        description="URL de conexão com o PostgreSQL"
    )
    database_pool_size: int = 5
    database_max_overflow: int = 10
    database_pool_recycle: int = 300  # Reciclar conexões a cada 5 minutos
    database_pool_timeout: int = 30  # Timeout para obter conexão do pool

    # Redis
    redis_url: str = Field(
        default="redis://localhost:6380/0",
        description="URL de conexão com o Redis"
    )

    # Segurança
    ml_api_key: str = Field(
        default="development-key-change-in-production",
        description="API Key para autenticação"
    )
    jwt_secret: str = Field(
        default="",
        description="Segredo JWT compartilhado com o famachat principal"
    )

    # Storage
    models_storage_path: str = "/app/models_storage"

    # Celery
    celery_broker_url: Optional[str] = None
    celery_result_backend: Optional[str] = None

    # Timezone
    timezone: str = "America/Sao_Paulo"

    # ML Configuration
    ml_default_horizon_days: int = 7
    ml_min_samples_for_training: int = 30
    ml_classification_tiers: list[str] = [
        "HIGH_PERFORMER",
        "MODERATE",
        "LOW",
        "UNDERPERFORMER"
    ]

    # Thresholds para recomendações (percentuais relativos à média)
    threshold_cpl_low: float = 0.7  # CPL < 70% da média = baixo
    threshold_cpl_high: float = 1.3  # CPL > 130% da média = alto
    threshold_ctr_good: float = 1.2  # CTR > 120% da média = bom
    threshold_frequency_high: float = 3.0  # Frequência > 3 = alta
    threshold_days_underperforming: int = 7  # Dias para considerar pausar

    # Limites de recursos
    max_concurrent_predictions: int = 10
    cache_ttl_seconds: int = 3600  # 1 hora

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    @property
    def celery_broker(self) -> str:
        """Retorna URL do broker Celery."""
        return self.celery_broker_url or self.redis_url

    @property
    def celery_backend(self) -> str:
        """Retorna URL do backend Celery."""
        return self.celery_result_backend or self.redis_url


@lru_cache()
def get_settings() -> Settings:
    """
    Retorna instância cacheada das configurações.
    Use esta função para obter as configurações em qualquer lugar da aplicação.
    """
    return Settings()


# Instância global para imports diretos
settings = get_settings()
