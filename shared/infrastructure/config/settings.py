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
    app_name: str = "Marketing"
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
    threshold_cpl_very_high: float = 1.5  # CPL > 150% da média = muito alto
    threshold_ctr_good: float = 1.2  # CTR > 120% da média = bom
    threshold_frequency_high: float = 3.0  # Frequência > 3 = alta
    threshold_days_underperforming: int = 7  # Dias para considerar pausar

    # Anomaly detection thresholds
    anomaly_z_threshold: float = 2.5  # Z-score threshold for anomaly detection
    anomaly_iqr_multiplier: float = 1.5  # IQR multiplier for outlier detection
    anomaly_min_history_days: int = 7  # Minimum days of history required

    # Frequency thresholds for audience fatigue detection
    frequency_low: float = 3.0  # Low severity threshold
    frequency_medium: float = 5.0  # Medium severity threshold
    frequency_high: float = 7.0  # High severity threshold
    frequency_critical: float = 10.0  # Critical severity threshold
    frequency_ideal: float = 2.5  # Ideal target frequency

    # Isolation Forest configuration
    use_isolation_forest: bool = True  # Enable by default
    isolation_forest_min_samples: int = 50  # Minimum samples to train
    isolation_forest_contamination: float = 0.1  # Expected anomaly proportion
    isolation_forest_history_days: int = 90  # Days of history for training

    # Celery worker configuration
    celery_worker_concurrency: int = 4  # Number of concurrent workers

    # Limites de recursos
    max_concurrent_predictions: int = 10
    cache_ttl_seconds: int = 3600  # 1 hora
    model_cache_ttl_seconds: int = 1800  # 30 minutos para cache de modelos

    # Rate limiting configuration
    rate_limit_enabled: bool = True
    rate_limit_requests_per_minute: int = 60  # Requests per minute per client
    rate_limit_requests_per_hour: int = 1000  # Requests per hour per client
    rate_limit_burst: int = 10  # Maximum burst requests

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"

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
