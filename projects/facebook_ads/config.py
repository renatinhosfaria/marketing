"""
Configurações do módulo Facebook Ads.
Carrega variáveis de ambiente específicas para integração com a API do Facebook/Meta.
"""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings


class FacebookAdsSettings(BaseSettings):
    """Configurações para integração com Facebook Ads API."""

    # OAuth / App
    facebook_app_id: str = Field(
        default="",
        description="ID do aplicativo Facebook"
    )
    facebook_app_secret: str = Field(
        default="",
        description="Secret do aplicativo Facebook"
    )
    facebook_api_version: str = Field(
        default="v24.0",
        description="Versão da Graph API do Facebook"
    )
    facebook_token_encryption_key: str = Field(
        default="",
        description="Chave de criptografia AES-256-GCM (64 hex chars = 32 bytes)"
    )
    facebook_oauth_callback_url: str = Field(
        default="",
        description="URL de callback OAuth do Facebook"
    )
    facebook_oauth_frontend_redirect_url: str = Field(
        default="",
        description="URL do frontend para retorno pós-OAuth"
    )
    facebook_oauth_scopes: str = Field(
        default="ads_read,ads_management",
        description="Escopos OAuth separados por vírgula"
    )

    # Rate Limiting
    facebook_rate_limit_threshold: int = Field(
        default=75,
        description="Percentual de uso da API para começar a aplicar delay (0-100)"
    )
    facebook_rate_limit_pause_threshold: int = Field(
        default=90,
        description="Percentual de uso da API para pausar requisições (0-100)"
    )

    # Sync
    facebook_sync_interval_hours: int = Field(
        default=1,
        description="Intervalo em horas entre sincronizações automáticas"
    )
    facebook_sync_backfill_days: int = Field(
        default=90,
        description="Dias de histórico para backfill inicial"
    )
    facebook_sync_async_threshold_days: int = Field(
        default=90,
        description="Limiar de dias para usar relatórios assíncronos"
    )

    # Token
    facebook_token_refresh_days_before: int = Field(
        default=14,
        description="Dias antes da expiração para renovar o token"
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"

    @property
    def graph_api_base_url(self) -> str:
        """URL base da Graph API do Facebook."""
        return f"https://graph.facebook.com/{self.facebook_api_version}"

    @property
    def oauth_scopes_list(self) -> list[str]:
        """Lista de escopos OAuth."""
        return self.facebook_oauth_scopes.split(",")


@lru_cache()
def get_facebook_ads_settings() -> FacebookAdsSettings:
    """
    Retorna instância cacheada das configurações do Facebook Ads.
    Use esta função para obter as configurações em qualquer lugar do módulo.
    """
    return FacebookAdsSettings()


# Instância global para imports diretos
fb_settings = get_facebook_ads_settings()
