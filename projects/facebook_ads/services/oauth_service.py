"""
Serviço de OAuth 2.0 para Facebook Ads.
Gerencia fluxo de autorização, troca de tokens e renovação.
"""

import secrets
from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from shared.core.logging import get_logger
from projects.facebook_ads.config import fb_settings
from projects.facebook_ads.client.base import FacebookGraphClient, FacebookAPIError
from projects.facebook_ads.security.token_encryption import encrypt_token, decrypt_token
from shared.db.models.famachat_readonly import SistemaFacebookAdsConfig

logger = get_logger(__name__)


class OAuthService:
    """Serviço para fluxo OAuth 2.0 do Facebook."""

    def __init__(self, db: AsyncSession):
        self.db = db

    def get_authorization_url(self, user_id: int) -> dict:
        """Gera URL de autorização do Facebook com state CSRF."""
        state = f"{user_id}:{secrets.token_urlsafe(32)}"

        scopes = fb_settings.facebook_oauth_scopes
        app_id = fb_settings.facebook_app_id
        callback = fb_settings.facebook_oauth_callback_url

        url = (
            f"https://www.facebook.com/{fb_settings.facebook_api_version}/dialog/oauth?"
            f"client_id={app_id}"
            f"&redirect_uri={callback}"
            f"&scope={scopes}"
            f"&state={state}"
            f"&response_type=code"
        )

        return {"url": url, "state": state}

    async def exchange_code_for_token(self, code: str) -> dict:
        """Troca authorization code por access token de curta duração."""
        client = FacebookGraphClient("", "oauth")
        try:
            response = await client.request(
                "GET",
                "oauth/access_token",
                params={
                    "client_id": fb_settings.facebook_app_id,
                    "client_secret": fb_settings.facebook_app_secret,
                    "redirect_uri": fb_settings.facebook_oauth_callback_url,
                    "code": code,
                },
                retry=False,
            )
            return {
                "access_token": response.get("access_token"),
                "token_type": response.get("token_type", "bearer"),
                "expires_in": response.get("expires_in", 0),
            }
        finally:
            await client.close()

    async def exchange_for_long_lived_token(self, short_token: str) -> dict:
        """Troca token de curta duração por token de longa duração (60 dias)."""
        client = FacebookGraphClient(short_token, "oauth")
        try:
            response = await client.request(
                "GET",
                "oauth/access_token",
                params={
                    "grant_type": "fb_exchange_token",
                    "client_id": fb_settings.facebook_app_id,
                    "client_secret": fb_settings.facebook_app_secret,
                    "fb_exchange_token": short_token,
                },
                retry=False,
            )

            expires_in = response.get("expires_in", 5184000)  # 60 days default
            expires_at = datetime.utcnow() + timedelta(seconds=expires_in)

            return {
                "access_token": response.get("access_token"),
                "token_type": response.get("token_type", "bearer"),
                "expires_in": expires_in,
                "expires_at": expires_at,
            }
        finally:
            await client.close()

    async def debug_token(self, access_token: str) -> dict:
        """Verifica informações do token (validade, scopes, etc)."""
        client = FacebookGraphClient(access_token, "oauth")
        try:
            response = await client.get(
                "debug_token",
                params={
                    "input_token": access_token,
                    "access_token": f"{fb_settings.facebook_app_id}|{fb_settings.facebook_app_secret}",
                },
            )
            data = response.get("data", {})
            return {
                "is_valid": data.get("is_valid", False),
                "expires_at": datetime.fromtimestamp(data["expires_at"]) if data.get("expires_at") else None,
                "scopes": data.get("scopes", []),
                "app_id": data.get("app_id"),
                "user_id": data.get("user_id"),
            }
        except FacebookAPIError:
            return {"is_valid": False, "error": "Não foi possível verificar o token"}
        finally:
            await client.close()

    async def refresh_token(self, config_id: int) -> dict:
        """Renova o token de acesso de uma configuração."""
        result = await self.db.execute(
            select(SistemaFacebookAdsConfig).where(
                SistemaFacebookAdsConfig.id == config_id
            )
        )
        config = result.scalar_one_or_none()

        if not config:
            raise ValueError(f"Configuração {config_id} não encontrada")

        current_token = decrypt_token(config.access_token)

        # Trocar por token de longa duração
        token_data = await self.exchange_for_long_lived_token(current_token)
        new_token = token_data["access_token"]
        encrypted_token = encrypt_token(new_token)

        # Atualizar no banco
        await self.db.execute(
            update(SistemaFacebookAdsConfig)
            .where(SistemaFacebookAdsConfig.id == config_id)
            .values(
                access_token=encrypted_token,
                token_expires_at=token_data["expires_at"],
                updated_at=datetime.utcnow(),
            )
        )
        await self.db.flush()

        logger.info(
            "Token renovado",
            config_id=config_id,
            expires_at=str(token_data["expires_at"]),
        )

        return {
            "success": True,
            "expires_at": token_data["expires_at"],
            "expires_in": token_data["expires_in"],
        }

    async def get_token_status(self, config_id: int) -> dict:
        """Verifica o status do token de uma configuração."""
        result = await self.db.execute(
            select(SistemaFacebookAdsConfig).where(
                SistemaFacebookAdsConfig.id == config_id
            )
        )
        config = result.scalar_one_or_none()

        if not config:
            raise ValueError(f"Configuração {config_id} não encontrada")

        now = datetime.utcnow()
        expires_at = config.token_expires_at

        if not expires_at:
            return {"status": "unknown", "expires_at": None}

        days_until_expiry = (expires_at - now).days

        if days_until_expiry < 0:
            status = "expired"
        elif days_until_expiry < fb_settings.facebook_token_refresh_days_before:
            status = "expiring_soon"
        else:
            status = "valid"

        return {
            "status": status,
            "expires_at": expires_at,
            "days_until_expiry": max(days_until_expiry, 0),
            "needs_refresh": days_until_expiry < fb_settings.facebook_token_refresh_days_before,
        }

    async def get_available_ad_accounts(self, access_token: str) -> list[dict]:
        """Lista ad accounts disponíveis para um token."""
        client = FacebookGraphClient(access_token, "oauth")
        try:
            results = await client.get_all(
                "me/adaccounts",
                params={
                    "fields": "id,name,account_id,currency,timezone_name,account_status",
                },
            )
            return results
        finally:
            await client.close()
