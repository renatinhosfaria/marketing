"""Endpoints OAuth 2.0 do Facebook Ads."""

import base64
import json
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query, status
from fastapi.responses import RedirectResponse
from sqlalchemy.ext.asyncio import AsyncSession

from shared.db.session import get_db
from shared.core.logging import get_logger
from projects.facebook_ads.services.oauth_service import OAuthService
from projects.facebook_ads.schemas.base import camel_keys
from projects.facebook_ads.security.token_encryption import encrypt_token

logger = get_logger(__name__)
router = APIRouter()


@router.get("/url")
async def get_oauth_url():
    """Gera URL de autorização do Facebook."""
    from projects.facebook_ads.config import fb_settings

    if not fb_settings.facebook_app_id:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Facebook App não configurado",
        )

    service = OAuthService.__new__(OAuthService)
    result = service.get_authorization_url(1)
    return camel_keys({"success": True, **result})


@router.get("/callback")
async def oauth_callback(
    code: str = Query(...),
    state: str = Query(""),
    db: AsyncSession = Depends(get_db),
    accept: str = Header(""),
    user_agent: str = Header("", alias="User-Agent"),
):
    """Callback do Facebook OAuth. Recebe code e troca por token."""
    service = OAuthService(db)

    try:
        short_token_data = await service.exchange_code_for_token(code)
        short_token = short_token_data["access_token"]

        long_token_data = await service.exchange_for_long_lived_token(short_token)

        accounts = await service.get_available_ad_accounts(
            long_token_data["access_token"]
        )

        wants_redirect = "text/html" in (accept or "") or "Mozilla" in (user_agent or "")

        if wants_redirect:
            from projects.facebook_ads.config import fb_settings

            frontend_url = fb_settings.facebook_oauth_frontend_redirect_url
            if frontend_url:
                safe_accounts = base64.urlsafe_b64encode(
                    json.dumps(accounts).encode("utf-8")
                ).decode("utf-8")

                encrypted_token = encrypt_token(long_token_data["access_token"])
                temp_payload = {
                    "token": encrypted_token,
                    "expires_at": long_token_data.get("expires_at").isoformat()
                    if long_token_data.get("expires_at")
                    else None,
                }
                temp_data = base64.urlsafe_b64encode(
                    json.dumps(temp_payload).encode("utf-8")
                ).decode("utf-8")

                redirect_url = (
                    f"{frontend_url}?step=select_account"
                    f"&accounts={safe_accounts}"
                    f"&tempData={temp_data}"
                )
                return RedirectResponse(url=redirect_url, status_code=302)

        return camel_keys({
            "success": True,
            "token_expires_at": str(long_token_data.get("expires_at", "")),
            "available_accounts": accounts,
            "message": "OAuth concluído. Selecione uma ad account para configurar.",
        })
    except Exception as e:
        logger.error("Erro no OAuth callback", error=str(e))
        wants_redirect = "text/html" in (accept or "") or "Mozilla" in (user_agent or "")
        if wants_redirect:
            from projects.facebook_ads.config import fb_settings

            frontend_url = fb_settings.facebook_oauth_frontend_redirect_url
            if frontend_url:
                redirect_url = (
                    f"{frontend_url}?error=oauth_error"
                    f"&message={str(e)}"
                )
                return RedirectResponse(url=redirect_url, status_code=302)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Erro no fluxo OAuth: {str(e)}",
        )


@router.post("/refresh/{config_id}")
async def refresh_token(
    config_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Renova token de acesso manualmente."""
    service = OAuthService(db)

    try:
        result = await service.refresh_token(config_id)
        return camel_keys({"success": True, **result})
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.error("Erro ao renovar token", config_id=config_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro ao renovar token: {str(e)}",
        )


@router.get("/token-status/{config_id}")
async def get_token_status(
    config_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Verifica status do token de uma configuração."""
    service = OAuthService(db)

    try:
        result = await service.get_token_status(config_id)
        return camel_keys({"success": True, **result})
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.get("/ad-accounts")
async def get_ad_accounts(
    config_id: Optional[int] = Query(None, alias="configId"),
    db: AsyncSession = Depends(get_db),
):
    """Lista ad accounts disponíveis do token armazenado."""
    from shared.db.models.famachat_readonly import SistemaFacebookAdsConfig
    from sqlalchemy import select
    from projects.facebook_ads.security.token_encryption import decrypt_token

    if config_id:
        result = await db.execute(
            select(SistemaFacebookAdsConfig).where(
                SistemaFacebookAdsConfig.id == config_id
            )
        )
        config = result.scalar_one_or_none()
        if not config:
            raise HTTPException(status_code=404, detail="Configuração não encontrada")
        access_token = decrypt_token(config.access_token)
    else:
        result = await db.execute(
            select(SistemaFacebookAdsConfig)
            .where(SistemaFacebookAdsConfig.is_active == True)
            .order_by(SistemaFacebookAdsConfig.updated_at.desc())
            .limit(1)
        )
        config = result.scalar_one_or_none()
        if not config:
            raise HTTPException(
                status_code=404,
                detail="Nenhuma configuração ativa encontrada"
            )
        access_token = decrypt_token(config.access_token)

    service = OAuthService(db)
    accounts = await service.get_available_ad_accounts(access_token)

    return camel_keys({"success": True, "data": accounts})


@router.post("/complete")
async def complete_oauth(
    body: dict,
    db: AsyncSession = Depends(get_db),
):
    """Finaliza fluxo OAuth criando/atualizando configuração."""
    ad_account_id = body.get("adAccountId")
    account_name = body.get("accountName") or ad_account_id
    access_token = body.get("accessToken")

    if not ad_account_id:
        raise HTTPException(
            status_code=400,
            detail="adAccountId é obrigatório"
        )

    try:
        from shared.db.models.famachat_readonly import SistemaFacebookAdsConfig
        from sqlalchemy import select
        from projects.facebook_ads.config import fb_settings

        result = await db.execute(
            select(SistemaFacebookAdsConfig).where(
                SistemaFacebookAdsConfig.account_id == ad_account_id
            )
        )
        existing = result.scalar_one_or_none()

        if existing:
            if access_token:
                existing.access_token = encrypt_token(access_token)
            if account_name:
                existing.account_name = account_name
            existing.is_active = True
            existing.updated_at = datetime.utcnow()
            await db.commit()
            await db.refresh(existing)

            return camel_keys({
                "success": True,
                "data": {"id": existing.id, "ad_account_id": ad_account_id},
                "message": "Configuração atualizada com sucesso",
            })

        if not access_token:
            raise HTTPException(
                status_code=400,
                detail="accessToken é obrigatório para nova configuração"
            )

        new_config = SistemaFacebookAdsConfig(
            account_id=ad_account_id,
            account_name=account_name or ad_account_id,
            access_token=encrypt_token(access_token),
            token_expires_at=None,
            app_id=fb_settings.facebook_app_id,
            app_secret=fb_settings.facebook_app_secret,
            is_active=True,
            created_by=1,
        )
        db.add(new_config)
        await db.commit()
        await db.refresh(new_config)

        return camel_keys({
            "success": True,
            "data": {"id": new_config.id, "ad_account_id": ad_account_id},
            "message": "Configuração criada com sucesso",
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Erro ao completar OAuth", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao completar configuração: {str(e)}",
        )


@router.post("/complete-setup")
async def complete_oauth_setup(
    body: dict,
    db: AsyncSession = Depends(get_db),
):
    """Finaliza OAuth após callback redirecionado do navegador."""
    account_id = body.get("accountId")
    account_name = body.get("accountName") or account_id
    temp_data = body.get("tempData")

    if not account_id or not temp_data:
        raise HTTPException(
            status_code=400,
            detail="accountId e tempData são obrigatórios",
        )

    try:
        decoded = base64.urlsafe_b64decode(temp_data.encode("utf-8"))
        payload = json.loads(decoded.decode("utf-8"))
        encrypted_token = payload.get("token")
        expires_at_raw = payload.get("expires_at")

        if not encrypted_token:
            raise ValueError("token ausente em tempData")

        token_expires_at = (
            datetime.fromisoformat(expires_at_raw)
            if expires_at_raw
            else None
        )

        from shared.db.models.famachat_readonly import SistemaFacebookAdsConfig
        from sqlalchemy import select
        from projects.facebook_ads.config import fb_settings

        result = await db.execute(
            select(SistemaFacebookAdsConfig).where(
                SistemaFacebookAdsConfig.account_id == account_id
            )
        )
        existing = result.scalar_one_or_none()

        if existing:
            existing.access_token = encrypted_token
            existing.account_name = account_name
            existing.is_active = True
            existing.token_expires_at = token_expires_at
            existing.updated_at = datetime.utcnow()
            await db.commit()
            await db.refresh(existing)
            return camel_keys({
                "success": True,
                "data": {"id": existing.id, "ad_account_id": account_id},
                "message": "Configuração atualizada com sucesso",
            })

        new_config = SistemaFacebookAdsConfig(
            account_id=account_id,
            account_name=account_name or account_id,
            access_token=encrypted_token,
            token_expires_at=token_expires_at,
            app_id=fb_settings.facebook_app_id,
            app_secret=fb_settings.facebook_app_secret,
            is_active=True,
            created_by=1,
        )
        db.add(new_config)
        await db.commit()
        await db.refresh(new_config)

        return camel_keys({
            "success": True,
            "data": {"id": new_config.id, "ad_account_id": account_id},
            "message": "Configuração criada com sucesso",
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Erro ao completar OAuth setup", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao completar configuração: {str(e)}",
        )
