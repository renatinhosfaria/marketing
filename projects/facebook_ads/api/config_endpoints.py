"""Endpoints CRUD para configuração de contas Facebook Ads."""

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from shared.db.session import get_db
from shared.core.logging import get_logger
from shared.db.models.famachat_readonly import SistemaFacebookAdsConfig
from projects.facebook_ads.schemas.config import (
    ConfigResponse,
    ConfigCreateRequest,
    ConfigUpdateRequest,
    ConfigTestResponse,
)
from projects.facebook_ads.schemas.base import camel_keys
from projects.facebook_ads.security.token_encryption import encrypt_token, decrypt_token
from projects.facebook_ads.client.base import FacebookGraphClient
from projects.facebook_ads.config import fb_settings
from projects.facebook_ads.services.config_deletion import hard_delete_config

logger = get_logger(__name__)
router = APIRouter()


def _config_to_response(config: SistemaFacebookAdsConfig) -> dict:
    """Converte model para response dict (sem tokens) em camelCase."""
    now = datetime.utcnow()
    expires = config.token_expires_at

    if not expires:
        token_status = "unknown"
    elif expires < now:
        token_status = "expired"
    elif (expires - now).days < fb_settings.facebook_token_refresh_days_before:
        token_status = "expiring_soon"
    else:
        token_status = "valid"

    return camel_keys({
        "id": config.id,
        "account_id": config.account_id,
        "account_name": config.account_name,
        "is_active": config.is_active,
        "sync_enabled": config.sync_enabled,
        "sync_frequency_minutes": config.sync_frequency_minutes,
        "last_sync_at": config.last_sync_at,
        "token_expires_at": config.token_expires_at,
        "token_status": token_status,
        "created_at": config.created_at,
        "updated_at": config.updated_at,
    })


@router.get("")
async def list_configs(
    db: AsyncSession = Depends(get_db),
):
    """Lista todas as configurações de contas."""
    result = await db.execute(
        select(SistemaFacebookAdsConfig).order_by(SistemaFacebookAdsConfig.id)
    )
    configs = result.scalars().all()

    return {
        "success": True,
        "data": [_config_to_response(c) for c in configs],
        "total": len(configs),
    }


@router.get("/{config_id}")
async def get_config(
    config_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Busca uma configuração específica."""
    result = await db.execute(
        select(SistemaFacebookAdsConfig).where(
            SistemaFacebookAdsConfig.id == config_id
        )
    )
    config = result.scalar_one_or_none()

    if not config:
        raise HTTPException(status_code=404, detail="Configuração não encontrada")

    return {"success": True, "data": _config_to_response(config)}


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_config(
    request: ConfigCreateRequest,
    db: AsyncSession = Depends(get_db),
):
    """Cria nova configuração de conta."""
    encrypted_token = encrypt_token(request.access_token)

    config = SistemaFacebookAdsConfig(
        account_id=request.account_id,
        account_name=request.account_name,
        access_token=encrypted_token,
        app_id=request.app_id or fb_settings.facebook_app_id,
        app_secret=request.app_secret or fb_settings.facebook_app_secret,
        is_active=True,
        sync_enabled=request.sync_enabled,
        sync_frequency_minutes=request.sync_frequency_minutes,
        created_by=1,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )
    db.add(config)
    await db.commit()

    logger.info("Configuração criada", config_id=config.id, account_id=request.account_id)

    return {"success": True, "data": _config_to_response(config)}


@router.put("/{config_id}")
async def update_config(
    config_id: int,
    request: ConfigUpdateRequest,
    db: AsyncSession = Depends(get_db),
):
    """Atualiza configuração existente."""
    result = await db.execute(
        select(SistemaFacebookAdsConfig).where(
            SistemaFacebookAdsConfig.id == config_id
        )
    )
    config = result.scalar_one_or_none()

    if not config:
        raise HTTPException(status_code=404, detail="Configuração não encontrada")

    update_data = request.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(config, key, value)
    config.updated_at = datetime.utcnow()

    await db.commit()

    logger.info("Configuração atualizada", config_id=config_id)
    return {"success": True, "data": _config_to_response(config)}


@router.delete("/{config_id}")
async def delete_config(
    config_id: int,
    hard_delete: bool = Query(False, alias="hardDelete"),
    db: AsyncSession = Depends(get_db),
):
    """Desativa uma configuração (soft delete) ou exclui permanentemente."""
    result = await db.execute(
        select(SistemaFacebookAdsConfig).where(
            SistemaFacebookAdsConfig.id == config_id
        )
    )
    config = result.scalar_one_or_none()

    if not config:
        raise HTTPException(status_code=404, detail="Configuração não encontrada")

    if hard_delete:
        await hard_delete_config(db, config_id)
        await db.commit()

        logger.info("Configuração excluída permanentemente", config_id=config_id)
        return {"success": True, "message": "Configuração excluída permanentemente"}

    config.is_active = False
    config.sync_enabled = False
    config.updated_at = datetime.utcnow()
    await db.commit()

    logger.info("Configuração desativada", config_id=config_id)
    return {"success": True, "message": "Configuração desativada"}


@router.post("/{config_id}/test")
async def test_config(
    config_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Testa conexão com a conta do Facebook Ads."""
    result = await db.execute(
        select(SistemaFacebookAdsConfig).where(
            SistemaFacebookAdsConfig.id == config_id
        )
    )
    config = result.scalar_one_or_none()

    if not config:
        raise HTTPException(status_code=404, detail="Configuração não encontrada")

    try:
        access_token = decrypt_token(config.access_token)
        client = FacebookGraphClient(access_token, config.account_id)

        try:
            response = await client.get(
                config.account_id if config.account_id.startswith("act_") else f"act_{config.account_id}",
                params={"fields": "name,currency,timezone_name,account_status"},
            )

            return {
                "success": True,
                "data": ConfigTestResponse(
                    success=True,
                    account_name=response.get("name"),
                    currency=response.get("currency"),
                    timezone=response.get("timezone_name"),
                ).model_dump(by_alias=True),
            }
        finally:
            await client.close()

    except Exception as e:
        logger.error("Teste de conexão falhou", config_id=config_id, error=str(e))
        return {
            "success": False,
            "data": ConfigTestResponse(
                success=False,
                error=str(e),
            ).model_dump(by_alias=True),
        }


@router.get("/{config_id}/ad-accounts")
async def list_ad_accounts(
    config_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Lista ad accounts disponíveis para o token da configuração."""
    result = await db.execute(
        select(SistemaFacebookAdsConfig).where(
            SistemaFacebookAdsConfig.id == config_id
        )
    )
    config = result.scalar_one_or_none()

    if not config:
        raise HTTPException(status_code=404, detail="Configuração não encontrada")

    try:
        access_token = decrypt_token(config.access_token)
        from projects.facebook_ads.services.oauth_service import OAuthService
        service = OAuthService(db)
        accounts = await service.get_available_ad_accounts(access_token)

        return {"success": True, "data": accounts, "total": len(accounts)}
    except Exception as e:
        logger.error("Erro ao listar ad accounts", config_id=config_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )
