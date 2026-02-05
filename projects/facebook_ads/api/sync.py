"""Endpoints de sincronização do Facebook Ads."""

import asyncio
from datetime import timedelta
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from shared.db.session import get_db
from shared.core.logging import get_logger
from shared.db.models.famachat_readonly import SistemaFacebookAdsConfig
from projects.facebook_ads.models.sync import SistemaFacebookAdsSyncHistory, SyncStatus
from projects.facebook_ads.services.sync_service import SyncService
from projects.facebook_ads.schemas.sync import (
    SyncStartRequest,
    SyncStartResponse,
    SyncStatusResponse,
    SyncSummaryResponse,
    SyncProgressResponse,
    SyncHistoryResponse,
)

logger = get_logger(__name__)
router = APIRouter()


async def _run_sync_background(sync_id: int, sync_type: str, days_back: Optional[int] = None):
    """Executa sync em background com sessão isolada."""
    from shared.db.session import isolated_async_session

    async with isolated_async_session() as db:
        service = SyncService(db)
        try:
            await service.execute_sync(sync_id, sync_type, days_back)
        except Exception as e:
            logger.error("Erro no sync background", sync_id=sync_id, error=str(e))


@router.post("/{config_id}", status_code=status.HTTP_202_ACCEPTED)
async def start_sync(
    config_id: int,
    request: SyncStartRequest = SyncStartRequest(),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    db: AsyncSession = Depends(get_db),
):
    """Inicia sincronização em background."""
    service = SyncService(db)

    try:
        sync_history = await service.start_sync(
            config_id=config_id,
            sync_type=request.sync_type,
            days_back=request.days_back,
            date_range_start=request.date_range_start,
            date_range_end=request.date_range_end,
        )

        # Executar em background
        background_tasks.add_task(
            _run_sync_background,
            sync_history.id,
            request.sync_type,
            request.days_back,
        )

        return SyncStartResponse(
            sync_id=sync_history.id,
            status="pending",
            message=f"Sincronização {request.sync_type} iniciada",
        ).model_dump(by_alias=True)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))


@router.get("/{config_id}/status")
async def get_sync_status(
    config_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Obtém status da última sincronização."""
    result = await db.execute(
        select(SistemaFacebookAdsConfig).where(SistemaFacebookAdsConfig.id == config_id)
    )
    config = result.scalar_one_or_none()

    if not config:
        raise HTTPException(status_code=404, detail="Configuração não encontrada")

    service = SyncService(db)
    items, _ = await service.get_sync_history(config_id, limit=1)
    last_sync = items[0] if items else None

    running_result = await db.execute(
        select(SistemaFacebookAdsSyncHistory)
        .where(
            SistemaFacebookAdsSyncHistory.config_id == config_id,
            SistemaFacebookAdsSyncHistory.status == SyncStatus.RUNNING.value,
        )
        .order_by(desc(SistemaFacebookAdsSyncHistory.started_at))
        .limit(1)
    )
    running_sync = running_result.scalar_one_or_none()

    is_running = running_sync is not None
    progress = None
    if running_sync:
        progress = SyncProgressResponse(
            stage=running_sync.sync_type,
            campaigns_synced=running_sync.campaigns_synced or 0,
            adsets_synced=running_sync.adsets_synced or 0,
            ads_synced=running_sync.ads_synced or 0,
            insights_synced=running_sync.insights_synced or 0,
        )

    next_sync_at = None
    if config.sync_enabled and config.last_sync_at and config.sync_frequency_minutes:
        next_sync_at = config.last_sync_at + timedelta(
            minutes=config.sync_frequency_minutes
        )

    summary = SyncSummaryResponse(
        config_id=config.id,
        account_name=config.account_name,
        is_running=is_running,
        progress=progress,
        last_sync=SyncStatusResponse.model_validate(last_sync) if last_sync else None,
        last_sync_at=config.last_sync_at,
        next_sync_at=next_sync_at,
        sync_enabled=config.sync_enabled,
        sync_frequency_minutes=config.sync_frequency_minutes,
    )

    return {"success": True, "data": summary.model_dump(by_alias=True)}


@router.get("/{config_id}/history")
async def get_sync_history(
    config_id: int,
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
):
    """Lista histórico de sincronizações."""
    service = SyncService(db)
    items, total = await service.get_sync_history(config_id, limit, offset)

    return {
        "success": True,
        "data": [SyncStatusResponse.model_validate(item).model_dump(by_alias=True) for item in items],
        "pagination": {
            "total": total,
            "limit": limit,
            "offset": offset,
            "hasMore": offset + limit < total,
        },
    }


@router.post("/{config_id}/cancel")
async def cancel_sync(
    config_id: int,
    sync_id: Optional[int] = Query(None, alias="syncId"),
    db: AsyncSession = Depends(get_db),
):
    """Cancela sincronização em andamento."""
    service = SyncService(db)
    target_sync_id = sync_id
    if target_sync_id is None:
        result = await db.execute(
            select(SistemaFacebookAdsSyncHistory.id)
            .where(
                SistemaFacebookAdsSyncHistory.config_id == config_id,
                SistemaFacebookAdsSyncHistory.status == SyncStatus.RUNNING.value,
            )
            .order_by(desc(SistemaFacebookAdsSyncHistory.started_at))
            .limit(1)
        )
        target_sync_id = result.scalar_one_or_none()

    if target_sync_id is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Nenhuma sincronização em andamento encontrada",
        )

    success = await service.cancel_sync(target_sync_id)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Sincronização não encontrada ou não está em andamento",
        )

    return {"success": True, "message": "Sincronização cancelada"}


@router.post("/{config_id}/today")
async def sync_today(
    config_id: int,
    background_tasks: BackgroundTasks = BackgroundTasks(),
    db: AsyncSession = Depends(get_db),
):
    """Sincroniza apenas insights de hoje."""
    service = SyncService(db)

    try:
        sync_history = await service.start_sync(config_id, "today_only")
        background_tasks.add_task(_run_sync_background, sync_history.id, "today_only")

        return SyncStartResponse(
            sync_id=sync_history.id,
            status="pending",
            message="Sincronização de hoje iniciada",
        ).model_dump(by_alias=True)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))


@router.post("/{config_id}/historical")
async def sync_historical(
    config_id: int,
    days_back: int = Query(90, ge=1, le=365, alias="daysBack"),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    db: AsyncSession = Depends(get_db),
):
    """Sincroniza insights históricos."""
    service = SyncService(db)

    try:
        sync_history = await service.start_sync(config_id, "historical", days_back)
        background_tasks.add_task(_run_sync_background, sync_history.id, "historical", days_back)

        return SyncStartResponse(
            sync_id=sync_history.id,
            status="pending",
            message=f"Sincronização histórica ({days_back} dias) iniciada",
        ).model_dump(by_alias=True)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
