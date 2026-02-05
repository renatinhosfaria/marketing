"""
Endpoints de gestão de modelos ML.
"""

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status, Query
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from shared.db.session import get_db
from projects.ml.db.repositories.ml_repo import MLRepository
from projects.ml.db.models import ModelType, ModelStatus, JobStatus, MLTrainedModel
from shared.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


# ==================== SCHEMAS ====================

class ModelResponse(BaseModel):
    """Resposta de modelo."""
    id: int
    name: str
    model_type: str
    version: str
    config_id: Optional[int] = None
    model_path: str
    parameters: Optional[dict] = None
    feature_columns: Optional[list] = None
    training_metrics: Optional[dict] = None
    validation_metrics: Optional[dict] = None
    status: str
    is_active: bool
    training_data_start: Optional[datetime] = None
    training_data_end: Optional[datetime] = None
    samples_count: Optional[int] = None
    created_at: datetime
    trained_at: Optional[datetime] = None
    last_used_at: Optional[datetime] = None


class ModelListResponse(BaseModel):
    """Lista de modelos."""
    total: int
    by_type: dict[str, int]
    by_status: dict[str, int]
    models: list[ModelResponse]


class TrainRequest(BaseModel):
    """Request para treinar modelo."""
    model_type: str = Field(..., description="Tipo do modelo")
    config_id: Optional[int] = Field(None, description="ID da configuração FB Ads")
    parameters: Optional[dict] = Field(None, description="Hiperparâmetros")


class TrainResponse(BaseModel):
    """Resposta de solicitação de treinamento."""
    job_id: int
    model_type: str
    status: str
    message: str


class JobResponse(BaseModel):
    """Resposta de job de treinamento."""
    id: int
    model_type: str
    config_id: Optional[int] = None
    celery_task_id: Optional[str] = None
    status: str
    progress: float
    model_id: Optional[int] = None
    error_message: Optional[str] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


# ==================== ENDPOINTS ====================

@router.get("", response_model=ModelListResponse)
async def list_models(
    model_type: Optional[str] = Query(None, description="Tipo do modelo"),
    status: Optional[str] = Query(None, description="Status do modelo"),
    active_only: bool = Query(False, description="Apenas modelos ativos"),
    db: AsyncSession = Depends(get_db),
):
    """
    Lista todos os modelos ML registrados.
    """
    query = select(MLTrainedModel)

    type_filter = None
    if model_type:
        try:
            type_filter = ModelType(model_type)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Tipo de modelo inválido: {model_type}",
            )
        query = query.where(MLTrainedModel.model_type == type_filter)

    status_filter = None
    if status:
        try:
            status_filter = ModelStatus(status)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Status inválido: {status}",
            )
        query = query.where(MLTrainedModel.status == status_filter)

    if active_only:
        query = query.where(MLTrainedModel.is_active == True)

    query = query.order_by(MLTrainedModel.created_at.desc()).limit(200)
    result = await db.execute(query)
    models = list(result.scalars().all())

    by_type = {}
    by_status = {}
    for model in models:
        t = model.model_type.value
        by_type[t] = by_type.get(t, 0) + 1
        s = model.status.value
        by_status[s] = by_status.get(s, 0) + 1

    return ModelListResponse(
        total=len(models),
        by_type=by_type,
        by_status=by_status,
        models=[
            ModelResponse(
                id=m.id,
                name=m.name,
                model_type=m.model_type.value,
                version=m.version,
                config_id=m.config_id,
                model_path=m.model_path,
                parameters=m.parameters,
                feature_columns=m.feature_columns,
                training_metrics=m.training_metrics,
                validation_metrics=m.validation_metrics,
                status=m.status.value,
                is_active=m.is_active,
                training_data_start=m.training_data_start,
                training_data_end=m.training_data_end,
                samples_count=m.samples_count,
                created_at=m.created_at,
                trained_at=m.trained_at,
                last_used_at=m.last_used_at,
            )
            for m in models
        ],
    )


@router.get("/{model_id}", response_model=ModelResponse)
async def get_model(
    model_id: int,
    db: AsyncSession = Depends(get_db),
):
    """
    Obtém detalhes de um modelo específico.
    """
    ml_repo = MLRepository(db)

    model = await ml_repo.get_model(model_id)
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Modelo {model_id} não encontrado"
        )

    return ModelResponse(
        id=model.id,
        name=model.name,
        model_type=model.model_type.value,
        version=model.version,
        config_id=model.config_id,
        model_path=model.model_path,
        parameters=model.parameters,
        feature_columns=model.feature_columns,
        training_metrics=model.training_metrics,
        validation_metrics=model.validation_metrics,
        status=model.status.value,
        is_active=model.is_active,
        training_data_start=model.training_data_start,
        training_data_end=model.training_data_end,
        samples_count=model.samples_count,
        created_at=model.created_at,
        trained_at=model.trained_at,
        last_used_at=model.last_used_at
    )


@router.post("/train", response_model=TrainResponse)
async def train_model(
    request: TrainRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Inicia treinamento de um modelo.

    Cria um job de treinamento que será executado pelo Celery worker.
    """
    ml_repo = MLRepository(db)

    # Validar tipo de modelo
    try:
        model_type = ModelType(request.model_type)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Tipo de modelo inválido: {request.model_type}. "
                   f"Opções: {[t.value for t in ModelType]}"
        )

    # Criar job de treinamento
    job = await ml_repo.create_training_job(
        model_type=model_type,
        config_id=request.config_id
    )

    logger.info(
        "Job de treinamento criado",
        job_id=job.id,
        model_type=request.model_type,
        config_id=request.config_id
    )

    # TODO: Disparar task Celery
    # from projects.ml.jobs.training_tasks import train_model_task
    # result = train_model_task.delay(job.id, request.parameters)
    # job.celery_task_id = result.id

    return TrainResponse(
        job_id=job.id,
        model_type=model_type.value,
        status=job.status.value,
        message="Job de treinamento criado. Implementação completa na Fase 2."
    )


@router.get("/jobs/{job_id}", response_model=JobResponse)
async def get_training_job(
    job_id: int,
    db: AsyncSession = Depends(get_db),
):
    """
    Obtém status de um job de treinamento.
    """
    ml_repo = MLRepository(db)

    job = await ml_repo.get_training_job(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} não encontrado"
        )

    return JobResponse(
        id=job.id,
        model_type=job.model_type.value,
        config_id=job.config_id,
        celery_task_id=job.celery_task_id,
        status=job.status.value,
        progress=job.progress,
        model_id=job.model_id,
        error_message=job.error_message,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at
    )


@router.post("/{model_id}/activate")
async def activate_model(
    model_id: int,
    db: AsyncSession = Depends(get_db),
):
    """
    Ativa um modelo para uso em produção.

    Desativa outros modelos do mesmo tipo e ativa o especificado.
    """
    ml_repo = MLRepository(db)

    model = await ml_repo.get_model(model_id)
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Modelo {model_id} não encontrado"
        )

    if model.status != ModelStatus.READY:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Modelo deve estar com status READY para ser ativado. "
                   f"Status atual: {model.status.value}"
        )

    await ml_repo.activate_model(model_id, model.model_type)

    logger.info(
        "Modelo ativado",
        model_id=model_id,
        model_type=model.model_type.value
    )

    return {"status": "activated", "model_id": model_id}
