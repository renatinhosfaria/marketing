"""
Tasks de treinamento de modelos ML.
"""

from datetime import datetime
from typing import Optional

from celery import shared_task
from sqlalchemy.orm import Session

from app.tasks.celery_app import celery_app
from app.db.session import sync_engine
from app.db.models.ml_models import ModelType, ModelStatus, JobStatus
from app.core.logging import get_logger

logger = get_logger(__name__)


@celery_app.task(
    bind=True,
    name="app.tasks.training_tasks.train_model_task",
    max_retries=3,
    default_retry_delay=60,
    autoretry_for=(Exception,),
    acks_late=True,
)
def train_model_task(
    self,
    job_id: int,
    parameters: Optional[dict] = None,
):
    """
    Task para treinar um modelo ML.

    Args:
        job_id: ID do job de treinamento
        parameters: Hiperparâmetros opcionais
    """
    from sqlalchemy.orm import sessionmaker
    from app.db.models.ml_models import MLTrainingJob

    Session = sessionmaker(bind=sync_engine)
    session = Session()

    try:
        # Obter job
        job = session.query(MLTrainingJob).get(job_id)
        if not job:
            logger.error("Job não encontrado", job_id=job_id)
            return {"status": "error", "message": "Job não encontrado"}

        # Atualizar status
        job.status = JobStatus.RUNNING
        job.started_at = datetime.utcnow()
        job.celery_task_id = self.request.id
        session.commit()

        logger.info(
            "Iniciando treinamento",
            job_id=job_id,
            model_type=job.model_type.value
        )

        # TODO: Implementar treinamento real por tipo de modelo
        # Por enquanto, simula processamento
        self.update_state(state="PROGRESS", meta={"progress": 0.5})

        # Simular conclusão
        job.status = JobStatus.COMPLETED
        job.progress = 1.0
        job.completed_at = datetime.utcnow()
        session.commit()

        logger.info(
            "Treinamento concluído",
            job_id=job_id,
            model_type=job.model_type.value
        )

        return {
            "status": "completed",
            "job_id": job_id,
            "model_type": job.model_type.value,
        }

    except Exception as e:
        logger.error(
            "Erro no treinamento",
            job_id=job_id,
            error=str(e)
        )

        # Atualizar job com erro
        if job:
            job.status = JobStatus.FAILED
            job.error_message = str(e)
            job.completed_at = datetime.utcnow()
            session.commit()

        raise

    finally:
        session.close()


@celery_app.task(
    name="app.tasks.training_tasks.train_cpl_forecaster",
    max_retries=2,
)
def train_cpl_forecaster(config_id: int):
    """Treina modelo Prophet para previsão de CPL."""
    logger.info("Iniciando treinamento CPL forecaster", config_id=config_id)
    # TODO: Implementar na Fase 5
    return {"status": "not_implemented", "config_id": config_id}


@celery_app.task(
    name="app.tasks.training_tasks.train_leads_forecaster",
    max_retries=2,
)
def train_leads_forecaster(config_id: int):
    """Treina modelo Prophet para previsão de leads."""
    logger.info("Iniciando treinamento leads forecaster", config_id=config_id)
    # TODO: Implementar na Fase 5
    return {"status": "not_implemented", "config_id": config_id}


@celery_app.task(
    name="app.tasks.training_tasks.train_campaign_classifier",
    max_retries=2,
)
def train_campaign_classifier(config_id: int):
    """Treina classificador de campanhas XGBoost/LightGBM."""
    logger.info("Iniciando treinamento classifier", config_id=config_id)
    # TODO: Implementar na Fase 4
    return {"status": "not_implemented", "config_id": config_id}


@celery_app.task(
    name="app.tasks.training_tasks.train_anomaly_detector",
    max_retries=2,
)
def train_anomaly_detector(config_id: int):
    """Treina detector de anomalias Isolation Forest."""
    logger.info("Iniciando treinamento anomaly detector", config_id=config_id)
    # TODO: Implementar na Fase 6
    return {"status": "not_implemented", "config_id": config_id}
