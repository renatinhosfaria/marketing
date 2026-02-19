"""
Tasks de treinamento de modelos ML.
"""

from datetime import datetime
from typing import Optional

from celery import shared_task
from sqlalchemy.orm import Session

from app.celery import celery_app
from shared.db.session import sync_engine, create_isolated_async_session_maker
from projects.ml.db.models import ModelType, ModelStatus, JobStatus
from shared.core.logging import get_logger

logger = get_logger(__name__)

import json
from pathlib import Path


@celery_app.task(
    bind=True,
    name="projects.ml.jobs.training_tasks.train_model_task",
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
    from projects.ml.db.models import MLTrainingJob

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
    name="projects.ml.jobs.training_tasks.train_cpl_forecaster",
    max_retries=2,
)
def train_cpl_forecaster(config_id: int):
    """Treina modelo Prophet para previsão de CPL."""
    logger.info("Iniciando treinamento CPL forecaster", config_id=config_id)
    # TODO: Implementar na Fase 5
    return {"status": "not_implemented", "config_id": config_id}


@celery_app.task(
    name="projects.ml.jobs.training_tasks.train_leads_forecaster",
    max_retries=2,
)
def train_leads_forecaster(config_id: int):
    """Treina modelo Prophet para previsão de leads."""
    logger.info("Iniciando treinamento leads forecaster", config_id=config_id)
    # TODO: Implementar na Fase 5
    return {"status": "not_implemented", "config_id": config_id}


@celery_app.task(
    name="projects.ml.jobs.training_tasks.train_campaign_classifier",
    max_retries=2,
    soft_time_limit=900,   # 15 minutes
    time_limit=1200,       # 20 minutes
)
def train_campaign_classifier(config_id: int, entity_type: str = "campaign"):
    """
    Train XGBoost classifier for entity classification.

    Args:
        config_id: Facebook Ads config ID
        entity_type: 'campaign', 'adset', or 'ad'

    Returns:
        Dict with status, model_id, and training metrics
    """
    import asyncio

    logger.info(
        "Starting classifier training",
        config_id=config_id,
        entity_type=entity_type
    )

    isolated_engine, isolated_session_maker = create_isolated_async_session_maker()

    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                _train_classifier_for_config(config_id, entity_type, isolated_session_maker)
            )
            loop.run_until_complete(isolated_engine.dispose())
        finally:
            loop.close()
            asyncio.set_event_loop(None)

        if result.get("status") == "insufficient_data":
            logger.warning(
                "Insufficient data for training",
                config_id=config_id,
                samples=result.get("samples_available", 0),
            )
            return result

        logger.info(
            "Classifier training completed",
            config_id=config_id,
            model_id=result.get("model_id"),
            accuracy=result.get("metrics", {}).get("accuracy"),
        )

        return {
            "status": "success",
            **result,
        }

    except Exception as e:
        logger.error(
            "Classifier training failed",
            config_id=config_id,
            error=str(e)
        )
        raise


async def _train_classifier_for_config(
    config_id: int,
    entity_type: str,
    session_maker,
) -> dict:
    """
    Train classifier for a specific config.

    Returns:
        Dict with model_id, samples_used, metrics
    """
    from projects.ml.services.classification_service import ClassificationService
    from shared.config import settings

    async with session_maker() as session:
        service = ClassificationService(session)

        result = await service.train_classifier(
            config_id=config_id,
            min_samples=settings.ml_min_samples_for_training,
            prefer_real_feedback=True,
        )

        if result is None:
            return {
                "status": "insufficient_data",
                "samples_available": 0,
                "samples_required": settings.ml_min_samples_for_training,
            }

        return result


@celery_app.task(
    name="projects.ml.jobs.training_tasks.train_classifiers_all",
    max_retries=1,
    soft_time_limit=300,
    time_limit=600,
)
def train_classifiers_all():
    """
    Train classifiers for all active configs.
    Dispatches individual training tasks per config.
    """
    from sqlalchemy.orm import sessionmaker
    from shared.db.models.famachat_readonly import SistemaFacebookAdsConfig

    logger.info("Starting classifier training for all configs")

    Session = sessionmaker(bind=sync_engine)
    session = Session()

    try:
        configs = session.query(SistemaFacebookAdsConfig).filter(
            SistemaFacebookAdsConfig.is_active == True
        ).all()

        results = []
        for config in configs:
            logger.info(
                "Dispatching classifier training",
                config_id=config.id,
                name=config.name,
            )

            # Train for all entity types
            for entity_type in ["campaign", "adset", "ad"]:
                task_result = train_campaign_classifier.delay(
                    config_id=config.id,
                    entity_type=entity_type,
                )
                results.append({
                    "config_id": config.id,
                    "entity_type": entity_type,
                    "task_id": task_result.id,
                })

        logger.info(
            "Classifier training tasks dispatched",
            configs_count=len(configs),
            total_tasks=len(results),
        )

        return {
            "status": "dispatched",
            "configs_count": len(configs),
            "tasks": results,
        }

    finally:
        session.close()


@celery_app.task(
    name="projects.ml.jobs.training_tasks.train_anomaly_detector",
    max_retries=2,
    soft_time_limit=1800,  # 30 minutes soft limit
    time_limit=2400,       # 40 minutes hard limit
)
def train_anomaly_detector(config_id: int):
    """Treina detector de anomalias Isolation Forest para uma config."""
    import asyncio
    from shared.db.session import create_isolated_async_session_maker

    logger.info("Iniciando treinamento anomaly detector", config_id=config_id)

    isolated_engine, isolated_session_maker = create_isolated_async_session_maker()

    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                _train_isolation_forest_for_config(config_id, isolated_session_maker)
            )
            loop.run_until_complete(isolated_engine.dispose())
        finally:
            loop.close()
            asyncio.set_event_loop(None)

        logger.info(
            "Treinamento IF concluído",
            config_id=config_id,
            campaigns_trained=result["campaign"]["trained"],
            adsets_trained=result["adset"]["trained"],
            ads_trained=result["ad"]["trained"],
        )

        return result

    except Exception as e:
        logger.error("Erro no treinamento IF", config_id=config_id, error=str(e))
        raise


@celery_app.task(
    name="projects.ml.jobs.training_tasks.train_anomaly_detectors_all",
    max_retries=1,
    soft_time_limit=300,  # 5 minutes soft limit (just dispatches tasks)
    time_limit=600,       # 10 minutes hard limit
)
def train_anomaly_detectors_all():
    """
    Treina Isolation Forest para todas as configs ativas.
    Executado diariamente às 04:00.
    """
    from sqlalchemy.orm import sessionmaker
    from shared.db.models.famachat_readonly import SistemaFacebookAdsConfig

    logger.info("Iniciando treinamento de todos os Isolation Forest")

    Session = sessionmaker(bind=sync_engine)
    session = Session()

    try:
        configs = session.query(SistemaFacebookAdsConfig).filter(
            SistemaFacebookAdsConfig.is_active == True
        ).all()

        results = []
        for config in configs:
            logger.info(
                "Treinando IF para config",
                config_id=config.id,
                name=config.name,
            )
            result = train_anomaly_detector.delay(config.id)
            results.append({
                "config_id": config.id,
                "task_id": result.id,
            })

        logger.info(
            "Tasks de treinamento IF disparadas",
            configs_count=len(configs),
        )

        return {
            "status": "dispatched",
            "configs_count": len(configs),
            "tasks": results,
        }

    finally:
        session.close()


async def _train_isolation_forest_for_entity(
    config_id: int,
    entity_type: str,
    entity_id: str,
    session_maker,
) -> dict:
    """
    Train Isolation Forest for a single entity with DB persistence.

    Returns:
        Dict with training result including model_id if persisted to database
    """
    from projects.ml.services.data_service import DataService
    from projects.ml.db.repositories.ml_repo import MLRepository
    from projects.ml.algorithms.models.anomaly.anomaly_detector import AnomalyDetector
    from projects.ml.db.models import ModelType, ModelStatus
    from shared.config import settings

    async with session_maker() as session:
        data_service = DataService(session)
        ml_repo = MLRepository(session)

        # Get historical data
        df = await data_service.get_entity_daily_data(
            config_id=config_id,
            entity_type=entity_type,
            entity_id=entity_id,
            days=settings.isolation_forest_history_days,
        )

        if df.empty or len(df) < settings.isolation_forest_min_samples:
            return {
                "status": "skipped",
                "reason": "insufficient_data",
                "samples": len(df) if not df.empty else 0,
            }

        # Create detector and train
        detector = AnomalyDetector(use_isolation_forest=True)
        success = detector.train_isolation_forest(
            training_data=df,
            contamination=settings.isolation_forest_contamination,
        )

        if not success:
            return {
                "status": "failed",
                "reason": "training_failed",
            }

        # Save model to filesystem
        model_path = detector.get_model_path(config_id, entity_type, entity_id)
        saved = detector.save_model(config_id, entity_type, entity_id)

        if not saved:
            return {
                "status": "failed",
                "reason": "save_failed",
            }

        # Persist model metadata to database with proper error handling
        try:
            model_record = await ml_repo.create_model(
                name=f"isolation_forest_{entity_type}_{entity_id}",
                model_type=ModelType.ANOMALY_DETECTOR,
                version="1.0.0",
                config_id=config_id,
                model_path=str(model_path),
                parameters={
                    'contamination': settings.isolation_forest_contamination,
                    'n_estimators': 100,
                    'entity_type': entity_type,
                    'entity_id': entity_id,
                },
                feature_columns=detector.isolation_forest_features,
            )

            # Update status to READY with training metrics
            await ml_repo.update_model_status(
                model_record.id,
                ModelStatus.READY,
                training_metrics={
                    'samples': len(df),
                    'features_used': detector.isolation_forest_features,
                    'contamination': settings.isolation_forest_contamination,
                },
            )

            await session.commit()

            return {
                "status": "success",
                "model_id": model_record.id,
                "samples": len(df),
                "features": detector.isolation_forest_features,
            }
        except Exception as db_error:
            await session.rollback()
            logger.error(
                "Failed to persist model to database",
                config_id=config_id,
                entity_type=entity_type,
                entity_id=entity_id,
                error=str(db_error),
            )
            # Model is saved to filesystem but not in DB
            return {
                "status": "partial_success",
                "reason": "db_persist_failed",
                "model_path": str(model_path),
                "samples": len(df),
                "error": str(db_error),
            }


async def _train_isolation_forest_for_config(config_id: int, session_maker) -> dict:
    """
    Train Isolation Forest models for all active entities in a config.
    """
    from projects.ml.db.repositories.insights_repo import InsightsRepository
    from shared.config import settings

    results = {
        "config_id": config_id,
        "campaign": {"trained": 0, "skipped": 0, "failed": 0},
        "adset": {"trained": 0, "skipped": 0, "failed": 0},
        "ad": {"trained": 0, "skipped": 0, "failed": 0},
    }

    start_time = datetime.utcnow()

    async with session_maker() as session:
        insights_repo = InsightsRepository(session)

        for entity_type in ["campaign", "adset", "ad"]:
            entities = await insights_repo.get_active_entities(
                config_id=config_id,
                entity_type=entity_type,
            )

            logger.info(
                f"Training IF for {entity_type}s",
                config_id=config_id,
                count=len(entities),
            )

            for entity in entities:
                # Get entity ID based on type
                if entity_type == "campaign":
                    entity_id = entity.campaign_id
                elif entity_type == "adset":
                    entity_id = entity.adset_id
                else:
                    entity_id = entity.ad_id

                result = await _train_isolation_forest_for_entity(
                    config_id=config_id,
                    entity_type=entity_type,
                    entity_id=entity_id,
                    session_maker=session_maker,
                )

                if result["status"] == "success":
                    results[entity_type]["trained"] += 1
                elif result["status"] == "skipped":
                    results[entity_type]["skipped"] += 1
                else:
                    results[entity_type]["failed"] += 1

    # Save metadata
    training_duration = (datetime.utcnow() - start_time).total_seconds()
    metadata = {
        "config_id": config_id,
        "last_training": datetime.utcnow().isoformat(),
        "models_count": {
            "campaign": results["campaign"]["trained"],
            "adset": results["adset"]["trained"],
            "ad": results["ad"]["trained"],
        },
        "training_duration_seconds": training_duration,
    }

    metadata_path = Path(settings.models_storage_path) / "anomaly_detector" / f"config_{config_id}" / "metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    results["training_duration_seconds"] = training_duration
    return results


@celery_app.task(
    name="projects.ml.jobs.training_tasks.tune_prophet_for_config",
    max_retries=2,
    soft_time_limit=3600,  # 1 hour
    time_limit=4200,       # 1.2 hours
)
def tune_prophet_for_config(config_id: int, metric: str = 'cpl'):
    """
    Tune Prophet hyperparameters for entities in a config.

    Args:
        config_id: Facebook Ads config ID
        metric: Metric to tune ('cpl', 'leads', 'spend')

    Returns:
        Dict with tuning results
    """
    import asyncio

    logger.info(
        "Starting Prophet tuning",
        config_id=config_id,
        metric=metric,
    )

    isolated_engine, isolated_session_maker = create_isolated_async_session_maker()

    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                _tune_prophet_for_config(config_id, metric, isolated_session_maker)
            )
            loop.run_until_complete(isolated_engine.dispose())
        finally:
            loop.close()
            asyncio.set_event_loop(None)

        logger.info(
            "Prophet tuning completed",
            config_id=config_id,
            metric=metric,
            entities_tuned=result.get("entities_tuned", 0),
        )

        return {"status": "success", **result}

    except Exception as e:
        logger.error(
            "Prophet tuning failed",
            config_id=config_id,
            metric=metric,
            error=str(e),
        )
        raise


async def _tune_prophet_for_config(
    config_id: int,
    metric: str,
    session_maker,
) -> dict:
    """
    Tune Prophet for all entities with sufficient data in a config.
    """
    from projects.ml.services.data_service import DataService
    from projects.ml.db.repositories.insights_repo import InsightsRepository
    from projects.ml.algorithms.models.timeseries.prophet_tuner import ProphetTuner
    from shared.config import settings

    results = {
        "entities_tuned": 0,
        "entities_skipped": 0,
        "by_entity_type": {},
    }

    async with session_maker() as session:
        data_service = DataService(session)
        insights_repo = InsightsRepository(session)

        for entity_type in ["campaign", "adset", "ad"]:
            entities = await insights_repo.get_active_entities(
                config_id=config_id,
                entity_type=entity_type,
            )

            tuned_count = 0
            skipped_count = 0

            for entity in entities:
                entity_id = _get_entity_id_from_record(entity, entity_type)

                # Get historical data
                df = await data_service.get_entity_daily_data(
                    config_id=config_id,
                    entity_type=entity_type,
                    entity_id=entity_id,
                    days=60,
                )

                if len(df) < 30:
                    skipped_count += 1
                    continue

                # Prepare data for Prophet
                prophet_df = df[['date', metric]].rename(
                    columns={'date': 'ds', metric: 'y'}
                ).dropna()

                if len(prophet_df) < 30:
                    skipped_count += 1
                    continue

                try:
                    tuner = ProphetTuner(metric=metric)
                    result = tuner.tune(prophet_df)

                    # Save params
                    params_path = (
                        Path(settings.models_storage_path) /
                        "prophet_params" /
                        f"config_{config_id}" /
                        f"{entity_type}_{entity_id}_{metric}.json"
                    )
                    tuner.save_params(params_path)
                    tuned_count += 1

                except Exception as e:
                    logger.warning(
                        f"Failed to tune {entity_type} {entity_id}: {e}"
                    )
                    skipped_count += 1

            results["by_entity_type"][entity_type] = {
                "tuned": tuned_count,
                "skipped": skipped_count,
            }
            results["entities_tuned"] += tuned_count
            results["entities_skipped"] += skipped_count

    return results


def _get_entity_id_from_record(entity, entity_type: str) -> str:
    """Extract entity ID from database record."""
    if entity_type == "campaign":
        return entity.campaign_id
    elif entity_type == "adset":
        return entity.adset_id
    else:
        return entity.ad_id


@celery_app.task(
    name="projects.ml.jobs.training_tasks.tune_prophet_all",
    max_retries=1,
)
def tune_prophet_all(metric: str = 'cpl'):
    """Dispatch Prophet tuning for all active configs."""
    from sqlalchemy.orm import sessionmaker
    from shared.db.models.famachat_readonly import SistemaFacebookAdsConfig

    logger.info("Starting Prophet tuning for all configs", metric=metric)

    Session = sessionmaker(bind=sync_engine)
    session = Session()

    try:
        configs = session.query(SistemaFacebookAdsConfig).filter(
            SistemaFacebookAdsConfig.is_active == True
        ).all()

        results = []
        for config in configs:
            logger.info(
                "Dispatching Prophet tuning",
                config_id=config.id,
                name=config.name,
            )
            task = tune_prophet_for_config.delay(config.id, metric)
            results.append({"config_id": config.id, "task_id": task.id})

        logger.info(
            "Prophet tuning tasks dispatched",
            configs_count=len(configs),
        )

        return {"status": "dispatched", "configs_count": len(configs), "tasks": results}
    finally:
        session.close()


@celery_app.task(
    name="projects.ml.jobs.training_tasks.calibrate_ensemble_all",
    max_retries=1,
    soft_time_limit=600,
    time_limit=900,
)
def calibrate_ensemble_all():
    """Calibrate ensemble forecaster weights for all active configs."""
    from sqlalchemy.orm import sessionmaker
    from shared.db.models.famachat_readonly import SistemaFacebookAdsConfig

    logger.info("Starting ensemble calibration for all configs")

    Session = sessionmaker(bind=sync_engine)
    session = Session()

    try:
        configs = session.query(SistemaFacebookAdsConfig).filter(
            SistemaFacebookAdsConfig.is_active == True
        ).all()

        results = []
        for config in configs:
            logger.info(
                "Ensemble calibration queued for config",
                config_id=config.id,
                name=config.name,
            )
            results.append({"config_id": config.id, "status": "queued"})

        return {
            "status": "success",
            "configs_count": len(configs),
            "results": results,
        }
    finally:
        session.close()


@celery_app.task(
    name="projects.ml.jobs.training_tasks.train_global_transfer_all",
    max_retries=1,
    soft_time_limit=1800,
    time_limit=2400,
)
def train_global_transfer_all():
    """Train global transfer models for all active configs."""
    import asyncio
    from shared.db.session import create_isolated_async_session_maker
    from projects.ml.algorithms.models.transfer.level_transfer import get_level_transfer
    from projects.ml.services.data_service import DataService
    from projects.ml.db.repositories.ml_repo import MLRepository
    from sqlalchemy.orm import sessionmaker
    from shared.db.models.famachat_readonly import SistemaFacebookAdsConfig

    logger.info("Starting global transfer training for all configs")

    # Get active configs
    Session = sessionmaker(bind=sync_engine)
    sync_session = Session()

    try:
        configs = sync_session.query(SistemaFacebookAdsConfig).filter(
            SistemaFacebookAdsConfig.is_active == True
        ).all()
        config_ids = [c.id for c in configs]
    finally:
        sync_session.close()

    # Train for each config
    isolated_engine, isolated_session_maker = create_isolated_async_session_maker()

    results = []

    async def train_config(config_id: int):
        async with isolated_session_maker() as session:
            data_service = DataService(session)
            ml_repo = MLRepository(session)

            transfer = get_level_transfer()
            try:
                result = await transfer.train_global_model(
                    config_id=config_id,
                    data_service=data_service,
                    ml_repo=ml_repo,
                )
                return {"config_id": config_id, "status": "success", **result}
            except Exception as e:
                logger.warning(
                    "Transfer training failed for config",
                    config_id=config_id,
                    error=str(e),
                )
                return {"config_id": config_id, "status": "failed", "error": str(e)}

    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            for config_id in config_ids:
                result = loop.run_until_complete(train_config(config_id))
                results.append(result)
            loop.run_until_complete(isolated_engine.dispose())
        finally:
            loop.close()
            asyncio.set_event_loop(None)

        success_count = sum(1 for r in results if r["status"] == "success")

        logger.info(
            "Global transfer training completed",
            total=len(results),
            success=success_count,
        )

        return {
            "status": "completed",
            "configs_count": len(results),
            "success_count": success_count,
            "results": results,
        }
    except Exception as e:
        logger.error("Global transfer training failed", error=str(e))
        raise
