"""
Tasks agendadas (Celery Beat).
Executam automaticamente conforme cronograma definido em celery_app.py.
"""

from datetime import datetime, timedelta
import asyncio
from dataclasses import asdict

from app.tasks.celery_app import celery_app
from app.db.session import sync_engine, async_session_maker
from app.core.logging import get_logger

logger = get_logger(__name__)

def _serialize_features(features) -> dict:
    data = asdict(features)
    for key, value in list(data.items()):
        if isinstance(value, datetime):
            data[key] = value.isoformat()
    return data


async def _run_compute_features_for_config(config_id: int, window_days: int = 30) -> dict:
    from app.db.repositories.insights_repo import InsightsRepository
    from app.db.repositories.ml_repo import MLRepository
    from app.services.data_service import DataService

    async with async_session_maker() as session:
        insights_repo = InsightsRepository(session)
        ml_repo = MLRepository(session)
        data_service = DataService(session)

        campaigns = await insights_repo.get_active_campaigns(config_id)
        feature_date = datetime.utcnow().date()

        processed = 0
        inserted = 0
        skipped = 0
        insufficient = 0

        for campaign in campaigns:
            processed += 1
            exists = await ml_repo.feature_exists(
                config_id=config_id,
                campaign_id=campaign.campaign_id,
                window_days=window_days,
                feature_date=feature_date,
            )
            if exists:
                skipped += 1
                continue

            features = await data_service.get_campaign_features(
                config_id=config_id,
                campaign_id=campaign.campaign_id,
                days=window_days,
            )
            if features is None:
                await ml_repo.create_feature(
                    config_id=config_id,
                    campaign_id=campaign.campaign_id,
                    window_days=window_days,
                    feature_date=feature_date,
                    features=None,
                    insufficient_data=True,
                )
                insufficient += 1
            else:
                await ml_repo.create_feature(
                    config_id=config_id,
                    campaign_id=campaign.campaign_id,
                    window_days=window_days,
                    feature_date=feature_date,
                    features=_serialize_features(features),
                    insufficient_data=False,
                )
                inserted += 1

        await session.commit()

        return {
            "processed": processed,
            "inserted": inserted,
            "skipped": skipped,
            "insufficient": insufficient,
        }


async def _run_forecasts_for_config(config_id: int, window_days: int = 30) -> dict:
    from app.db.repositories.insights_repo import InsightsRepository
    from app.db.repositories.ml_repo import MLRepository
    from app.services.data_service import DataService
    from app.ml.models.timeseries.forecaster import forecaster
    from app.config import settings

    async with async_session_maker() as session:
        insights_repo = InsightsRepository(session)
        ml_repo = MLRepository(session)
        data_service = DataService(session)

        campaigns = await insights_repo.get_active_campaigns(config_id)
        forecast_date = datetime.utcnow().date()
        horizon_days = settings.ml_default_horizon_days

        generated = 0
        skipped = 0
        insufficient = 0

        for campaign in campaigns:
            df = await data_service.get_campaign_daily_data(
                config_id=config_id,
                campaign_id=campaign.campaign_id,
                days=window_days,
            )
            if df.empty or len(df) < 7:
                exists = await ml_repo.forecast_exists(
                    config_id=config_id,
                    entity_type="campaign",
                    entity_id=campaign.campaign_id,
                    target_metric="cpl",
                    horizon_days=horizon_days,
                    forecast_date=forecast_date,
                )
                if not exists:
                    await ml_repo.create_forecast(
                        config_id=config_id,
                        entity_type="campaign",
                        entity_id=campaign.campaign_id,
                        target_metric="cpl",
                        horizon_days=horizon_days,
                        method=forecaster.method,
                        predictions=[],
                        forecast_date=forecast_date,
                        window_days=window_days,
                        model_version=forecaster.model_version,
                        insufficient_data=True,
                    )
                    insufficient += 1
                continue

            metrics = [
                ("cpl", forecaster.forecast_cpl),
                ("leads", forecaster.forecast_leads),
                ("spend", forecaster.forecast_spend),
            ]

            for metric_name, fn in metrics:
                exists = await ml_repo.forecast_exists(
                    config_id=config_id,
                    entity_type="campaign",
                    entity_id=campaign.campaign_id,
                    target_metric=metric_name,
                    horizon_days=horizon_days,
                    forecast_date=forecast_date,
                )
                if exists:
                    skipped += 1
                    continue

                try:
                    series = fn(
                        df,
                        "campaign",
                        campaign.campaign_id,
                        horizon_days=horizon_days,
                    )
                except Exception:
                    insufficient += 1
                    continue

                predictions = []
                for item in series.forecasts:
                    predictions.append({
                        "date": item.forecast_date.isoformat(),
                        "predicted_value": item.predicted_value,
                        "confidence_lower": item.confidence_lower,
                        "confidence_upper": item.confidence_upper,
                    })

                await ml_repo.create_forecast(
                    config_id=config_id,
                    entity_type="campaign",
                    entity_id=campaign.campaign_id,
                    target_metric=metric_name,
                    horizon_days=horizon_days,
                    method=series.method,
                    predictions=predictions,
                    forecast_date=forecast_date,
                    window_days=window_days,
                    model_version=forecaster.model_version,
                    insufficient_data=False,
                )
                generated += 1

        await session.commit()

        return {
            "generated": generated,
            "skipped": skipped,
            "insufficient": insufficient,
        }


async def _run_retraining_for_config(config_id: int) -> dict:
    from app.db.repositories.ml_repo import MLRepository
    from app.db.models.ml_models import ModelType, JobStatus
    from app.services.classification_service import ClassificationService

    async with async_session_maker() as session:
        ml_repo = MLRepository(session)
        job = await ml_repo.create_training_job(
            model_type=ModelType.CAMPAIGN_CLASSIFIER,
            config_id=config_id,
        )
        await ml_repo.update_job_status(
            job.id,
            JobStatus.RUNNING,
            progress=0.0,
        )
        await session.commit()

        try:
            service = ClassificationService(session)
            metrics = await service.train_classifier(config_id=config_id)
            await session.commit()

            if not metrics:
                await ml_repo.update_job_status(
                    job.id,
                    JobStatus.FAILED,
                    error_message="Dados insuficientes para treinamento",
                )
                await session.commit()
                return {
                    "job_id": job.id,
                    "status": "insufficient_data",
                }

            await ml_repo.update_job_status(
                job.id,
                JobStatus.COMPLETED,
                progress=1.0,
                model_id=metrics.get("model_id") if isinstance(metrics, dict) else None,
            )
            await session.commit()

            return {
                "job_id": job.id,
                "metrics": metrics,
            }
        except Exception as exc:
            await ml_repo.update_job_status(
                job.id,
                JobStatus.FAILED,
                error_message=str(exc),
            )
            await session.commit()
            raise

async def _run_classification_for_config(config_id: int) -> dict:
    from app.db.repositories.ml_repo import MLRepository
    from app.db.models.ml_models import ModelType, JobStatus
    from app.services.classification_service import ClassificationService

    async with async_session_maker() as session:
        ml_repo = MLRepository(session)
        job = await ml_repo.create_training_job(
            model_type=ModelType.CAMPAIGN_CLASSIFIER,
            config_id=config_id,
        )
        await ml_repo.update_job_status(
            job.id,
            JobStatus.RUNNING,
            progress=0.0,
        )
        await session.commit()

        try:
            start_time = datetime.utcnow()
            service = ClassificationService(session)
            classifications = await service.classify_campaigns(
                config_id=config_id,
                force_reclassify=False,
            )
            await session.commit()

            inserted = 0
            for c in classifications:
                classified_at = c.get("classified_at")
                if classified_at:
                    try:
                        ts = datetime.fromisoformat(classified_at)
                    except ValueError:
                        ts = None
                    if ts and ts >= start_time:
                        inserted += 1

            ignored = len(classifications) - inserted

            await ml_repo.update_job_status(
                job.id,
                JobStatus.COMPLETED,
                progress=1.0,
            )
            await session.commit()

            return {
                "job_id": job.id,
                "classified": len(classifications),
                "inserted": inserted,
                "ignored": ignored,
            }
        except Exception as exc:
            await ml_repo.update_job_status(
                job.id,
                JobStatus.FAILED,
                error_message=str(exc),
            )
            await session.commit()
            raise


async def _run_recommendations_for_config(config_id: int) -> dict:
    from app.db.repositories.ml_repo import MLRepository
    from app.services.recommendation_service import RecommendationService

    async with async_session_maker() as session:
        ml_repo = MLRepository(session)
        expired = await ml_repo.expire_recommendations(config_id)
        await session.commit()

        start_time = datetime.utcnow()
        service = RecommendationService(session)
        recommendations = await service.generate_recommendations(
            config_id=config_id,
            force_refresh=False,
        )
        await session.commit()

        inserted = 0
        for rec in recommendations:
            created_at = rec.get("created_at")
            if created_at:
                try:
                    ts = datetime.fromisoformat(created_at)
                except ValueError:
                    ts = None
                if ts and ts >= start_time:
                    inserted += 1

        return {
            "expired": expired,
            "generated": len(recommendations),
            "inserted": inserted,
        }


async def _run_anomaly_detection_for_config(config_id: int) -> dict:
    from app.services.anomaly_service import AnomalyService

    async with async_session_maker() as session:
        service = AnomalyService(session)
        result = await service.detect_anomalies(
            config_id=config_id,
            days_to_analyze=1,
            history_days=30,
        )
        return {
            "anomalies_detected": result.anomalies_detected,
            "campaigns_analyzed": result.campaigns_analyzed,
        }


async def _run_daily_pipeline_for_configs(configs: list[dict]) -> list[dict]:
    results = []
    for config in configs:
        config_id = config["id"]
        logger.info(
            "Pipeline para config",
            config_id=config_id,
            name=config.get("name"),
        )
        features_result = await _run_compute_features_for_config(config_id)
        classification_result = await _run_classification_for_config(config_id)
        recommendations_result = await _run_recommendations_for_config(config_id)
        forecasts_result = await _run_forecasts_for_config(config_id)

        results.append({
            "config_id": config_id,
            "features": features_result,
            "classifications": classification_result,
            "recommendations": recommendations_result,
            "forecasts": forecasts_result,
        })
    return results


@celery_app.task(name="app.tasks.scheduled_tasks.daily_model_retraining")
def daily_model_retraining():
    """
    Retreina todos os modelos com dados novos.
    Executado diariamente às 05:00.
    """
    from sqlalchemy.orm import sessionmaker
    from app.db.models.famachat_readonly import SistemaFacebookAdsConfig

    logger.info("Iniciando retreinamento diário de modelos")

    Session = sessionmaker(bind=sync_engine)
    session = Session()

    try:
        # Obter configs ativas
        configs = session.query(SistemaFacebookAdsConfig).filter(
            SistemaFacebookAdsConfig.is_active == True
        ).all()

        results = []
        for config in configs:
            logger.info(
                "Retreinando modelos para config",
                config_id=config.id,
                name=config.name
            )
            result = asyncio.run(_run_retraining_for_config(config.id))
            results.append({"config_id": config.id, **result})

        logger.info(
            "Retreinamento diário enfileirado",
            configs_count=len(configs)
        )

        return {
            "status": "completed",
            "configs_processed": len(configs),
            "results": results
        }

    finally:
        session.close()


@celery_app.task(name="app.tasks.scheduled_tasks.daily_classification")
def daily_classification():
    """
    Reclassifica todas as campanhas ativas.
    Executado diariamente às 06:00.
    """
    from sqlalchemy.orm import sessionmaker
    from app.db.models.famachat_readonly import SistemaFacebookAdsConfig

    logger.info("Iniciando classificação diária de campanhas")

    Session = sessionmaker(bind=sync_engine)
    session = Session()

    try:
        configs = session.query(SistemaFacebookAdsConfig).filter(
            SistemaFacebookAdsConfig.is_active == True
        ).all()

        results = []
        for config in configs:
            logger.info(
                "Classificando campanhas para config",
                config_id=config.id,
                name=config.name,
            )
            result = asyncio.run(_run_classification_for_config(config.id))
            results.append({"config_id": config.id, **result})

        logger.info(
            "Classificação diária concluída",
            configs_count=len(configs),
        )

        return {
            "status": "completed",
            "configs_processed": len(configs),
            "results": results,
        }

    finally:
        session.close()


@celery_app.task(name="app.tasks.scheduled_tasks.daily_recommendations")
def daily_recommendations():
    """
    Gera novas recomendações para campanhas.
    Executado diariamente às 07:00.
    """
    from sqlalchemy.orm import sessionmaker
    from app.db.models.famachat_readonly import SistemaFacebookAdsConfig

    logger.info("Iniciando geração diária de recomendações")

    Session = sessionmaker(bind=sync_engine)
    session = Session()

    try:
        configs = session.query(SistemaFacebookAdsConfig).filter(
            SistemaFacebookAdsConfig.is_active == True
        ).all()

        results = []
        for config in configs:
            logger.info(
                "Gerando recomendações para config",
                config_id=config.id
            )
            result = asyncio.run(_run_recommendations_for_config(config.id))
            results.append({"config_id": config.id, **result})

        logger.info(
            "Geração de recomendações concluída",
            configs_count=len(configs),
        )

        return {
            "status": "completed",
            "configs_processed": len(configs),
            "results": results,
        }

    finally:
        session.close()


@celery_app.task(name="app.tasks.scheduled_tasks.hourly_anomaly_detection")
def hourly_anomaly_detection():
    """
    Detecta anomalias em métricas recentes.
    Executado a cada hora no minuto 30.
    """
    from sqlalchemy.orm import sessionmaker
    from app.db.models.famachat_readonly import SistemaFacebookAdsConfig

    logger.info("Iniciando detecção horária de anomalias")

    Session = sessionmaker(bind=sync_engine)
    session = Session()

    try:
        configs = session.query(SistemaFacebookAdsConfig).filter(
            SistemaFacebookAdsConfig.is_active == True
        ).all()

        results = []
        for config in configs:
            logger.info(
                "Detectando anomalias para config",
                config_id=config.id
            )
            result = asyncio.run(_run_anomaly_detection_for_config(config.id))
            results.append({"config_id": config.id, **result})

        logger.info(
            "Detecção de anomalias concluída",
            configs_count=len(configs),
        )

        return {
            "status": "completed",
            "configs_processed": len(configs),
            "results": results,
        }

    finally:
        session.close()


@celery_app.task(name="app.tasks.scheduled_tasks.daily_pipeline")
def daily_pipeline():
    """
    Pipeline diario completo.
    Executado diariamente às 02:00.
    """
    from sqlalchemy.orm import sessionmaker
    from app.db.models.famachat_readonly import SistemaFacebookAdsConfig

    logger.info("Iniciando pipeline diario de ML")

    Session = sessionmaker(bind=sync_engine)
    session = Session()

    try:
        configs = session.query(SistemaFacebookAdsConfig).filter(
            SistemaFacebookAdsConfig.is_active == True
        ).all()

        config_refs = [
            {"id": config.id, "name": config.name}
            for config in configs
        ]
        results = asyncio.run(_run_daily_pipeline_for_configs(config_refs))

        logger.info(
            "Pipeline diario concluido",
            configs_count=len(configs),
        )

        return {
            "status": "completed",
            "configs_processed": len(configs),
            "results": results,
        }

    finally:
        session.close()


@celery_app.task(name="app.tasks.scheduled_tasks.batch_predictions")
def batch_predictions():
    """
    Gera previsões em batch para entidades ativas.
    Executado a cada 4 horas.
    """
    from sqlalchemy.orm import sessionmaker
    from app.db.models.famachat_readonly import (
        SistemaFacebookAdsConfig,
        SistemaFacebookAdsCampaigns,
    )

    logger.info("Iniciando previsões em batch")

    Session = sessionmaker(bind=sync_engine)
    session = Session()

    try:
        configs = session.query(SistemaFacebookAdsConfig).filter(
            SistemaFacebookAdsConfig.is_active == True
        ).all()

        predictions_generated = 0
        for config in configs:
            campaigns = session.query(SistemaFacebookAdsCampaigns).filter(
                SistemaFacebookAdsCampaigns.config_id == config.id,
                SistemaFacebookAdsCampaigns.status == "ACTIVE"
            ).all()

            # TODO: Implementar previsões na Fase 5
            for campaign in campaigns:
                logger.debug(
                    "Gerando previsão para campanha",
                    campaign_id=campaign.campaign_id
                )

        logger.info(
            "Previsões em batch concluídas",
            configs_count=len(configs),
            predictions_count=predictions_generated
        )

        return {
            "status": "completed",
            "configs_processed": len(configs),
            "predictions_generated": predictions_generated
        }

    finally:
        session.close()


@celery_app.task(name="app.tasks.scheduled_tasks.validate_predictions")
def validate_predictions():
    """
    Valida previsões anteriores com valores reais.
    Executado diariamente às 08:00.
    """
    from sqlalchemy.orm import sessionmaker
    from app.db.models.ml_models import MLPrediction

    logger.info("Iniciando validação de previsões")

    Session = sessionmaker(bind=sync_engine)
    session = Session()

    try:
        # Buscar previsões de ontem que ainda não foram validadas
        yesterday = datetime.utcnow().date() - timedelta(days=1)

        predictions = session.query(MLPrediction).filter(
            MLPrediction.forecast_date >= yesterday,
            MLPrediction.forecast_date < datetime.utcnow().date(),
            MLPrediction.actual_value.is_(None)
        ).all()

        validated_count = 0
        for prediction in predictions:
            # TODO: Buscar valor real e atualizar previsão
            logger.debug(
                "Validando previsão",
                prediction_id=prediction.id,
                entity_id=prediction.entity_id
            )

        logger.info(
            "Validação de previsões concluída",
            predictions_validated=validated_count
        )

        return {
            "status": "completed",
            "predictions_validated": validated_count
        }

    finally:
        session.close()
