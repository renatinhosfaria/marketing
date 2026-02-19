"""
Tasks agendadas (Celery Beat).
Executam automaticamente conforme cronograma definido em celery_app.py.
"""

from datetime import datetime, timedelta
import asyncio
from dataclasses import asdict

from app.celery import celery_app
from shared.db.session import sync_engine, async_session_maker
from shared.core.logging import get_logger

logger = get_logger(__name__)

def _serialize_features(features) -> dict:
    data = asdict(features)
    for key, value in list(data.items()):
        if isinstance(value, datetime):
            data[key] = value.isoformat()
    return data


async def _run_compute_features_for_config(config_id: int, window_days: int = 30, session_maker=None) -> dict:
    from projects.ml.db.repositories.insights_repo import InsightsRepository
    from projects.ml.db.repositories.ml_repo import MLRepository
    from projects.ml.services.data_service import DataService

    if session_maker is None:
        session_maker = async_session_maker

    async with session_maker() as session:
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
                entity_id=campaign.campaign_id,
                entity_type="campaign",
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
            # O nome da campanha já está disponível no objeto campaign
            campaign_name = campaign.name if hasattr(campaign, 'name') else None

            if features is None:
                await ml_repo.create_feature(
                    config_id=config_id,
                    entity_id=campaign.campaign_id,
                    entity_type="campaign",
                    window_days=window_days,
                    feature_date=feature_date,
                    features=None,
                    insufficient_data=True,
                    campaign_name=campaign_name,
                )
                insufficient += 1
            else:
                await ml_repo.create_feature(
                    config_id=config_id,
                    entity_id=campaign.campaign_id,
                    entity_type="campaign",
                    window_days=window_days,
                    feature_date=feature_date,
                    features=_serialize_features(features),
                    insufficient_data=False,
                    campaign_name=campaign_name,
                )
                inserted += 1

        await session.commit()

        return {
            "processed": processed,
            "inserted": inserted,
            "skipped": skipped,
            "insufficient": insufficient,
        }


async def _run_forecasts_for_config(config_id: int, window_days: int = 90, session_maker=None) -> dict:
    from projects.ml.db.repositories.insights_repo import InsightsRepository
    from projects.ml.db.repositories.ml_repo import MLRepository
    from projects.ml.services.data_service import DataService
    from projects.ml.algorithms.models.timeseries.ensemble_forecaster import get_ensemble_forecaster
    from projects.ml.algorithms.models.timeseries.forecaster import get_forecaster
    from shared.config import settings

    if session_maker is None:
        session_maker = async_session_maker

    async with session_maker() as session:
        insights_repo = InsightsRepository(session)
        ml_repo = MLRepository(session)
        data_service = DataService(session)

        campaigns = await insights_repo.get_active_campaigns(config_id)
        forecast_date = datetime.utcnow().date()
        horizon_days = settings.ml_default_horizon_days

        generated = 0
        skipped = 0
        insufficient = 0

        # Criar EnsembleForecaster com fallback para forecaster simples
        try:
            ensemble = get_ensemble_forecaster(include_prophet=True)
        except Exception:
            ensemble = None
            logger.warning("EnsembleForecaster indisponível, usando forecaster simples")

        fallback = get_forecaster(method='auto')

        for campaign in campaigns:
            # O nome da campanha já está disponível no objeto campaign
            campaign_name = campaign.name if hasattr(campaign, 'name') else None

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
                        method="ensemble",
                        predictions=[],
                        forecast_date=forecast_date,
                        window_days=window_days,
                        model_version="1.0.0",
                        insufficient_data=True,
                        campaign_name=campaign_name,
                    )
                    insufficient += 1
                continue

            # Calibrar pesos do ensemble se houver dados suficientes
            if ensemble and len(df) > 21:
                try:
                    ensemble.calibrate_weights(df, 'spend', validation_days=min(14, len(df) // 3))
                except Exception:
                    pass  # Usa pesos iguais

            for metric_name in ("cpl", "leads", "spend"):
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

                series = None
                # Tentar ensemble primeiro, fallback para forecaster simples
                if ensemble:
                    try:
                        series = ensemble.forecast(
                            df, metric_name, "campaign",
                            campaign.campaign_id, horizon_days,
                        )
                    except Exception:
                        series = None

                if series is None:
                    try:
                        series = fallback.forecast(
                            df, metric_name, "campaign",
                            campaign.campaign_id, horizon_days,
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
                    model_version="1.0.0",
                    insufficient_data=False,
                    campaign_name=campaign_name,
                )
                generated += 1

        await session.commit()

        return {
            "generated": generated,
            "skipped": skipped,
            "insufficient": insufficient,
        }


async def _run_retraining_for_config(config_id: int, session_maker=None) -> dict:
    from projects.ml.db.repositories.ml_repo import MLRepository
    from projects.ml.db.models import ModelType, JobStatus
    from projects.ml.services.classification_service import ClassificationService

    if session_maker is None:
        session_maker = async_session_maker

    async with session_maker() as session:
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

async def _run_classification_for_config(config_id: int, session_maker=None) -> dict:
    from projects.ml.db.repositories.ml_repo import MLRepository
    from projects.ml.db.models import ModelType, JobStatus
    from projects.ml.services.classification_service import ClassificationService

    if session_maker is None:
        session_maker = async_session_maker

    async with session_maker() as session:
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


async def _run_recommendations_for_config(config_id: int, session_maker=None) -> dict:
    from projects.ml.db.repositories.ml_repo import MLRepository
    from projects.ml.services.recommendation_service import RecommendationService

    if session_maker is None:
        session_maker = async_session_maker

    async with session_maker() as session:
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


async def _run_anomaly_detection_for_config(config_id: int, session_maker=None) -> dict:
    from projects.ml.services.anomaly_service import AnomalyService

    if session_maker is None:
        session_maker = async_session_maker

    async with session_maker() as session:
        service = AnomalyService(session)
        result = await service.detect_anomalies(
            config_id=config_id,
            days_to_analyze=1,
            history_days=30,
        )
        return {
            "anomalies_detected": result.anomalies_detected,
            "entities_analyzed": result.entities_analyzed,
        }


async def _run_daily_pipeline_for_configs(configs: list[dict], session_maker=None) -> list[dict]:
    results = []
    for config in configs:
        config_id = config["id"]
        logger.info(
            "Pipeline para config",
            config_id=config_id,
            name=config.get("name"),
        )
        features_result = await _run_compute_features_for_config(config_id, session_maker=session_maker)
        classification_result = await _run_classification_for_config(config_id, session_maker=session_maker)
        recommendations_result = await _run_recommendations_for_config(config_id, session_maker=session_maker)
        forecasts_result = await _run_forecasts_for_config(config_id, session_maker=session_maker)

        results.append({
            "config_id": config_id,
            "features": features_result,
            "classifications": classification_result,
            "recommendations": recommendations_result,
            "forecasts": forecasts_result,
        })
    return results


@celery_app.task(name="projects.ml.jobs.scheduled_tasks.daily_model_retraining")
def daily_model_retraining():
    """
    Retreina todos os modelos com dados novos.
    Executado diariamente às 05:00.
    """
    from sqlalchemy.orm import sessionmaker
    from shared.db.models.famachat_readonly import SistemaFacebookAdsConfig
    from shared.db.session import create_isolated_async_session_maker

    logger.info("Iniciando retreinamento diário de modelos")

    Session = sessionmaker(bind=sync_engine)
    session = Session()

    try:
        # Obter configs ativas
        configs = session.query(SistemaFacebookAdsConfig).filter(
            SistemaFacebookAdsConfig.is_active == True
        ).all()

        # Criar session_maker isolado
        isolated_engine, isolated_session_maker = create_isolated_async_session_maker()

        try:
            results = []
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                for config in configs:
                    logger.info(
                        "Retreinando modelos para config",
                        config_id=config.id,
                        name=config.name
                    )
                    result = loop.run_until_complete(_run_retraining_for_config(config.id, session_maker=isolated_session_maker))
                    results.append({"config_id": config.id, **result})
                # Dispose do engine dentro do mesmo loop
                loop.run_until_complete(isolated_engine.dispose())
            finally:
                loop.close()
                asyncio.set_event_loop(None)
        except Exception as e:
            logger.error("Erro no retreinamento", error=str(e), exc_info=True)
            raise

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


@celery_app.task(name="projects.ml.jobs.scheduled_tasks.daily_classification")
def daily_classification():
    """
    Reclassifica todas as campanhas ativas.
    Executado diariamente às 06:00.
    """
    from sqlalchemy.orm import sessionmaker
    from shared.db.models.famachat_readonly import SistemaFacebookAdsConfig
    from shared.db.session import create_isolated_async_session_maker

    logger.info("Iniciando classificação diária de campanhas")

    Session = sessionmaker(bind=sync_engine)
    session = Session()

    try:
        configs = session.query(SistemaFacebookAdsConfig).filter(
            SistemaFacebookAdsConfig.is_active == True
        ).all()

        # Criar session_maker isolado
        isolated_engine, isolated_session_maker = create_isolated_async_session_maker()

        try:
            results = []
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                for config in configs:
                    logger.info(
                        "Classificando campanhas para config",
                        config_id=config.id,
                        name=config.name,
                    )
                    result = loop.run_until_complete(_run_classification_for_config(config.id, session_maker=isolated_session_maker))
                    results.append({"config_id": config.id, **result})
                # Dispose do engine dentro do mesmo loop
                loop.run_until_complete(isolated_engine.dispose())
            finally:
                loop.close()
                asyncio.set_event_loop(None)
        except Exception as e:
            logger.error("Erro na classificação", error=str(e), exc_info=True)
            raise

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


@celery_app.task(name="projects.ml.jobs.scheduled_tasks.daily_recommendations")
def daily_recommendations():
    """
    Gera novas recomendações para campanhas.
    Executado diariamente às 07:00.
    """
    from sqlalchemy.orm import sessionmaker
    from shared.db.models.famachat_readonly import SistemaFacebookAdsConfig
    from shared.db.session import create_isolated_async_session_maker

    logger.info("Iniciando geração diária de recomendações")

    Session = sessionmaker(bind=sync_engine)
    session = Session()

    try:
        configs = session.query(SistemaFacebookAdsConfig).filter(
            SistemaFacebookAdsConfig.is_active == True
        ).all()

        # Criar session_maker isolado
        isolated_engine, isolated_session_maker = create_isolated_async_session_maker()

        try:
            results = []
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                for config in configs:
                    logger.info(
                        "Gerando recomendações para config",
                        config_id=config.id
                    )
                    result = loop.run_until_complete(_run_recommendations_for_config(config.id, session_maker=isolated_session_maker))
                    results.append({"config_id": config.id, **result})
                # Dispose do engine dentro do mesmo loop
                loop.run_until_complete(isolated_engine.dispose())
            finally:
                loop.close()
                asyncio.set_event_loop(None)
        except Exception as e:
            logger.error("Erro na geração de recomendações", error=str(e), exc_info=True)
            raise

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


@celery_app.task(name="projects.ml.jobs.scheduled_tasks.hourly_anomaly_detection")
def hourly_anomaly_detection():
    """
    Detecta anomalias em métricas recentes.
    Executado a cada hora no minuto 30.
    """
    from sqlalchemy.orm import sessionmaker
    from shared.db.models.famachat_readonly import SistemaFacebookAdsConfig
    from shared.db.session import create_isolated_async_session_maker

    logger.info("Iniciando detecção horária de anomalias")

    Session = sessionmaker(bind=sync_engine)
    session = Session()

    try:
        configs = session.query(SistemaFacebookAdsConfig).filter(
            SistemaFacebookAdsConfig.is_active == True
        ).all()

        # Criar session_maker isolado
        isolated_engine, isolated_session_maker = create_isolated_async_session_maker()

        try:
            results = []
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                for config in configs:
                    logger.info(
                        "Detectando anomalias para config",
                        config_id=config.id
                    )
                    result = loop.run_until_complete(_run_anomaly_detection_for_config(config.id, session_maker=isolated_session_maker))
                    results.append({"config_id": config.id, **result})
                # Dispose do engine dentro do mesmo loop
                loop.run_until_complete(isolated_engine.dispose())
            finally:
                loop.close()
                asyncio.set_event_loop(None)
        except Exception as e:
            logger.error("Erro na detecção de anomalias", error=str(e), exc_info=True)
            raise

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


@celery_app.task(name="projects.ml.jobs.scheduled_tasks.daily_pipeline")
def daily_pipeline():
    """
    Pipeline diario completo.
    Executado diariamente às 02:00.
    """
    from sqlalchemy.orm import sessionmaker
    from shared.db.models.famachat_readonly import SistemaFacebookAdsConfig
    from shared.db.session import create_isolated_async_session_maker

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

        # Criar session_maker isolado para uso em código async
        isolated_engine, isolated_session_maker = create_isolated_async_session_maker()

        try:
            # Criar novo event loop limpo
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                results = loop.run_until_complete(
                    _run_daily_pipeline_for_configs(config_refs, session_maker=isolated_session_maker)
                )
                # Dispose do engine dentro do mesmo loop
                loop.run_until_complete(isolated_engine.dispose())
            finally:
                loop.close()
                asyncio.set_event_loop(None)
        except Exception as e:
            logger.error("Erro no pipeline", error=str(e), exc_info=True)
            raise

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


@celery_app.task(name="projects.ml.jobs.scheduled_tasks.batch_predictions")
def batch_predictions():
    """
    Gera previsões em batch para entidades ativas.
    Executado a cada 4 horas.
    """
    from sqlalchemy.orm import sessionmaker
    from shared.db.models.famachat_readonly import (
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


@celery_app.task(name="projects.ml.jobs.scheduled_tasks.validate_predictions")
def validate_predictions():
    """
    Valida previsões anteriores com valores reais.
    Busca previsões com forecast_date no passado e actual_value=NULL,
    consulta insights_history para obter o valor real e atualiza.
    Executado diariamente às 08:00.
    """
    from shared.db.session import create_isolated_async_session_maker

    logger.info("Iniciando validação de previsões")

    isolated_engine, isolated_session_maker = create_isolated_async_session_maker()

    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                _validate_predictions_async(isolated_session_maker)
            )
            loop.run_until_complete(isolated_engine.dispose())
        finally:
            loop.close()
            asyncio.set_event_loop(None)

        logger.info(
            "Validação de previsões concluída",
            validated=result["validated"],
            no_data=result["no_data"],
        )

        return {"status": "completed", **result}

    except Exception as e:
        logger.error("Erro na validação de previsões", error=str(e), exc_info=True)
        raise


async def _validate_predictions_async(session_maker) -> dict:
    """
    Busca previsões pendentes e compara com valores reais do insights_history.
    """
    from sqlalchemy import select, and_, func, cast, Date
    from projects.ml.db.models import MLPrediction, PredictionType
    from projects.ml.db.repositories.ml_repo import MLRepository
    from shared.db.models.famachat_readonly import SistemaFacebookAdsInsightsHistory

    # Mapeamento prediction_type -> coluna de insights
    METRIC_MAP = {
        PredictionType.CPL_FORECAST: None,  # CPL é derivado (spend/leads)
        PredictionType.LEADS_FORECAST: "leads",
        PredictionType.SPEND_FORECAST: "spend",
    }

    async with session_maker() as session:
        ml_repo = MLRepository(session)

        # Buscar previsões com data passada (até 30 dias atrás) sem validação
        cutoff = datetime.utcnow().date() - timedelta(days=30)
        today = datetime.utcnow().date()

        result = await session.execute(
            select(MLPrediction).where(
                and_(
                    MLPrediction.forecast_date >= cutoff,
                    MLPrediction.forecast_date < today,
                    MLPrediction.actual_value.is_(None),
                )
            )
        )
        predictions = result.scalars().all()

        validated = 0
        no_data = 0

        for pred in predictions:
            # Determinar coluna da métrica
            metric_col = METRIC_MAP.get(pred.prediction_type)

            # Determinar filtro de entidade
            entity_filter = None
            if pred.entity_type == "campaign":
                entity_filter = SistemaFacebookAdsInsightsHistory.campaign_id == pred.entity_id
            elif pred.entity_type == "adset":
                entity_filter = SistemaFacebookAdsInsightsHistory.adset_id == pred.entity_id
            elif pred.entity_type == "ad":
                entity_filter = SistemaFacebookAdsInsightsHistory.ad_id == pred.entity_id

            if entity_filter is None:
                continue

            # Buscar valor real para a data prevista
            pred_date = pred.forecast_date.date() if hasattr(pred.forecast_date, 'date') else pred.forecast_date

            if pred.prediction_type == PredictionType.CPL_FORECAST:
                # CPL = spend / leads
                q = select(
                    func.sum(SistemaFacebookAdsInsightsHistory.spend).label('spend'),
                    func.sum(SistemaFacebookAdsInsightsHistory.leads).label('leads'),
                ).where(
                    and_(
                        SistemaFacebookAdsInsightsHistory.config_id == pred.config_id,
                        entity_filter,
                        cast(SistemaFacebookAdsInsightsHistory.date, Date) == pred_date,
                    )
                )
            else:
                col = getattr(SistemaFacebookAdsInsightsHistory, metric_col)
                q = select(
                    func.sum(col).label('value'),
                ).where(
                    and_(
                        SistemaFacebookAdsInsightsHistory.config_id == pred.config_id,
                        entity_filter,
                        cast(SistemaFacebookAdsInsightsHistory.date, Date) == pred_date,
                    )
                )

            row = (await session.execute(q)).first()

            actual_value = None
            if pred.prediction_type == PredictionType.CPL_FORECAST:
                if row and row.spend and row.leads and row.leads > 0:
                    actual_value = float(row.spend) / float(row.leads)
            else:
                if row and row.value is not None:
                    actual_value = float(row.value)

            if actual_value is not None:
                await ml_repo.update_prediction_actual(pred.id, actual_value)
                validated += 1
            else:
                no_data += 1

        await session.commit()

        return {
            "validated": validated,
            "no_data": no_data,
            "total_checked": len(predictions),
        }


# ==================== MULTI-LEVEL HELPER FUNCTIONS ====================

async def _run_classification_for_entity_type(
    config_id: int,
    entity_type: str,
    session_maker=None
) -> dict:
    """
    Executa classificação para um tipo específico de entidade.

    Args:
        config_id: ID da configuração
        entity_type: Tipo de entidade (campaign, adset, ad)
        session_maker: Session maker para conexão async
    """
    from projects.ml.db.repositories.ml_repo import MLRepository
    from projects.ml.db.models import ModelType, JobStatus
    from projects.ml.services.classification_service import ClassificationService

    if session_maker is None:
        session_maker = async_session_maker

    async with session_maker() as session:
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
            classifications = await service.classify_entities(
                config_id=config_id,
                entity_type=entity_type,
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
                "entity_type": entity_type,
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


async def _run_recommendations_for_entity_type(
    config_id: int,
    entity_type: str,
    session_maker=None
) -> dict:
    """
    Executa geração de recomendações para um tipo específico de entidade.

    Args:
        config_id: ID da configuração
        entity_type: Tipo de entidade (campaign, adset, ad)
        session_maker: Session maker para conexão async
    """
    from projects.ml.db.repositories.ml_repo import MLRepository
    from projects.ml.services.recommendation_service import RecommendationService

    if session_maker is None:
        session_maker = async_session_maker

    async with session_maker() as session:
        ml_repo = MLRepository(session)
        expired = await ml_repo.expire_recommendations(config_id, entity_type=entity_type)
        await session.commit()

        start_time = datetime.utcnow()
        service = RecommendationService(session)
        recommendations = await service.generate_entity_recommendations(
            config_id=config_id,
            entity_type=entity_type,
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
            "entity_type": entity_type,
            "expired": expired,
            "generated": len(recommendations),
            "inserted": inserted,
        }


async def _run_anomaly_detection_for_entity_type(
    config_id: int,
    entity_type: str,
    session_maker=None
) -> dict:
    """
    Executa detecção de anomalias para um tipo específico de entidade.

    Args:
        config_id: ID da configuração
        entity_type: Tipo de entidade (campaign, adset, ad)
        session_maker: Session maker para conexão async
    """
    from projects.ml.services.anomaly_service import AnomalyService

    if session_maker is None:
        session_maker = async_session_maker

    async with session_maker() as session:
        service = AnomalyService(session)
        result = await service.detect_anomalies(
            config_id=config_id,
            entity_type=entity_type,
            days_to_analyze=1,
            history_days=30,
        )
        return {
            "entity_type": entity_type,
            "anomalies_detected": result.anomalies_detected,
            "entities_analyzed": result.entities_analyzed,
        }


# ==================== ADSET-LEVEL TASKS ====================

@celery_app.task(name="projects.ml.jobs.scheduled_tasks.daily_adset_classification")
def daily_adset_classification():
    """
    Classifica todos os adsets ativos.
    Executado diariamente às 06:30.
    """
    from sqlalchemy.orm import sessionmaker
    from shared.db.models.famachat_readonly import SistemaFacebookAdsConfig
    from shared.db.session import create_isolated_async_session_maker

    logger.info("Iniciando classificação diária de adsets")

    Session = sessionmaker(bind=sync_engine)
    session = Session()

    try:
        configs = session.query(SistemaFacebookAdsConfig).filter(
            SistemaFacebookAdsConfig.is_active == True
        ).all()

        isolated_engine, isolated_session_maker = create_isolated_async_session_maker()

        try:
            results = []
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                for config in configs:
                    logger.info(
                        "Classificando adsets para config",
                        config_id=config.id,
                        name=config.name,
                    )
                    result = loop.run_until_complete(
                        _run_classification_for_entity_type(
                            config.id, "adset", session_maker=isolated_session_maker
                        )
                    )
                    results.append({"config_id": config.id, **result})
                loop.run_until_complete(isolated_engine.dispose())
            finally:
                loop.close()
                asyncio.set_event_loop(None)
        except Exception as e:
            logger.error("Erro na classificação de adsets", error=str(e), exc_info=True)
            raise

        logger.info(
            "Classificação de adsets concluída",
            configs_count=len(configs),
        )

        return {
            "status": "completed",
            "entity_type": "adset",
            "configs_processed": len(configs),
            "results": results,
        }

    finally:
        session.close()


@celery_app.task(name="projects.ml.jobs.scheduled_tasks.daily_adset_recommendations")
def daily_adset_recommendations():
    """
    Gera recomendações para adsets.
    Executado diariamente às 07:30.
    """
    from sqlalchemy.orm import sessionmaker
    from shared.db.models.famachat_readonly import SistemaFacebookAdsConfig
    from shared.db.session import create_isolated_async_session_maker

    logger.info("Iniciando geração de recomendações para adsets")

    Session = sessionmaker(bind=sync_engine)
    session = Session()

    try:
        configs = session.query(SistemaFacebookAdsConfig).filter(
            SistemaFacebookAdsConfig.is_active == True
        ).all()

        isolated_engine, isolated_session_maker = create_isolated_async_session_maker()

        try:
            results = []
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                for config in configs:
                    logger.info(
                        "Gerando recomendações de adsets para config",
                        config_id=config.id
                    )
                    result = loop.run_until_complete(
                        _run_recommendations_for_entity_type(
                            config.id, "adset", session_maker=isolated_session_maker
                        )
                    )
                    results.append({"config_id": config.id, **result})
                loop.run_until_complete(isolated_engine.dispose())
            finally:
                loop.close()
                asyncio.set_event_loop(None)
        except Exception as e:
            logger.error("Erro na geração de recomendações de adsets", error=str(e), exc_info=True)
            raise

        logger.info(
            "Geração de recomendações de adsets concluída",
            configs_count=len(configs),
        )

        return {
            "status": "completed",
            "entity_type": "adset",
            "configs_processed": len(configs),
            "results": results,
        }

    finally:
        session.close()


# ==================== AD-LEVEL TASKS ====================

@celery_app.task(name="projects.ml.jobs.scheduled_tasks.daily_ad_classification")
def daily_ad_classification():
    """
    Classifica todos os ads ativos.
    Executado diariamente às 07:00.
    """
    from sqlalchemy.orm import sessionmaker
    from shared.db.models.famachat_readonly import SistemaFacebookAdsConfig
    from shared.db.session import create_isolated_async_session_maker

    logger.info("Iniciando classificação diária de ads")

    Session = sessionmaker(bind=sync_engine)
    session = Session()

    try:
        configs = session.query(SistemaFacebookAdsConfig).filter(
            SistemaFacebookAdsConfig.is_active == True
        ).all()

        isolated_engine, isolated_session_maker = create_isolated_async_session_maker()

        try:
            results = []
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                for config in configs:
                    logger.info(
                        "Classificando ads para config",
                        config_id=config.id,
                        name=config.name,
                    )
                    result = loop.run_until_complete(
                        _run_classification_for_entity_type(
                            config.id, "ad", session_maker=isolated_session_maker
                        )
                    )
                    results.append({"config_id": config.id, **result})
                loop.run_until_complete(isolated_engine.dispose())
            finally:
                loop.close()
                asyncio.set_event_loop(None)
        except Exception as e:
            logger.error("Erro na classificação de ads", error=str(e), exc_info=True)
            raise

        logger.info(
            "Classificação de ads concluída",
            configs_count=len(configs),
        )

        return {
            "status": "completed",
            "entity_type": "ad",
            "configs_processed": len(configs),
            "results": results,
        }

    finally:
        session.close()


@celery_app.task(name="projects.ml.jobs.scheduled_tasks.daily_ad_recommendations")
def daily_ad_recommendations():
    """
    Gera recomendações para ads.
    Executado diariamente às 08:00.
    """
    from sqlalchemy.orm import sessionmaker
    from shared.db.models.famachat_readonly import SistemaFacebookAdsConfig
    from shared.db.session import create_isolated_async_session_maker

    logger.info("Iniciando geração de recomendações para ads")

    Session = sessionmaker(bind=sync_engine)
    session = Session()

    try:
        configs = session.query(SistemaFacebookAdsConfig).filter(
            SistemaFacebookAdsConfig.is_active == True
        ).all()

        isolated_engine, isolated_session_maker = create_isolated_async_session_maker()

        try:
            results = []
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                for config in configs:
                    logger.info(
                        "Gerando recomendações de ads para config",
                        config_id=config.id
                    )
                    result = loop.run_until_complete(
                        _run_recommendations_for_entity_type(
                            config.id, "ad", session_maker=isolated_session_maker
                        )
                    )
                    results.append({"config_id": config.id, **result})
                loop.run_until_complete(isolated_engine.dispose())
            finally:
                loop.close()
                asyncio.set_event_loop(None)
        except Exception as e:
            logger.error("Erro na geração de recomendações de ads", error=str(e), exc_info=True)
            raise

        logger.info(
            "Geração de recomendações de ads concluída",
            configs_count=len(configs),
        )

        return {
            "status": "completed",
            "entity_type": "ad",
            "configs_processed": len(configs),
            "results": results,
        }

    finally:
        session.close()


# ==================== MULTI-LEVEL ANOMALY DETECTION ====================

@celery_app.task(name="projects.ml.jobs.scheduled_tasks.hourly_adset_anomaly_detection")
def hourly_adset_anomaly_detection():
    """
    Detecta anomalias em adsets.
    Executado a cada hora no minuto 35.
    """
    from sqlalchemy.orm import sessionmaker
    from shared.db.models.famachat_readonly import SistemaFacebookAdsConfig
    from shared.db.session import create_isolated_async_session_maker

    logger.info("Iniciando detecção de anomalias em adsets")

    Session = sessionmaker(bind=sync_engine)
    session = Session()

    try:
        configs = session.query(SistemaFacebookAdsConfig).filter(
            SistemaFacebookAdsConfig.is_active == True
        ).all()

        isolated_engine, isolated_session_maker = create_isolated_async_session_maker()

        try:
            results = []
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                for config in configs:
                    logger.info(
                        "Detectando anomalias em adsets para config",
                        config_id=config.id
                    )
                    result = loop.run_until_complete(
                        _run_anomaly_detection_for_entity_type(
                            config.id, "adset", session_maker=isolated_session_maker
                        )
                    )
                    results.append({"config_id": config.id, **result})
                loop.run_until_complete(isolated_engine.dispose())
            finally:
                loop.close()
                asyncio.set_event_loop(None)
        except Exception as e:
            logger.error("Erro na detecção de anomalias em adsets", error=str(e), exc_info=True)
            raise

        logger.info(
            "Detecção de anomalias em adsets concluída",
            configs_count=len(configs),
        )

        return {
            "status": "completed",
            "entity_type": "adset",
            "configs_processed": len(configs),
            "results": results,
        }

    finally:
        session.close()


@celery_app.task(name="projects.ml.jobs.scheduled_tasks.hourly_ad_anomaly_detection")
def hourly_ad_anomaly_detection():
    """
    Detecta anomalias em ads.
    Executado a cada hora no minuto 40.
    """
    from sqlalchemy.orm import sessionmaker
    from shared.db.models.famachat_readonly import SistemaFacebookAdsConfig
    from shared.db.session import create_isolated_async_session_maker

    logger.info("Iniciando detecção de anomalias em ads")

    Session = sessionmaker(bind=sync_engine)
    session = Session()

    try:
        configs = session.query(SistemaFacebookAdsConfig).filter(
            SistemaFacebookAdsConfig.is_active == True
        ).all()

        isolated_engine, isolated_session_maker = create_isolated_async_session_maker()

        try:
            results = []
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                for config in configs:
                    logger.info(
                        "Detectando anomalias em ads para config",
                        config_id=config.id
                    )
                    result = loop.run_until_complete(
                        _run_anomaly_detection_for_entity_type(
                            config.id, "ad", session_maker=isolated_session_maker
                        )
                    )
                    results.append({"config_id": config.id, **result})
                loop.run_until_complete(isolated_engine.dispose())
            finally:
                loop.close()
                asyncio.set_event_loop(None)
        except Exception as e:
            logger.error("Erro na detecção de anomalias em ads", error=str(e), exc_info=True)
            raise

        logger.info(
            "Detecção de anomalias em ads concluída",
            configs_count=len(configs),
        )

        return {
            "status": "completed",
            "entity_type": "ad",
            "configs_processed": len(configs),
            "results": results,
        }

    finally:
        session.close()
