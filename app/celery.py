"""
Configuração do Celery para tasks assíncronas e agendadas.
Entry point limpo que referencia tasks nos projetos isolados.
"""

from celery import Celery
from celery.schedules import crontab

from shared.config import settings

# Criar aplicação Celery
celery_app = Celery(
    "marketing",
    broker=settings.celery_broker,
    backend=settings.celery_backend,
    include=[
        "projects.ml.jobs.training_tasks",
        "projects.ml.jobs.scheduled_tasks",
        "projects.facebook_ads.jobs.sync_job",
        "projects.facebook_ads.jobs.insights_consolidation",
        "projects.facebook_ads.jobs.token_refresh",
    ]
)

# Configurações do Celery
celery_app.conf.update(
    # Serialização
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",

    # Timezone
    timezone=settings.timezone,
    enable_utc=True,

    # Concorrência e recursos
    worker_concurrency=settings.celery_worker_concurrency,
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=100,

    # Resultados
    result_expires=3600,  # 1 hora
    task_track_started=True,

    # Retry
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    broker_connection_retry_on_startup=True,

    # Auto-retry configuration for failed tasks
    task_autoretry_for=(Exception,),
    task_retry_backoff=True,
    task_retry_backoff_max=600,  # Max 10 minutes between retries
    task_retry_jitter=True,
    task_max_retries=3,

    # Rate limiting
    task_default_rate_limit="10/m",

    # Beat schedule - Jobs agendados
    beat_schedule={
        # Pipeline diario completo às 02:00
        "daily-ml-pipeline": {
            "task": "projects.ml.jobs.scheduled_tasks.daily_pipeline",
            "schedule": crontab(hour=2, minute=0),
            "options": {"queue": "ml"},
        },

        # Retreinar modelos diariamente às 05:00
        "daily-model-retraining": {
            "task": "projects.ml.jobs.scheduled_tasks.daily_model_retraining",
            "schedule": crontab(hour=5, minute=0),
            "options": {"queue": "training"},
        },

        # ==================== CAMPAIGN LEVEL ====================
        # Classificar campanhas diariamente às 06:00
        "daily-campaign-classification": {
            "task": "projects.ml.jobs.scheduled_tasks.daily_classification",
            "schedule": crontab(hour=6, minute=0),
            "options": {"queue": "ml"},
        },

        # Gerar recomendações para campanhas às 07:00
        "daily-campaign-recommendations": {
            "task": "projects.ml.jobs.scheduled_tasks.daily_recommendations",
            "schedule": crontab(hour=7, minute=0),
            "options": {"queue": "ml"},
        },

        # Detectar anomalias em campanhas a cada hora (minuto 30)
        "hourly-campaign-anomaly-detection": {
            "task": "projects.ml.jobs.scheduled_tasks.hourly_anomaly_detection",
            "schedule": crontab(minute=30),
            "options": {"queue": "ml"},
        },

        # Treinar XGBoost classifiers diariamente às 03:00
        "daily-classifier-training": {
            "task": "projects.ml.jobs.training_tasks.train_classifiers_all",
            "schedule": crontab(hour=3, minute=0),
            "options": {"queue": "training"},
        },

        # Treinar Isolation Forest diariamente às 04:00
        "daily-anomaly-detector-training": {
            "task": "projects.ml.jobs.training_tasks.train_anomaly_detectors_all",
            "schedule": crontab(hour=4, minute=0),
            "options": {"queue": "training"},
        },

        # ==================== ADSET LEVEL ====================
        # Classificar adsets diariamente às 06:30
        "daily-adset-classification": {
            "task": "projects.ml.jobs.scheduled_tasks.daily_adset_classification",
            "schedule": crontab(hour=6, minute=30),
            "options": {"queue": "ml"},
        },

        # Gerar recomendações para adsets às 07:30
        "daily-adset-recommendations": {
            "task": "projects.ml.jobs.scheduled_tasks.daily_adset_recommendations",
            "schedule": crontab(hour=7, minute=30),
            "options": {"queue": "ml"},
        },

        # Detectar anomalias em adsets a cada hora (minuto 35)
        "hourly-adset-anomaly-detection": {
            "task": "projects.ml.jobs.scheduled_tasks.hourly_adset_anomaly_detection",
            "schedule": crontab(minute=35),
            "options": {"queue": "ml"},
        },

        # ==================== AD LEVEL ====================
        # Classificar ads diariamente às 07:00 (before recommendations)
        "daily-ad-classification": {
            "task": "projects.ml.jobs.scheduled_tasks.daily_ad_classification",
            "schedule": crontab(hour=7, minute=15),
            "options": {"queue": "ml"},
        },

        # Gerar recomendações para ads às 08:30
        "daily-ad-recommendations": {
            "task": "projects.ml.jobs.scheduled_tasks.daily_ad_recommendations",
            "schedule": crontab(hour=8, minute=30),
            "options": {"queue": "ml"},
        },

        # Detectar anomalias em ads a cada hora (minuto 40)
        "hourly-ad-anomaly-detection": {
            "task": "projects.ml.jobs.scheduled_tasks.hourly_ad_anomaly_detection",
            "schedule": crontab(minute=40),
            "options": {"queue": "ml"},
        },

        # Previsões em batch a cada 4 horas
        "batch-predictions": {
            "task": "projects.ml.jobs.scheduled_tasks.batch_predictions",
            "schedule": crontab(hour="*/4", minute=15),
            "options": {"queue": "ml"},
        },

        # Validar previsões anteriores diariamente às 08:00
        "daily-prediction-validation": {
            "task": "projects.ml.jobs.scheduled_tasks.validate_predictions",
            "schedule": crontab(hour=8, minute=0),
            "options": {"queue": "ml"},
        },

        # Facebook Ads - Sync incremental a cada hora
        "facebook-ads-sync-incremental": {
            "task": "projects.facebook_ads.jobs.sync_job.facebook_ads_sync_incremental",
            "schedule": crontab(minute=0),
            "options": {"queue": "default"},
        },

        # Facebook Ads - Sync completo diário às 02:30
        "facebook-ads-sync-full": {
            "task": "projects.facebook_ads.jobs.sync_job.facebook_ads_sync_full",
            "schedule": crontab(hour=2, minute=30),
            "options": {"queue": "default"},
        },

        # Facebook Ads - Consolidação de insights às 00:05
        "facebook-ads-consolidation": {
            "task": "projects.facebook_ads.jobs.insights_consolidation.facebook_ads_consolidation",
            "schedule": crontab(hour=0, minute=5),
            "options": {"queue": "default"},
        },

        # Facebook Ads - Renovação de tokens às 06:30
        "facebook-ads-token-refresh": {
            "task": "projects.facebook_ads.jobs.token_refresh.facebook_ads_token_refresh",
            "schedule": crontab(hour=6, minute=30),
            "options": {"queue": "default"},
        },
    },

    # Filas
    task_queues={
        "default": {},
        "training": {"exchange": "training", "routing_key": "training"},
        "ml": {"exchange": "ml", "routing_key": "ml"},
    },

    task_default_queue="default",

    # Routing
    task_routes={
        "projects.ml.jobs.training_tasks.*": {"queue": "training"},
        "projects.ml.jobs.scheduled_tasks.daily_model_retraining": {"queue": "training"},
        "projects.ml.jobs.scheduled_tasks.*": {"queue": "ml"},
        "projects.facebook_ads.jobs.*": {"queue": "default"},
    },
)
