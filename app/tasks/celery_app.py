"""
Configuração do Celery para tasks assíncronas e agendadas.
"""

from celery import Celery
from celery.schedules import crontab

from app.config import settings

# Criar aplicação Celery
celery_app = Celery(
    "famachat-ml",
    broker=settings.celery_broker,
    backend=settings.celery_backend,
    include=[
        "app.tasks.training_tasks",
        "app.tasks.scheduled_tasks",
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
    worker_concurrency=2,
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=100,

    # Resultados
    result_expires=3600,  # 1 hora
    task_track_started=True,

    # Retry
    task_acks_late=True,
    task_reject_on_worker_lost=True,

    # Rate limiting
    task_default_rate_limit="10/m",

    # Beat schedule - Jobs agendados
    beat_schedule={
        # Pipeline diario completo às 02:00
        "daily-ml-pipeline": {
            "task": "app.tasks.scheduled_tasks.daily_pipeline",
            "schedule": crontab(hour=2, minute=0),
            "options": {"queue": "ml"},
        },

        # Retreinar modelos diariamente às 05:00
        "daily-model-retraining": {
            "task": "app.tasks.scheduled_tasks.daily_model_retraining",
            "schedule": crontab(hour=5, minute=0),
            "options": {"queue": "training"},
        },

        # Classificar campanhas diariamente às 06:00
        "daily-classification": {
            "task": "app.tasks.scheduled_tasks.daily_classification",
            "schedule": crontab(hour=6, minute=0),
            "options": {"queue": "ml"},
        },

        # Gerar recomendações diariamente às 07:00
        "daily-recommendations": {
            "task": "app.tasks.scheduled_tasks.daily_recommendations",
            "schedule": crontab(hour=7, minute=0),
            "options": {"queue": "ml"},
        },

        # Detectar anomalias a cada hora (minuto 30)
        "hourly-anomaly-detection": {
            "task": "app.tasks.scheduled_tasks.hourly_anomaly_detection",
            "schedule": crontab(minute=30),
            "options": {"queue": "ml"},
        },

        # Previsões em batch a cada 4 horas
        "batch-predictions": {
            "task": "app.tasks.scheduled_tasks.batch_predictions",
            "schedule": crontab(hour="*/4", minute=15),
            "options": {"queue": "ml"},
        },

        # Validar previsões anteriores diariamente às 08:00
        "daily-prediction-validation": {
            "task": "app.tasks.scheduled_tasks.validate_predictions",
            "schedule": crontab(hour=8, minute=0),
            "options": {"queue": "ml"},
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
        "app.tasks.training_tasks.*": {"queue": "training"},
        "app.tasks.scheduled_tasks.daily_model_retraining": {"queue": "training"},
        "app.tasks.scheduled_tasks.*": {"queue": "ml"},
    },
)
