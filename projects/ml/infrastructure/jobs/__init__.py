"""ML background jobs.

Re-exports from the original jobs/ location.
"""
from projects.ml.jobs.training_tasks import (
    train_model_task,
    retrain_all_models_task,
)
from projects.ml.jobs.scheduled_tasks import (
    generate_daily_predictions_task,
    generate_weekly_classifications_task,
)

__all__ = [
    "train_model_task",
    "retrain_all_models_task",
    "generate_daily_predictions_task",
    "generate_weekly_classifications_task",
]
