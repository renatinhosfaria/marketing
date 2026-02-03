# Machine Learning

## Previsoes

Endpoints em `projects/ml/api/predictions.py` geram previsoes de CPL e leads com horizonte configuravel.

## Classificacoes

Campanhas sao classificadas por tier (HIGH_PERFORMER, MODERATE, LOW, UNDERPERFORMER) em `projects/ml/api/classifications.py`.

## Recomendacoes

Recomendacoes de otimizacao sao geradas em `projects/ml/api/recommendations.py`.

## Anomalias

Deteccao de comportamentos atipicos em `projects/ml/api/anomalies.py`.

## Modelos

Gestao de modelos em `projects/ml/api/models.py`.

## Jobs

Agendamentos definidos em `app/celery.py` (daily pipeline, retraining, anomaly detection, batch predictions).

## Parametros

Valores globais e thresholds em `shared/infrastructure/config/settings.py`.
