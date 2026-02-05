# Design: Treinamento Automático do Isolation Forest

**Data:** 2026-02-04
**Status:** Aprovado

## Resumo

Implementar treinamento automático diário do Isolation Forest para detecção de anomalias multivariadas, com modelos treinados por entidade (campaign, adset, ad) e persistidos em filesystem.

## Decisões de Design

| Decisão | Escolha | Justificativa |
|---------|---------|---------------|
| Escopo de treinamento | Por entidade | Captura padrões específicos de cada campaign/adset/ad |
| Frequência | Diário (04:00) | Balanceia atualização vs custo computacional |
| Armazenamento | Filesystem (joblib) | Padrão sklearn, persistente, já configurado |

## Arquitetura

### Fluxo de Treinamento (Diário às 04:00)

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Celery Beat    │────▶│  train_anomaly   │────▶│  Para cada      │
│  (04:00 daily)  │     │  _detectors_all  │     │  config ativa   │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                          │
                        ┌─────────────────────────────────▼─────────────────────────────────┐
                        │                    Para cada entidade:                            │
                        │  ┌──────────────┐  ┌───────────────┐  ┌────────────────────────┐  │
                        │  │ Buscar dados │─▶│ Treinar IF    │─▶│ Salvar em filesystem   │  │
                        │  │ históricos   │  │ (50+ samples) │  │ /models_storage/...    │  │
                        │  └──────────────┘  └───────────────┘  └────────────────────────┘  │
                        └───────────────────────────────────────────────────────────────────┘
```

### Fluxo de Detecção (Horário)

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  hourly_anomaly │────▶│  AnomalyDetector │────▶│  Carregar modelo│
│  _detection     │     │  .detect()       │     │  do filesystem  │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                          │
                                                          ▼
                                                 ┌─────────────────┐
                                                 │  Detecção       │
                                                 │  multivariada   │
                                                 └─────────────────┘
```

## Estrutura de Armazenamento

```
/app/models_storage/
  anomaly_detector/
    config_{id}/
      campaign_{id}.joblib
      adset_{id}.joblib
      ad_{id}.joblib
      metadata.json
```

**metadata.json:**
```json
{
  "config_id": 123,
  "last_training": "2026-02-04T04:00:00Z",
  "models_count": {
    "campaign": 15,
    "adset": 42,
    "ad": 128
  },
  "training_duration_seconds": 45
}
```

## Parâmetros do Modelo

| Parâmetro | Valor | Descrição |
|-----------|-------|-----------|
| n_estimators | 100 | Número de árvores |
| contamination | 0.1 | 10% anomalias esperadas |
| random_state | 42 | Reprodutibilidade |
| min_samples | 50 | Mínimo para treinar |
| history_days | 90 | Dias de histórico |
| features | spend, cpl, ctr, frequency, leads | Métricas usadas |

## Arquivos a Modificar

1. **shared/infrastructure/config/settings.py**
   - Adicionar `use_isolation_forest: bool = True`
   - Adicionar `isolation_forest_min_samples: int = 50`
   - Adicionar `isolation_forest_contamination: float = 0.1`
   - Adicionar `isolation_forest_history_days: int = 90`

2. **projects/ml/algorithms/models/anomaly/anomaly_detector.py**
   - Habilitar `use_isolation_forest=True` por padrão
   - Adicionar método `load_model(entity_type, entity_id, config_id)`
   - Adicionar método `save_model(entity_type, entity_id, config_id)`
   - Modificar `detect_anomalies()` para aceitar `config_id`
   - Adicionar cache de modelos em memória

3. **projects/ml/jobs/training_tasks.py**
   - Implementar `train_anomaly_detectors_all()`
   - Implementar `train_anomaly_detector_for_config(config_id)`

4. **app/celery.py**
   - Adicionar schedule `daily-anomaly-detector-training` às 04:00

5. **projects/ml/jobs/scheduled_tasks.py**
   - Atualizar chamadas para passar `config_id` ao detector

## Schedule do Pipeline Diário

```
02:00 - daily_pipeline
02:30 - facebook_ads_sync_full
04:00 - train_anomaly_detectors_all  ← NOVO
05:00 - daily_model_retraining
06:00 - daily_classification (campaigns)
06:30 - daily_adset_classification
07:00 - daily_ad_classification
07:15 - daily_recommendations (campaigns)
07:30 - daily_adset_recommendations
08:00 - daily_ad_recommendations
```

## Considerações

- Modelos são pequenos (~10-50KB), storage não é concern
- Se entidade não tem modelo, fallback para detecção estatística
- Cache em memória evita I/O repetido durante detecção horária
- Cleanup automático de modelos de entidades inativas (futuro)
