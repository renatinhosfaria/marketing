# Machine Learning

## Arquitetura Multi-Nivel

O sistema de ML suporta analise em tres niveis hierarquicos:

```
Campaign (Campanha)
    └── Adset (Conjunto de Anuncios)
            └── Ad (Anuncio)
```

Cada nivel pode ser analisado independentemente para:
- **Classificacoes**: Tier de performance (HIGH_PERFORMER, MODERATE, LOW, UNDERPERFORMER)
- **Anomalias**: Deteccao de comportamentos atipicos
- **Recomendacoes**: Sugestoes de otimizacao (genericas + especificas por nivel)
- **Forecasts**: Previsoes de metricas

### Contexto Hierarquico

Entidades nos niveis inferiores (adset, ad) incluem contexto do nivel superior:
- `share_of_parent_spend`: Percentual do spend do parent que esta entidade representa
- `performance_vs_siblings`: Performance relativa comparada a outras entidades do mesmo parent

## Arquitetura de Dados

### Fontes de Dados (Input)

Os algoritmos de ML consomem dados das seguintes tabelas do Facebook Ads:

| Tabela | Descricao | Uso no ML |
|--------|-----------|-----------|
| `sistema_facebook_ads_config` | Configuracao da conta de anuncios | Identificacao do config_id e parametros da conta |
| `sistema_facebook_ads_campaigns` | Campanhas e seus status | Classificacao e recomendacoes por campanha |
| `sistema_facebook_ads_adsets` | Conjuntos de anuncios | Analise granular por adset |
| `sistema_facebook_ads_ads` | Anuncios individuais | Analise granular por anuncio |
| `sistema_facebook_ads_insights_history` | Metricas historicas (spend, leads, clicks, impressions, CTR, CPL, frequency, reach) | Previsoes, classificacoes, anomalias e recomendacoes |
| `sistema_facebook_ads_insights_today` | Metricas do dia atual | Deteccao de anomalias em tempo real |
| `sistema_facebook_ads_insights_breakdowns` | Metricas segmentadas (idade, genero, regiao, dispositivo) | Analise de segmentacao e recomendacoes de audiencia |
| `sistema_facebook_ads_sync_history` | Historico de sincronizacoes | Validacao de dados e consistencia |

### Tabelas de Resultados (Output)

Os resultados dos algoritmos de ML sao armazenados nas seguintes tabelas:

| Tabela | Descricao | Dados Armazenados |
|--------|-----------|-------------------|
| `ml_predictions` | Previsoes individuais | CPL, Leads, Spend previstos com intervalos de confianca |
| `ml_forecasts` | Series de forecasts | Previsoes para multiplos dias (horizonte 7-30 dias) |
| `ml_classifications` | Classificacoes de entidades | Tier, confianca, metricas. Suporta campaign, adset, ad via entity_type |
| `ml_recommendations` | Recomendacoes de otimizacao | Tipo (SCALE_UP, PAUSE_CAMPAIGN, etc.), prioridade, acao sugerida |
| `ml_anomalies` | Anomalias detectadas | Tipo, severidade, valor observado vs esperado, z-score |
| `ml_features` | Features pre-calculadas | Features engineered para entidades (campaign, adset, ad) |
| `ml_trained_models` | Modelos treinados | XGBoost, Prophet serializados com metricas |
| `ml_training_jobs` | Jobs de treinamento | Status, progresso, erros |

### Fluxo de Dados

```
┌─────────────────────────────────────────────────────────────┐
│                    FONTE (Facebook Ads)                     │
│  sistema_facebook_ads_config                                │
│  sistema_facebook_ads_campaigns                             │
│  sistema_facebook_ads_adsets                                │
│  sistema_facebook_ads_ads                                   │
│  sistema_facebook_ads_insights_history                      │
│  sistema_facebook_ads_insights_today                        │
│  sistema_facebook_ads_insights_breakdowns                   │
│  sistema_facebook_ads_sync_history                          │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│               PROCESSAMENTO (Algoritmos ML)                 │
│  - AnomalyDetector (Z-Score, IQR, Change-point)             │
│  - CampaignClassifier (XGBoost)                             │
│  - TimeSeriesForecaster (Prophet, EMA, Linear)              │
│  - RuleEngine (7 regras de otimizacao)                      │
│  - FeatureEngineer (30+ features calculadas)                │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                  DESTINO (Resultados ML)                    │
│  ml_anomalies                                               │
│  ml_campaign_classifications                                │
│  ml_forecasts / ml_predictions                              │
│  ml_recommendations                                         │
│  ml_features                                                │
│  ml_trained_models                                          │
└─────────────────────────────────────────────────────────────┘
```

---

## Previsoes

Endpoints em `projects/ml/api/predictions.py` e `projects/ml/api/forecasts.py` geram previsoes de CPL, Leads e Spend com horizonte configuravel (1-30 dias).

**Metodos disponiveis:**
- Prophet (Facebook) - sazonalidade semanal
- EMA (Exponential Moving Average) - fallback
- Linear Regression - tendencia simples

## Classificacoes

Entidades (campaign, adset, ad) sao classificadas por tier em `projects/ml/api/classifications.py`:

| Tier | Criterio |
|------|----------|
| HIGH_PERFORMER | CPL <= 70% da media, leads consistentes |
| MODERATE | CPL 70-100% da media |
| LOW | CPL 100-150% da media |
| UNDERPERFORMER | CPL > 150% ou sem leads |

**Modelo:** XGBoost com 10+ features engineered.

**API:**
```
POST /api/v1/ml/classifications/classify
{
  "config_id": 1,
  "entity_type": "campaign|adset|ad",
  "entity_ids": ["123", "456"]  // opcional
}
```

## Recomendacoes

Recomendacoes de otimizacao sao geradas em `projects/ml/api/recommendations.py`.

### Tipos Genericos (Todos os Niveis)

| Tipo | Trigger |
|------|---------|
| SCALE_UP | CPL <= 70%, leads consistentes |
| BUDGET_DECREASE | CPL >= 150%, tendencia de alta |
| PAUSE_CAMPAIGN | Sem leads ou CPL > 250% por 7+ dias |
| CREATIVE_REFRESH | Frequencia > 3.0 com CTR/CPL em queda |
| AUDIENCE_REVIEW | CPL subindo > 20% com leads caindo |
| BUDGET_INCREASE | CPL 70-100%, performance estavel |
| REACTIVATE | Entidade pausada com bom historico |

### Tipos Especificos por Nivel

**Adset:**
| Tipo | Trigger |
|------|---------|
| AUDIENCE_EXPANSION | Adset HIGH_PERFORMER com share_of_parent < 30%, CPL < 80% media |
| AUDIENCE_NARROWING | Adset UNDERPERFORMER por 5+ dias, CPL > 150% |

**Ad:**
| Tipo | Trigger |
|------|---------|
| CREATIVE_TEST | Ad HIGH_PERFORMER recente (< 7 dias), share_of_parent < 20% |
| CREATIVE_WINNER | Ad com melhor performance_vs_siblings, CPL < 80% media |

**API:**
```
POST /api/v1/ml/recommendations/generate
{
  "config_id": 1,
  "entity_type": "campaign|adset|ad",
  "entity_ids": ["123", "456"]  // opcional
}
```

## Anomalias

Deteccao de comportamentos atipicos em `projects/ml/api/anomalies.py`. Suporta todos os niveis (campaign, adset, ad).

| Tipo | Severidade |
|------|------------|
| SPEND_SPIKE / SPEND_DROP | LOW a CRITICAL |
| CPL_SPIKE / CPL_DROP | MEDIUM a CRITICAL |
| CTR_DROP / CTR_SPIKE | LOW a HIGH |
| FREQUENCY_ALERT | LOW a CRITICAL (> 3.0) |
| ZERO_SPEND / ZERO_IMPRESSIONS | HIGH |

**Metodos:** Z-Score (threshold 2.5σ), IQR, Change-point detection.

**API:**
```
POST /api/v1/ml/anomalies/detect
{
  "config_id": 1,
  "entity_type": "campaign|adset|ad",
  "days": 1,
  "entity_ids": ["123", "456"]  // opcional
}
```

## Modelos

Gestao de modelos em `projects/ml/api/models.py`.

## Jobs

Agendamentos definidos em `app/celery.py`. Jobs sao separados por nivel para controle granular e escalonamento de recursos.

### Pipeline e Treinamento

| Job | Frequencia | Descricao |
|-----|------------|-----------|
| `daily_ml_pipeline` | Diario 02:00 | Pipeline completo de ML |
| `daily_model_retraining` | Diario 05:00 | Re-treinamento dos modelos |
| `batch_predictions` | A cada 4h (minuto 15) | Forecasts de CPL, Leads, Spend |
| `daily_prediction_validation` | Diario 08:00 | Validacao de previsoes anteriores |

### Campaign Level

| Job | Frequencia | Descricao |
|-----|------------|-----------|
| `daily_campaign_classification` | Diario 06:00 | Classificacao de campanhas |
| `daily_campaign_recommendations` | Diario 07:00 | Recomendacoes para campanhas |
| `hourly_campaign_anomaly_detection` | A cada hora (minuto 30) | Anomalias em campanhas |

### Adset Level

| Job | Frequencia | Descricao |
|-----|------------|-----------|
| `daily_adset_classification` | Diario 06:30 | Classificacao de adsets |
| `daily_adset_recommendations` | Diario 07:30 | Recomendacoes para adsets |
| `hourly_adset_anomaly_detection` | A cada hora (minuto 35) | Anomalias em adsets |

### Ad Level

| Job | Frequencia | Descricao |
|-----|------------|-----------|
| `daily_ad_classification` | Diario 07:15 | Classificacao de ads |
| `daily_ad_recommendations` | Diario 08:30 | Recomendacoes para ads |
| `hourly_ad_anomaly_detection` | A cada hora (minuto 40) | Anomalias em ads |

### Facebook Ads Sync

| Job | Frequencia | Descricao |
|-----|------------|-----------|
| `facebook_ads_sync_incremental` | A cada hora (minuto 0) | Sync incremental |
| `facebook_ads_sync_full` | Diario 02:30 | Sync completo |
| `facebook_ads_consolidation` | Diario 00:05 | Consolidacao de insights |
| `facebook_ads_token_refresh` | Diario 06:30 | Renovacao de tokens |

## Parametros

Valores globais e thresholds em `shared/infrastructure/config/settings.py`:

| Parametro | Valor | Descricao |
|-----------|-------|-----------|
| `threshold_cpl_low` | 0.7 | CPL excelente (70% da media) |
| `threshold_cpl_high` | 1.3 | CPL alto (130% da media) |
| `threshold_ctr_good` | 1.2 | CTR bom (120% da media) |
| `threshold_frequency_high` | 3.0 | Frequencia alta (fadiga) |
| `threshold_days_underperforming` | 7 | Dias para considerar pausar |
