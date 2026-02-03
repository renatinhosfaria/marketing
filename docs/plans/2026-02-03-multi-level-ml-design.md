# Multi-Level ML Design

Data: 2026-02-03

## Objetivo

Expandir os servicos de Machine Learning para suportar analise em tres niveis: Campaign, Adset e Ad.

## Decisoes de Design

1. **Algoritmos**: Todos (Anomalias, Classificacoes, Recomendacoes, Forecasts) expandidos para os tres niveis
2. **Classificacoes**: Tabela generica `ml_classifications` com coluna `entity_type`
3. **Recomendacoes**: Tipos genericos + especificos (CREATIVE_TEST, CREATIVE_WINNER para ads)
4. **Jobs**: Separados por nivel com horarios diferentes

## Arquitetura

```
┌─────────────────────────────────────────────────────────────────┐
│                         ENTITY LEVELS                           │
├─────────────────────────────────────────────────────────────────┤
│  Campaign (18)  ──►  Adset (91)  ──►  Ad (278)                  │
│       │                  │               │                       │
│       ▼                  ▼               ▼                       │
│  ┌─────────┐        ┌─────────┐    ┌─────────┐                  │
│  │Features │        │Features │    │Features │                  │
│  │Anomalies│        │Anomalies│    │Anomalies│                  │
│  │Classif. │        │Classif. │    │Classif. │                  │
│  │Recomm.  │        │Recomm.  │    │Recomm.  │                  │
│  │Forecasts│        │Forecasts│    │Forecasts│                  │
│  └─────────┘        └─────────┘    └─────────┘                  │
└─────────────────────────────────────────────────────────────────┘
```

## Mudancas no Banco de Dados

### Tabela ml_classifications (renomeada de ml_campaign_classifications)

```sql
ALTER TABLE ml_campaign_classifications RENAME TO ml_classifications;

ALTER TABLE ml_classifications
ADD COLUMN entity_type VARCHAR(20) NOT NULL DEFAULT 'campaign';

CREATE INDEX idx_ml_classifications_entity_type
ON ml_classifications(entity_type);

CREATE INDEX idx_ml_classifications_config_entity
ON ml_classifications(config_id, entity_type, entity_id);
```

### Tabela ml_features (adicionar entity_type)

```sql
ALTER TABLE ml_features
ADD COLUMN entity_type VARCHAR(20) NOT NULL DEFAULT 'campaign';
```

## Feature Engineering

### EntityFeatures (estrutura generica)

```python
@dataclass
class EntityFeatures:
    entity_type: str  # "campaign", "adset", "ad"
    entity_id: str
    config_id: int
    parent_id: Optional[str]  # campaign_id para adset, adset_id para ad

    # Metricas basicas (7d)
    spend_7d: float
    impressions_7d: int
    clicks_7d: int
    leads_7d: int

    # Metricas calculadas
    cpl_7d: float
    ctr_7d: float
    cpc_7d: float
    conversion_rate_7d: float

    # Tendencias
    cpl_trend: float
    leads_trend: float
    spend_trend: float
    ctr_trend: float

    # Volatilidade e consistencia
    cpl_std_7d: float
    days_with_leads_7d: int
    frequency_7d: float

    # Contexto hierarquico
    share_of_parent_spend: float
    share_of_parent_leads: float
    performance_vs_siblings: float
```

### Features Especificas por Nivel

| Feature | Campaign | Adset | Ad |
|---------|----------|-------|-----|
| share_of_parent_spend | - | % da campanha | % do adset |
| audience_overlap_score | - | sim | - |
| creative_fatigue_score | - | - | sim |
| performance_vs_siblings | - | sim | sim |

## Tipos de Recomendacao

```python
class RecommendationType(str, Enum):
    # Genericos (todos os niveis)
    SCALE_UP = "SCALE_UP"
    BUDGET_INCREASE = "BUDGET_INCREASE"
    BUDGET_DECREASE = "BUDGET_DECREASE"
    PAUSE = "PAUSE"
    REACTIVATE = "REACTIVATE"

    # Especificos para Adset
    AUDIENCE_REVIEW = "AUDIENCE_REVIEW"
    AUDIENCE_EXPANSION = "AUDIENCE_EXPANSION"
    AUDIENCE_NARROWING = "AUDIENCE_NARROWING"

    # Especificos para Ad
    CREATIVE_REFRESH = "CREATIVE_REFRESH"
    CREATIVE_TEST = "CREATIVE_TEST"
    CREATIVE_WINNER = "CREATIVE_WINNER"
```

### Regras por Nivel

| Regra | Campaign | Adset | Ad |
|-------|----------|-------|-----|
| CPL baixo → SCALE_UP | sim | sim | sim |
| CPL alto → BUDGET_DECREASE | sim | sim | - |
| Sem leads → PAUSE | sim | sim | sim |
| Frequencia alta → CREATIVE_REFRESH | - | - | sim |
| CPL subindo + leads caindo → AUDIENCE_REVIEW | - | sim | - |
| Melhor que irmaos → CREATIVE_WINNER | - | - | sim |
| Pior que irmaos → PAUSE | - | sim | sim |

## Jobs Agendados

| Job | Horario | Nivel | Descricao |
|-----|---------|-------|-----------|
| campaign_classification | 05:00 | campaign | Classificacao de campanhas |
| adset_classification | 05:30 | adset | Classificacao de adsets |
| ad_classification | 06:00 | ad | Classificacao de ads |
| campaign_anomaly_detection | */60 min | campaign | Anomalias em campanhas |
| adset_anomaly_detection | */60 min | adset | Anomalias em adsets |
| ad_anomaly_detection | */120 min | ad | Anomalias em ads |
| campaign_recommendations | 05:00 | campaign | Recomendacoes para campanhas |
| adset_recommendations | 05:30 | adset | Recomendacoes para adsets |
| ad_recommendations | 06:00 | ad | Recomendacoes para ads |
| campaign_forecasts | 04:00 | campaign | Forecasts de campanhas |
| adset_forecasts | 04:30 | adset | Forecasts de adsets |

## API Endpoints

```
# Classificacoes
GET  /api/v1/ml/classifications?entity_type=campaign|adset|ad
POST /api/v1/ml/classifications/classify
     body: { config_id, entity_type, entity_ids? }

# Anomalias
GET  /api/v1/ml/anomalies?entity_type=campaign|adset|ad
POST /api/v1/ml/anomalies/detect
     body: { config_id, entity_type, entity_ids?, days }

# Recomendacoes
GET  /api/v1/ml/recommendations?entity_type=campaign|adset|ad
POST /api/v1/ml/recommendations/generate
     body: { config_id, entity_type?, force_refresh }

# Forecasts
GET  /api/v1/ml/forecasts?entity_type=campaign|adset|ad
POST /api/v1/ml/forecasts/generate
     body: { config_id, entity_type, entity_ids?, horizon_days }
```

## Arquivos a Modificar

| Arquivo | Mudanca |
|---------|---------|
| projects/ml/db/models.py | Renomear MLCampaignClassification → MLClassification |
| projects/ml/services/feature_engineering.py | Criar EntityFeatures generico |
| projects/ml/services/data_service.py | Adicionar get_adsets(), get_ads() |
| projects/ml/services/anomaly_service.py | Aceitar entity_type |
| projects/ml/services/classification_service.py | Aceitar entity_type |
| projects/ml/services/recommendation_service.py | Novos tipos de recomendacao |
| projects/ml/algorithms/models/recommendation/rule_engine.py | Regras por nivel |
| projects/ml/db/repositories/insights_repo.py | get_active_adsets(), get_active_ads() |
| projects/ml/db/repositories/ml_repo.py | Queries para ml_classifications |
| projects/ml/api/classifications.py | Parametro entity_type |
| projects/ml/jobs/scheduled_tasks.py | Jobs separados por nivel |

## Ordem de Implementacao

### Fase 1: Database & Models
1. Migracao Alembic
2. Atualizar models.py
3. Atualizar ml_repo.py

### Fase 2: Feature Engineering
4. EntityFeatures generico
5. data_service.py metodos novos

### Fase 3: Services
6. anomaly_service.py
7. classification_service.py
8. recommendation_service.py + rule_engine.py

### Fase 4: API & Jobs
9. Endpoints com entity_type
10. Jobs separados por nivel

## Compatibilidade

- Manter `CampaignFeatures` como alias para `EntityFeatures` com entity_type="campaign"
- APIs existentes continuam funcionando (entity_type default = "campaign")
- Migracao preserva dados existentes
