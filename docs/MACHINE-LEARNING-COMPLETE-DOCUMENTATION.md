# FamaChat ML - Documentação Completa do Sistema de Machine Learning

**Versão:** 1.0.0
**Data:** Janeiro 2026
**Autor:** Equipe FamaChat
**Status:** Produção

---

## Sumário

1. [Visão Geral](#1-visão-geral)
2. [Arquitetura do Sistema](#2-arquitetura-do-sistema)
3. [Stack Tecnológica](#3-stack-tecnológica)
4. [Modelos de Machine Learning](#4-modelos-de-machine-learning)
5. [Engenharia de Features](#5-engenharia-de-features)
6. [Agente IA Conversacional](#6-agente-ia-conversacional)
7. [API REST](#7-api-rest)
8. [Banco de Dados](#8-banco-de-dados)
9. [Jobs Agendados (Celery)](#9-jobs-agendados-celery)
10. [Integração com FamaChat](#10-integração-com-famachat)
11. [Deploy e Infraestrutura](#11-deploy-e-infraestrutura)
12. [Monitoramento e Observabilidade](#12-monitoramento-e-observabilidade)
13. [Segurança](#13-segurança)
14. [Guia de Uso](#14-guia-de-uso)
15. [Troubleshooting](#15-troubleshooting)
16. [Referências Técnicas](#16-referências-técnicas)

---

## 1. Visão Geral

### 1.1 O que é o FamaChat ML?

O **FamaChat ML** é um microserviço especializado em Machine Learning para otimização de campanhas de Facebook Ads no ecossistema FamaChat. O sistema oferece:

- **Classificação de Campanhas**: Categorização automática por performance (HIGH_PERFORMER, MODERATE, LOW, UNDERPERFORMER)
- **Detecção de Anomalias**: Identificação de comportamentos atípicos em métricas
- **Previsão de Métricas**: Forecast de CPL, leads e gastos usando séries temporais
- **Recomendações Inteligentes**: Sugestões de otimização baseadas em regras e ML
- **Agente IA Conversacional**: Assistente para análise de tráfego pago com LangGraph

### 1.2 Objetivos

| Objetivo | Descrição |
|----------|-----------|
| **Automatização** | Reduzir trabalho manual na análise de campanhas |
| **Proatividade** | Detectar problemas antes que afetem resultados |
| **Inteligência** | Fornecer insights acionáveis baseados em dados |
| **Escalabilidade** | Processar múltiplas contas simultaneamente |

### 1.3 Métricas Alvo

- **CPL (Cost Per Lead)**: Custo por lead gerado
- **CTR (Click-Through Rate)**: Taxa de cliques
- **Leads**: Volume de leads gerados
- **Spend**: Gastos com publicidade
- **Frequency**: Frequência de exibição
- **Reach**: Alcance das campanhas

---

## 2. Arquitetura do Sistema

### 2.1 Diagrama de Arquitetura

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          FamaChat Principal                              │
│                      (Express.js + React)                               │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 │ HTTP/REST
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         FamaChat ML API                                  │
│                      (FastAPI + Uvicorn)                                │
│                                                                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │  Predictions │  │Classifications│  │Recommendations│  │  Anomalies  │    │
│  │   Endpoint   │  │   Endpoint   │  │   Endpoint   │  │   Endpoint  │    │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘    │
│         │                │                │                │            │
│         ▼                ▼                ▼                ▼            │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                      Services Layer                              │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐           │   │
│  │  │  Data    │ │  Feature │ │  Class.  │ │  Recom.  │           │   │
│  │  │ Service  │ │ Engineer │ │ Service  │ │ Service  │           │   │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘           │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                 │                                       │
│                                 ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                       ML Models Layer                            │   │
│  │  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐   │   │
│  │  │  XGBoost   │ │  Anomaly   │ │   Time     │ │   Rule     │   │   │
│  │  │ Classifier │ │  Detector  │ │  Series    │ │  Engine    │   │   │
│  │  └────────────┘ └────────────┘ └────────────┘ └────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
          ┌──────────────────────┼──────────────────────┐
          │                      │                      │
          ▼                      ▼                      ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   PostgreSQL    │    │     Redis       │    │  Celery Worker  │
│   (Shared DB)   │    │  (Cache/Broker) │    │  + Beat         │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### 2.2 Estrutura de Diretórios

```
/var/www/famachat-ml/
├── app/                              # Código principal
│   ├── agent/                        # Agente IA (LangGraph)
│   │   ├── graph/                    # Grafo do agente
│   │   │   ├── builder.py            # Construtor do grafo
│   │   │   ├── nodes.py              # Nós de processamento
│   │   │   ├── edges.py              # Transições condicionais
│   │   │   └── state.py              # Estado do agente
│   │   ├── llm/                      # Provider de LLM
│   │   ├── memory/                   # Persistência (PostgreSQL)
│   │   ├── prompts/                  # System prompts
│   │   ├── tools/                    # 21 ferramentas do agente
│   │   ├── config.py                 # Configurações
│   │   └── service.py                # Serviço principal
│   │
│   ├── api/v1/                       # API REST
│   │   ├── endpoints/                # Endpoints por funcionalidade
│   │   │   ├── health.py             # Health checks
│   │   │   ├── predictions.py        # Previsões
│   │   │   ├── forecasts.py          # Séries temporais
│   │   │   ├── classifications.py    # Classificações
│   │   │   ├── recommendations.py    # Recomendações
│   │   │   ├── anomalies.py          # Anomalias
│   │   │   └── models.py             # Gestão de modelos
│   │   ├── agent/                    # Endpoints do agente
│   │   └── router.py                 # Router principal
│   │
│   ├── ml/                           # Modelos de ML
│   │   └── models/
│   │       ├── classification/       # XGBoost Classifier
│   │       ├── anomaly/              # Isolation Forest + Z-Score
│   │       ├── timeseries/           # Prophet/EMA/Linear
│   │       └── recommendation/       # Rule Engine
│   │
│   ├── services/                     # Lógica de negócio
│   │   ├── anomaly_service.py        # Gestão de anomalias
│   │   ├── classification_service.py # Classificação
│   │   ├── data_service.py           # Acesso a dados
│   │   ├── feature_engineering.py    # Extração de features
│   │   ├── recommendation_service.py # Recomendações
│   │   └── rule_engine.py            # Motor de regras
│   │
│   ├── db/                           # Database
│   │   ├── models/                   # SQLAlchemy models
│   │   ├── repositories/             # Data access layer
│   │   └── session.py                # Engine e sessão
│   │
│   ├── tasks/                        # Celery tasks
│   │   ├── celery_app.py             # Configuração Celery
│   │   ├── scheduled_tasks.py        # Tasks agendadas
│   │   └── training_tasks.py         # Tasks de treinamento
│   │
│   ├── core/                         # Utilitários
│   │   ├── security.py               # Autenticação
│   │   ├── logging.py                # Logging estruturado
│   │   └── exceptions.py             # Exceções customizadas
│   │
│   ├── config.py                     # Configurações gerais
│   └── main.py                       # Entry point FastAPI
│
├── alembic/                          # Migrações de banco
│   └── versions/                     # 7 migrações
│
├── models_storage/                   # Modelos treinados (.pkl)
├── logs/                             # Logs da aplicação
├── scripts/                          # Scripts utilitários
├── tests/                            # Testes (pytest)
├── docs/                             # Documentação
│
├── docker-compose.yml                # Orquestração Docker
├── Dockerfile                        # Build da imagem
├── requirements.txt                  # Dependências Python
└── README.md                         # Documentação principal
```

### 2.3 Fluxo de Dados

```
1. Dados Facebook Ads (FamaChat Principal)
         │
         ▼
2. Feature Engineering (Extração de 25+ features)
         │
         ▼
3. Modelos ML (Classificação, Anomalias, Previsões)
         │
         ▼
4. Motor de Regras (Geração de recomendações)
         │
         ▼
5. Persistência (PostgreSQL)
         │
         ▼
6. API REST / Agente IA (Consumo)
```

---

## 3. Stack Tecnológica

### 3.1 Componentes Principais

| Componente | Tecnologia | Versão | Função |
|------------|------------|--------|--------|
| **Framework Web** | FastAPI | 0.109.0 | API REST assíncrona |
| **Runtime** | Uvicorn | 0.27.0 | ASGI server |
| **Database** | PostgreSQL | 18.1 | Armazenamento principal |
| **ORM** | SQLAlchemy | 2.0.25 | Mapeamento objeto-relacional |
| **Async Driver** | asyncpg | 0.29.0 | Driver PostgreSQL assíncrono |
| **Cache/Broker** | Redis | 7.x | Cache e message broker |
| **Task Queue** | Celery | 5.3.6 | Processamento assíncrono |

### 3.2 Machine Learning

| Biblioteca | Versão | Função |
|------------|--------|--------|
| **scikit-learn** | 1.4.0 | Algoritmos ML base |
| **XGBoost** | 2.0.3 | Gradient Boosting para classificação |
| **LightGBM** | 4.3.0 | Gradient Boosting alternativo |
| **pandas** | 2.2.0 | Manipulação de dados |
| **numpy** | 1.26.3 | Computação numérica |
| **scipy** | 1.12.0 | Estatística e otimização |
| **statsmodels** | 0.14.1 | Modelos estatísticos |
| **Prophet** | 1.1.5 | Forecast de séries temporais (opcional) |

### 3.3 Agente IA

| Biblioteca | Versão | Função |
|------------|--------|--------|
| **LangGraph** | ≥0.2.0 | Framework de agentes |
| **langchain-core** | ≥0.3.0 | Componentes LangChain |
| **langchain-anthropic** | ≥0.2.0 | Provider Claude |
| **langchain-openai** | ≥0.2.0 | Provider OpenAI |
| **langgraph-checkpoint-postgres** | ≥1.0.0 | Persistência de estado |
| **tiktoken** | ≥0.5.0 | Tokenização |

### 3.4 Validação e Segurança

| Biblioteca | Versão | Função |
|------------|--------|--------|
| **Pydantic** | ≥2.7.4 | Validação de dados |
| **pydantic-settings** | ≥2.1.0 | Configuração |
| **python-jose** | 3.3.0 | JWT handling |
| **PyJWT** | ≥2.8.0 | JWT tokens |

### 3.5 Monitoramento

| Biblioteca | Versão | Função |
|------------|--------|--------|
| **structlog** | 24.1.0 | Logging estruturado |
| **flower** | 2.0.1 | Monitoramento Celery |

---

## 4. Modelos de Machine Learning

### 4.1 Campaign Classifier (XGBoost)

**Arquivo:** `app/ml/models/classification/campaign_classifier.py`

#### Descrição
Classificador de campanhas que categoriza performance em 4 tiers usando XGBoost com fallback para regras heurísticas.

#### Tiers de Classificação

| Tier | Critérios | Ação Sugerida |
|------|-----------|---------------|
| **HIGH_PERFORMER** | CPL ≤ 70% da média, ≥4 dias com leads | Escalar investimento |
| **MODERATE** | CPL ≤ 100% da média, ≥2 leads | Manter e otimizar |
| **LOW** | CPL ≤ 150% da média | Revisar estratégia |
| **UNDERPERFORMER** | CPL > 150% ou 0 leads com gasto | Pausar ou reestruturar |

#### Features de Entrada (10 colunas)

```python
FEATURE_COLUMNS = [
    'cpl_ratio',            # CPL / média (normalizado)
    'ctr_ratio',            # CTR / média
    'leads_7d_normalized',  # Leads por R$100 gastos
    'cpl_trend',            # Tendência do CPL (%)
    'leads_trend',          # Tendência de leads (%)
    'cpl_volatility',       # Desvio padrão do CPL
    'conversion_rate_7d',   # Taxa de conversão
    'days_with_leads_ratio',# % de dias com leads
    'frequency_score',      # Score de frequência (0-1)
    'consistency_score',    # Score de consistência
]
```

#### Hiperparâmetros XGBoost

```python
XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='multi:softprob',
    num_class=4,
    eval_metric='mlogloss',
)
```

#### Métricas de Avaliação

- **Accuracy**: Acurácia geral
- **F1-Macro**: F1 balanceado entre classes
- **F1-Weighted**: F1 ponderado por frequência
- **CV Mean**: Média da validação cruzada (5-fold)

#### Uso

```python
from app.ml.models.classification.campaign_classifier import CampaignClassifier

# Inicializar
classifier = CampaignClassifier()

# Classificar (usa regras se modelo não treinado)
result = classifier.classify(campaign_features, avg_cpl=50.0, avg_ctr=1.0)

# Resultado
print(f"Tier: {result.tier}")           # HIGH_PERFORMER
print(f"Confiança: {result.confidence_score}")  # 0.85
print(f"Probabilidades: {result.probabilities}")
```

---

### 4.2 Anomaly Detector

**Arquivo:** `app/ml/models/anomaly/anomaly_detector.py`

#### Descrição
Detector de anomalias usando múltiplos métodos estatísticos: Z-Score, IQR (Interquartile Range), e detecção de change-points.

#### Tipos de Anomalias Detectáveis

| Tipo | Métrica | Descrição |
|------|---------|-----------|
| `SPEND_SPIKE` | Spend | Aumento abrupto de gastos |
| `SPEND_DROP` | Spend | Queda abrupta de gastos |
| `CPL_SPIKE` | CPL | CPL acima do esperado |
| `CPL_DROP` | CPL | CPL abaixo do esperado |
| `CTR_SPIKE` | CTR | CTR anormalmente alto |
| `CTR_DROP` | CTR | CTR anormalmente baixo |
| `PERFORMANCE_DROP` | Geral | Queda de performance |
| `PERFORMANCE_SPIKE` | Geral | Pico de performance |
| `FREQUENCY_ALERT` | Frequency | Frequência alta (fadiga) |
| `REACH_SATURATION` | Reach | Saturação de audiência |
| `ZERO_SPEND` | Spend | Campanha sem gastos |
| `ZERO_IMPRESSIONS` | Impressions | Sem impressões |

#### Níveis de Severidade

| Severidade | Z-Score | Mudança % | Cor |
|------------|---------|-----------|-----|
| **LOW** | < 3.0 | < 30% | Amarelo |
| **MEDIUM** | 3.0 - 4.0 | 30-50% | Laranja |
| **HIGH** | 4.0 - 5.0 | 50-80% | Vermelho |
| **CRITICAL** | > 5.0 | > 80% | Vermelho escuro |

#### Métodos de Detecção

**1. Z-Score:**
```python
z_score = (valor_atual - média_histórica) / desvio_padrão
is_anomaly = abs(z_score) > 2.5  # Threshold padrão
```

**2. IQR (Interquartile Range):**
```python
Q1, Q3 = history.quantile([0.25, 0.75])
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
is_anomaly = value < lower_bound or value > upper_bound
```

**3. Change-Point Detection:**
```python
# Compara últimos 3 dias vs 7 anteriores
recent_mean = df.tail(3)['metric'].mean()
previous_mean = df.iloc[-10:-3]['metric'].mean()
pct_change = (recent_mean - previous_mean) / previous_mean * 100
is_anomaly = abs(pct_change) > 30%
```

#### Configuração

```python
AnomalyDetector(
    z_threshold=2.5,        # Threshold do Z-score
    iqr_multiplier=1.5,     # Multiplicador IQR
    min_history_days=7,     # Mínimo de histórico
)
```

---

### 4.3 Time Series Forecaster

**Arquivo:** `app/ml/models/timeseries/forecaster.py`

#### Descrição
Forecaster de séries temporais para prever CPL, leads e gastos usando múltiplos métodos.

#### Métodos Disponíveis

| Método | Descrição | Quando Usar |
|--------|-----------|-------------|
| **Prophet** | Modelo Meta (Facebook) | Sazonalidade forte |
| **EMA** | Média Móvel Exponencial | Dados estáveis |
| **Linear** | Regressão Linear | Tendência clara |

#### Seleção Automática de Método

```python
if method == 'auto':
    method = 'prophet' if PROPHET_AVAILABLE else 'ema'
```

#### Algoritmo EMA (Padrão)

```python
# EMAs com diferentes spans
ema_short = Series.ewm(span=3).mean()   # Curto prazo
ema_medium = Series.ewm(span=7).mean()  # Médio prazo
ema_long = Series.ewm(span=14).mean()   # Longo prazo

# Valor base ponderado
base_value = ema_short * 0.5 + ema_medium * 0.3 + ema_long * 0.2

# Tendência
trend = (ema_short - ema_long) / 14

# Previsão com atenuação
predicted = base_value + (trend * day * 0.5)

# Intervalo de confiança (95%)
uncertainty = std * z_score * sqrt(day)
lower = predicted - uncertainty
upper = predicted + uncertainty
```

#### Estrutura de Resultado

```python
@dataclass
class ForecastResult:
    entity_type: str        # 'campaign', 'adset', 'ad'
    entity_id: str
    metric: str             # 'cpl', 'leads', 'spend'
    forecast_date: datetime
    predicted_value: float
    confidence_lower: float
    confidence_upper: float
    confidence_level: float  # 0.95 (95%)
    method: str             # 'ema', 'prophet', 'linear'
    model_version: str
```

---

### 4.4 Recommendation Engine (Rule-Based)

**Arquivo:** `app/services/recommendation_service.py` + `app/ml/models/recommendation/rule_engine.py`

#### Descrição
Motor de regras que gera recomendações de otimização baseado em features de campanhas e classificação.

#### Tipos de Recomendações

| Tipo | Trigger | Ação Sugerida | Prioridade |
|------|---------|---------------|------------|
| `BUDGET_INCREASE` | HIGH_PERFORMER, CPL baixo | Aumentar orçamento 20-50% | 8 |
| `BUDGET_DECREASE` | CPL alto, tendência negativa | Reduzir orçamento 20-30% | 7 |
| `PAUSE_CAMPAIGN` | UNDERPERFORMER, sem leads | Pausar campanha | 9 |
| `SCALE_UP` | HIGH_PERFORMER consistente | Escalar investimento | 8 |
| `CREATIVE_REFRESH` | CTR baixo, frequência alta | Trocar criativos | 6 |
| `AUDIENCE_REVIEW` | Frequência > 5, alcance saturado | Revisar audiência | 7 |
| `REACTIVATE` | Pausada com histórico bom | Reativar campanha | 5 |
| `OPTIMIZE_SCHEDULE` | Performance variável por dia | Ajustar horários | 4 |

#### Estrutura de Recomendação

```python
@dataclass
class Recommendation:
    config_id: int
    entity_type: str
    entity_id: str
    recommendation_type: RecommendationType
    priority: int           # 1-10 (10 = mais urgente)
    title: str
    description: str
    suggested_action: dict  # {field, current, suggested, impact}
    confidence_score: float
    reasoning: dict
    expires_in_days: int
```

#### Regras de Negócio

```python
# Exemplo: Regra para BUDGET_INCREASE
if (
    tier == 'HIGH_PERFORMER' and
    cpl_ratio < 0.7 and
    days_with_leads_ratio > 0.5 and
    cpl_trend < 10
):
    return Recommendation(
        type=BUDGET_INCREASE,
        priority=8,
        suggested_action={
            "field": "daily_budget",
            "change_percent": 30,
            "expected_impact": "Aumento de leads proporcional"
        }
    )
```

---

## 5. Engenharia de Features

**Arquivo:** `app/services/feature_engineering.py`

### 5.1 Features Extraídas

O sistema extrai **25+ features** de cada campanha:

#### Métricas Básicas (7 dias)

| Feature | Tipo | Descrição |
|---------|------|-----------|
| `spend_7d` | float | Gasto total últimos 7 dias |
| `impressions_7d` | int | Impressões totais |
| `clicks_7d` | int | Cliques totais |
| `leads_7d` | int | Leads gerados |
| `cpl_7d` | float | Custo por lead |
| `ctr_7d` | float | Taxa de cliques (%) |
| `cpc_7d` | float | Custo por clique |
| `conversion_rate_7d` | float | Taxa de conversão (%) |

#### Tendências (7d vs 7d anterior)

| Feature | Tipo | Descrição |
|---------|------|-----------|
| `cpl_trend` | float | Variação % do CPL |
| `leads_trend` | float | Variação % de leads |
| `spend_trend` | float | Variação % de gastos |
| `ctr_trend` | float | Variação % do CTR |

#### Métricas Estendidas

| Feature | Tipo | Descrição |
|---------|------|-----------|
| `cpl_14d` | float | CPL médio 14 dias |
| `leads_14d` | int | Leads 14 dias |
| `cpl_30d` | float | CPL médio 30 dias |
| `leads_30d` | int | Leads 30 dias |
| `avg_daily_spend_30d` | float | Gasto diário médio |

#### Volatilidade

| Feature | Tipo | Descrição |
|---------|------|-----------|
| `cpl_std_7d` | float | Desvio padrão CPL |
| `leads_std_7d` | float | Desvio padrão leads |

#### Sazonalidade

| Feature | Tipo | Descrição |
|---------|------|-----------|
| `best_day_of_week` | int | Melhor dia (0-6) |
| `worst_day_of_week` | int | Pior dia (0-6) |

#### Consistência

| Feature | Tipo | Descrição |
|---------|------|-----------|
| `frequency_7d` | float | Frequência média |
| `reach_7d` | int | Alcance total |
| `days_with_leads_7d` | int | Dias com leads |
| `days_active` | int | Total dias ativos |

#### Status

| Feature | Tipo | Descrição |
|---------|------|-----------|
| `is_active` | bool | Campanha ativa? |
| `has_budget` | bool | Tem orçamento? |

### 5.2 Cálculos

```python
# CPL Ratio (normalizado)
cpl_ratio = campaign_cpl / avg_cpl_reference

# Tendência (% de mudança)
trend = ((current - previous) / previous) * 100

# Frequência Score (penaliza alta frequência)
frequency_score = max(0, 1 - (frequency - 1) / 3)

# Consistência Score (composto)
consistency_score = (
    days_with_leads_ratio * 0.5 +
    (1 - abs(cpl_trend) / 50) * 0.3 +
    frequency_score * 0.2
)
```

---

## 6. Agente IA Conversacional

### 6.1 Visão Geral

O Agente IA é um assistente conversacional para análise de tráfego pago, construído com **LangGraph** e integrando **Claude (Anthropic)** ou **GPT (OpenAI)**.

### 6.2 Arquitetura do Grafo

```
                    START
                      │
                      ▼
            ┌─────────────────┐
            │ classify_intent │  ◄── Classifica intenção do usuário
            └────────┬────────┘
                     │
            ┌────────┴────────┐
            ▼                 ▼
    ┌──────────────┐   ┌──────────────┐
    │  gather_data │   │  call_model  │
    └──────┬───────┘   └──────┬───────┘
           │                  │
           ▼                  │
    ┌──────────────┐          │
    │check_quality │          │
    └──────┬───────┘          │
           │                  │
           └──────────────────┤
                              ▼
                    ┌─────────────────┐
                    │   call_model    │  ◄── Chama LLM (Claude/GPT)
                    └────────┬────────┘
                             │
                ┌────────────┼────────────┐
                ▼            ▼            ▼
        ┌────────────┐ ┌──────────────┐ ┌────────────┐
        │ call_tools │ │   generate   │ │handle_error│
        └─────┬──────┘ │   response   │ └─────┬──────┘
              │        └──────┬───────┘       │
              │               │               │
              └───────────────┴───────────────┘
                              │
                              ▼
                            END
```

### 6.3 Ferramentas do Agente (21 Tools)

#### Classificação

| Tool | Descrição |
|------|-----------|
| `get_classifications` | Lista todas as classificações |
| `get_campaign_tier` | Tier de uma campanha |
| `get_high_performers` | Campanhas de alta performance |
| `get_underperformers` | Campanhas com baixa performance |

#### Recomendações

| Tool | Descrição |
|------|-----------|
| `get_recommendations` | Lista recomendações ativas |
| `get_recommendations_by_type` | Filtra por tipo |
| `get_high_priority_recommendations` | Prioridade alta |

#### Anomalias

| Tool | Descrição |
|------|-----------|
| `get_anomalies` | Lista anomalias detectadas |
| `get_critical_anomalies` | Apenas críticas |
| `get_anomalies_by_type` | Filtra por tipo |

#### Previsões

| Tool | Descrição |
|------|-----------|
| `get_forecasts` | Previsões existentes |
| `predict_campaign_cpl` | Prever CPL |
| `predict_campaign_leads` | Prever leads |

#### Campanhas

| Tool | Descrição |
|------|-----------|
| `get_campaign_details` | Detalhes de campanha |
| `list_campaigns` | Listar campanhas |

#### Análise

| Tool | Descrição |
|------|-----------|
| `compare_campaigns` | Comparar campanhas |
| `analyze_trends` | Analisar tendências |
| `get_account_summary` | Resumo da conta |
| `calculate_roi` | Calcular ROI |
| `get_top_campaigns` | Top campanhas |

### 6.4 Persistência de Estado

O agente usa **PostgreSQL Checkpointer** do LangGraph para manter histórico de conversas:

```python
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

checkpointer = AsyncPostgresSaver.from_conn_string(database_url)
await checkpointer.setup()  # Cria tabelas se não existirem

agent = graph.compile(checkpointer=checkpointer)
```

### 6.5 Configuração do LLM

```python
# Via variáveis de ambiente
AGENT_LLM_PROVIDER=anthropic  # ou 'openai'
AGENT_LLM_MODEL=claude-sonnet-4-20250514
AGENT_TEMPERATURE=0.3
AGENT_MAX_TOKENS=4096

# API Keys
AGENT_ANTHROPIC_API_KEY=sk-ant-...
AGENT_OPENAI_API_KEY=sk-proj-...
```

---

## 7. API REST

### 7.1 Autenticação

**Método:** API Key via header `X-API-Key`

```bash
curl -H "X-API-Key: sua-chave-aqui" \
     https://ml.famachat.com/api/v1/health
```

### 7.2 Endpoints

#### Health Check (Sem Auth)

| Método | Endpoint | Descrição |
|--------|----------|-----------|
| GET | `/api/v1/health` | Status básico |
| GET | `/api/v1/health/detailed` | Status com dependências |
| GET | `/api/v1/health/ready` | Readiness probe |
| GET | `/api/v1/health/live` | Liveness probe |

#### Predictions

| Método | Endpoint | Descrição |
|--------|----------|-----------|
| POST | `/api/v1/predictions/cpl` | Prever CPL |
| POST | `/api/v1/predictions/leads` | Prever leads |
| POST | `/api/v1/predictions/batch` | Previsões em lote |
| GET | `/api/v1/predictions/series/{type}/{id}` | Histórico |

#### Classifications

| Método | Endpoint | Descrição |
|--------|----------|-----------|
| GET | `/api/v1/classifications/campaigns` | Listar classificações |
| GET | `/api/v1/classifications/campaigns/{id}` | Classificação específica |
| POST | `/api/v1/classifications/campaigns/classify` | Classificar agora |
| POST | `/api/v1/classifications/train` | Treinar modelo |
| GET | `/api/v1/classifications/summary` | Resumo |

#### Recommendations

| Método | Endpoint | Descrição |
|--------|----------|-----------|
| GET | `/api/v1/recommendations` | Listar recomendações |
| POST | `/api/v1/recommendations/generate` | Gerar novas |
| POST | `/api/v1/recommendations/{id}/apply` | Marcar como aplicada |
| POST | `/api/v1/recommendations/{id}/dismiss` | Descartar |
| GET | `/api/v1/recommendations/summary` | Resumo |

#### Anomalies

| Método | Endpoint | Descrição |
|--------|----------|-----------|
| GET | `/api/v1/anomalies` | Listar anomalias |
| GET | `/api/v1/anomalies/summary` | Resumo |
| POST | `/api/v1/anomalies/detect` | Detectar agora |

#### Forecasts

| Método | Endpoint | Descrição |
|--------|----------|-----------|
| POST | `/api/v1/forecasts/cpl` | Forecast de CPL |
| POST | `/api/v1/forecasts/leads` | Forecast de leads |
| GET | `/api/v1/forecasts/{id}` | Obter forecast |
| GET | `/api/v1/forecasts/series/{entity_id}` | Série histórica |

#### Models

| Método | Endpoint | Descrição |
|--------|----------|-----------|
| GET | `/api/v1/models` | Listar modelos |
| GET | `/api/v1/models/{id}` | Obter modelo |
| GET | `/api/v1/models/active/{type}` | Modelo ativo por tipo |
| GET | `/api/v1/models/{id}/metrics` | Métricas do modelo |

#### Agent (Chat)

| Método | Endpoint | Descrição |
|--------|----------|-----------|
| POST | `/api/v1/agent/chat` | Chat síncrono |
| POST | `/api/v1/agent/chat/stream` | Chat com streaming |
| POST | `/api/v1/agent/analyze` | Análise profunda |
| POST | `/api/v1/agent/suggestions` | Sugestões |
| GET | `/api/v1/agent/conversations/{thread_id}` | Histórico |
| GET | `/api/v1/agent/conversations` | Listar conversas |
| POST | `/api/v1/agent/conversations/{thread_id}/clear` | Limpar |
| POST | `/api/v1/agent/feedback` | Feedback |
| GET | `/api/v1/agent/status` | Status do agente |

### 7.3 Exemplos de Uso

**Classificar Campanhas:**
```bash
curl -X POST "https://ml.famachat.com/api/v1/classifications/campaigns/classify" \
     -H "X-API-Key: $API_KEY" \
     -H "Content-Type: application/json" \
     -d '{"config_id": 1}'
```

**Obter Recomendações:**
```bash
curl "https://ml.famachat.com/api/v1/recommendations?config_id=1&active_only=true" \
     -H "X-API-Key: $API_KEY"
```

**Chat com Agente:**
```bash
curl -X POST "https://ml.famachat.com/api/v1/agent/chat" \
     -H "X-API-Key: $API_KEY" \
     -H "Content-Type: application/json" \
     -d '{
       "message": "Quais campanhas estão com performance ruim?",
       "config_id": 1,
       "thread_id": "conversa-123"
     }'
```

---

## 8. Banco de Dados

### 8.1 Tabelas ML

O sistema utiliza 8 tabelas dedicadas para ML:

```sql
-- 1. Modelos treinados
CREATE TABLE ml_trained_models (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    model_type model_type_enum NOT NULL,
    version VARCHAR(50) NOT NULL,
    config_id INTEGER,
    model_path VARCHAR(500) NOT NULL,
    parameters JSONB,
    feature_columns JSONB,
    training_metrics JSONB,
    validation_metrics JSONB,
    status model_status_enum DEFAULT 'TRAINING',
    is_active BOOLEAN DEFAULT FALSE,
    training_data_start TIMESTAMP,
    training_data_end TIMESTAMP,
    samples_count INTEGER,
    created_at TIMESTAMP DEFAULT NOW(),
    trained_at TIMESTAMP,
    last_used_at TIMESTAMP
);

-- 2. Previsões
CREATE TABLE ml_predictions (
    id SERIAL PRIMARY KEY,
    model_id INTEGER REFERENCES ml_trained_models(id),
    config_id INTEGER NOT NULL,
    entity_type VARCHAR(50) NOT NULL,
    entity_id VARCHAR(100) NOT NULL,
    prediction_type prediction_type_enum NOT NULL,
    forecast_date TIMESTAMP NOT NULL,
    horizon_days INTEGER DEFAULT 1,
    predicted_value FLOAT NOT NULL,
    confidence_lower FLOAT,
    confidence_upper FLOAT,
    actual_value FLOAT,
    absolute_error FLOAT,
    percentage_error FLOAT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- 3. Classificações
CREATE TABLE ml_campaign_classifications (
    id SERIAL PRIMARY KEY,
    config_id INTEGER NOT NULL,
    campaign_id VARCHAR(100) NOT NULL,
    tier campaign_tier_enum NOT NULL,
    confidence_score FLOAT NOT NULL,
    metrics_snapshot JSONB,
    feature_importances JSONB,
    previous_tier campaign_tier_enum,
    tier_change_direction VARCHAR(20),
    classified_at TIMESTAMP DEFAULT NOW(),
    valid_until TIMESTAMP
);

-- 4. Recomendações
CREATE TABLE ml_recommendations (
    id SERIAL PRIMARY KEY,
    config_id INTEGER NOT NULL,
    entity_type VARCHAR(50) NOT NULL,
    entity_id VARCHAR(100) NOT NULL,
    recommendation_type recommendation_type_enum NOT NULL,
    priority INTEGER DEFAULT 5,
    title VARCHAR(255) NOT NULL,
    description TEXT NOT NULL,
    suggested_action JSONB,
    confidence_score FLOAT DEFAULT 0.5,
    reasoning JSONB,
    is_active BOOLEAN DEFAULT TRUE,
    was_applied BOOLEAN DEFAULT FALSE,
    applied_at TIMESTAMP,
    applied_by INTEGER,
    dismissed BOOLEAN DEFAULT FALSE,
    dismissed_at TIMESTAMP,
    dismissed_by INTEGER,
    dismissed_reason TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    expires_at TIMESTAMP
);

-- 5. Anomalias
CREATE TABLE ml_anomalies (
    id SERIAL PRIMARY KEY,
    config_id INTEGER NOT NULL,
    entity_type VARCHAR(50) NOT NULL,
    entity_id VARCHAR(100) NOT NULL,
    anomaly_type VARCHAR(50) NOT NULL,
    metric_name VARCHAR(50) NOT NULL,
    observed_value FLOAT NOT NULL,
    expected_value FLOAT NOT NULL,
    deviation_score FLOAT NOT NULL,
    severity anomaly_severity_enum NOT NULL,
    is_acknowledged BOOLEAN DEFAULT FALSE,
    acknowledged_at TIMESTAMP,
    acknowledged_by INTEGER,
    resolution_notes TEXT,
    anomaly_date TIMESTAMP NOT NULL,
    detected_at TIMESTAMP DEFAULT NOW(),
    recommendation_id INTEGER REFERENCES ml_recommendations(id)
);

-- 6. Features
CREATE TABLE ml_features (
    id SERIAL PRIMARY KEY,
    config_id INTEGER NOT NULL,
    campaign_id VARCHAR(100) NOT NULL,
    window_days INTEGER NOT NULL,
    feature_date TIMESTAMP NOT NULL,
    features JSONB,
    insufficient_data BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT NOW()
);

-- 7. Forecasts
CREATE TABLE ml_forecasts (
    id SERIAL PRIMARY KEY,
    config_id INTEGER NOT NULL,
    entity_type VARCHAR(50) NOT NULL,
    entity_id VARCHAR(100) NOT NULL,
    target_metric VARCHAR(50) NOT NULL,
    horizon_days INTEGER DEFAULT 7,
    method VARCHAR(50) NOT NULL,
    predictions JSONB,
    forecast_date TIMESTAMP NOT NULL,
    window_days INTEGER,
    model_version VARCHAR(50),
    insufficient_data BOOLEAN DEFAULT FALSE,
    prediction_type prediction_type_enum,
    created_at TIMESTAMP DEFAULT NOW()
);

-- 8. Training Jobs
CREATE TABLE ml_training_jobs (
    id SERIAL PRIMARY KEY,
    model_type model_type_enum NOT NULL,
    config_id INTEGER,
    celery_task_id VARCHAR(255),
    status job_status_enum DEFAULT 'PENDING',
    progress FLOAT DEFAULT 0.0,
    model_id INTEGER REFERENCES ml_trained_models(id),
    error_message TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    started_at TIMESTAMP,
    completed_at TIMESTAMP
);
```

### 8.2 Enums

```sql
CREATE TYPE model_type_enum AS ENUM (
    'TIME_SERIES_CPL', 'TIME_SERIES_LEADS',
    'CAMPAIGN_CLASSIFIER', 'ANOMALY_DETECTOR', 'RECOMMENDER'
);

CREATE TYPE model_status_enum AS ENUM (
    'TRAINING', 'READY', 'ACTIVE', 'DEPRECATED', 'FAILED'
);

CREATE TYPE prediction_type_enum AS ENUM (
    'CPL_FORECAST', 'LEADS_FORECAST', 'SPEND_FORECAST'
);

CREATE TYPE campaign_tier_enum AS ENUM (
    'HIGH_PERFORMER', 'MODERATE', 'LOW', 'UNDERPERFORMER'
);

CREATE TYPE recommendation_type_enum AS ENUM (
    'BUDGET_INCREASE', 'BUDGET_DECREASE', 'PAUSE_CAMPAIGN',
    'SCALE_UP', 'CREATIVE_REFRESH', 'AUDIENCE_REVIEW',
    'REACTIVATE', 'OPTIMIZE_SCHEDULE'
);

CREATE TYPE anomaly_severity_enum AS ENUM (
    'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'
);

CREATE TYPE job_status_enum AS ENUM (
    'PENDING', 'RUNNING', 'COMPLETED', 'FAILED', 'CANCELLED'
);
```

### 8.3 Índices

```sql
-- Performance indexes
CREATE INDEX ix_ml_trained_models_type_active ON ml_trained_models(model_type, is_active);
CREATE INDEX ix_ml_predictions_entity_date ON ml_predictions(entity_type, entity_id, forecast_date);
CREATE INDEX ix_ml_classifications_campaign ON ml_campaign_classifications(config_id, campaign_id, classified_at);
CREATE INDEX ix_ml_recommendations_active ON ml_recommendations(config_id, is_active);
CREATE INDEX ix_ml_anomalies_severity ON ml_anomalies(config_id, severity);
```

---

## 9. Jobs Agendados (Celery)

### 9.1 Configuração

```python
# app/tasks/celery_app.py
celery_app = Celery(
    "famachat-ml",
    broker="redis://localhost:6380/0",
    backend="redis://localhost:6380/0",
)

celery_app.conf.update(
    timezone="America/Sao_Paulo",
    worker_concurrency=2,
    task_default_rate_limit="10/m",
)
```

### 9.2 Jobs Agendados

| Job | Horário | Frequência | Descrição |
|-----|---------|------------|-----------|
| `daily-ml-pipeline` | 02:00 | Diária | Pipeline ML completo |
| `daily-model-retraining` | 05:00 | Diária | Retreinar modelos |
| `daily-classification` | 06:00 | Diária | Classificar campanhas |
| `daily-recommendations` | 07:00 | Diária | Gerar recomendações |
| `hourly-anomaly-detection` | *:30 | Horária | Detectar anomalias |
| `batch-predictions` | */4h:15 | 4 em 4h | Previsões em lote |
| `daily-prediction-validation` | 08:00 | Diária | Validar previsões |

### 9.3 Filas

| Fila | Uso |
|------|-----|
| `default` | Tasks gerais |
| `training` | Treinamento de modelos (CPU intensivo) |
| `ml` | Inferência e previsões |

### 9.4 Comandos

```bash
# Iniciar worker
celery -A app.tasks.celery_app worker --loglevel=info --queues=default,training,ml

# Iniciar scheduler
celery -A app.tasks.celery_app beat --loglevel=info

# Monitoramento
celery -A app.tasks.celery_app flower --port=5555
```

---

## 10. Integração com FamaChat

### 10.1 Banco de Dados Compartilhado

O FamaChat ML lê dados das tabelas do FamaChat principal via **modelos read-only**:

```python
# app/db/models/famachat_readonly.py
class FacebookAdsCampaign(Base):
    """Tabela sistema_facebook_ads_campaigns (somente leitura)"""
    __tablename__ = "sistema_facebook_ads_campaigns"
    __table_args__ = {"schema": "public"}

class FacebookAdsInsight(Base):
    """Tabela sistema_facebook_ads_insights (somente leitura)"""
    __tablename__ = "sistema_facebook_ads_insights"
```

### 10.2 Autenticação Compartilhada

Usa o mesmo **JWT_SECRET** do FamaChat para validar tokens:

```python
# app/core/security.py
async def verify_api_key(api_key: str) -> bool:
    """Verifica API Key ou JWT."""
    if api_key == settings.ml_api_key:
        return True
    # Fallback para JWT
    return verify_jwt_token(api_key, settings.jwt_secret)
```

### 10.3 Chamadas do FamaChat

O FamaChat principal faz chamadas HTTP para o ML:

```typescript
// FamaChat - server/services/ml-integration.ts
export async function getMLRecommendations(configId: number) {
  const response = await fetch(
    `${process.env.ML_API_URL}/api/v1/recommendations?config_id=${configId}`,
    {
      headers: { 'X-API-Key': process.env.ML_API_KEY }
    }
  );
  return response.json();
}
```

---

## 11. Deploy e Infraestrutura

### 11.1 Docker Compose

```yaml
services:
  # API FastAPI
  famachat-ml-api:
    image: famachat-ml:latest
    ports: ["8000:8000"]
    environment:
      - DATABASE_URL=${DATABASE_URL}
      - REDIS_URL=redis://famachat-ml-redis:6379/0
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 2G

  # Celery Worker
  famachat-ml-worker:
    image: famachat-ml:latest
    command: celery -A app.tasks.celery_app worker
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 3G

  # Celery Beat
  famachat-ml-beat:
    image: famachat-ml:latest
    command: celery -A app.tasks.celery_app beat

  # Redis
  famachat-ml-redis:
    image: redis:7-alpine
    ports: ["6380:6379"]

  # Flower
  famachat-ml-flower:
    image: famachat-ml:latest
    command: celery -A app.tasks.celery_app flower
    ports: ["5555:5555"]
```

### 11.2 Recursos Recomendados

| Serviço | CPU | RAM | Disco |
|---------|-----|-----|-------|
| API | 2 cores | 2 GB | 1 GB |
| Worker | 2 cores | 3 GB | 2 GB |
| Beat | 0.5 core | 256 MB | 100 MB |
| Redis | 0.5 core | 512 MB | 1 GB |

### 11.3 Variáveis de Ambiente

```env
# Aplicação
APP_NAME=FamaChat ML
APP_VERSION=1.0.0
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO

# Database
DATABASE_URL=postgresql://user:pass@host:5432/famachat

# Redis
REDIS_URL=redis://localhost:6380/0

# Segurança
ML_API_KEY=sua-chave-api-aqui
JWT_SECRET=compartilhado-com-famachat

# Agente IA
AGENT_LLM_PROVIDER=anthropic
AGENT_LLM_MODEL=claude-sonnet-4-20250514
AGENT_ANTHROPIC_API_KEY=sk-ant-...
AGENT_TEMPERATURE=0.3

# Storage
MODELS_STORAGE_PATH=/app/models_storage

# Flower
FLOWER_USER=admin
FLOWER_PASSWORD=sua-senha-aqui
```

---

## 12. Monitoramento e Observabilidade

### 12.1 Health Checks

```bash
# Liveness
curl http://localhost:8000/api/v1/health/live

# Readiness
curl http://localhost:8000/api/v1/health/ready

# Detailed
curl http://localhost:8000/api/v1/health/detailed
```

Resposta detailed:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2026-01-19T10:00:00Z",
  "checks": {
    "database": "ok",
    "redis": "ok",
    "celery": "ok"
  }
}
```

### 12.2 Logging Estruturado

```python
# structlog - JSON format
{
  "timestamp": "2026-01-19T10:00:00Z",
  "level": "info",
  "logger": "classification_service",
  "event": "Campanha classificada",
  "campaign_id": "123456",
  "tier": "HIGH_PERFORMER",
  "confidence": 0.85
}
```

### 12.3 Flower (Celery)

Acesse `http://localhost:5555` para:
- Monitorar workers ativos
- Ver tasks em execução
- Histórico de tasks
- Estatísticas de filas

### 12.4 Métricas de Modelos

```bash
curl http://localhost:8000/api/v1/models/1/metrics \
     -H "X-API-Key: $API_KEY"
```

Resposta:
```json
{
  "model_id": 1,
  "accuracy": 0.87,
  "f1_weighted": 0.85,
  "samples_trained": 1500,
  "last_trained": "2026-01-18T05:00:00Z",
  "predictions_count": 10500,
  "avg_confidence": 0.82
}
```

---

## 13. Segurança

### 13.1 Autenticação

- **API Key**: Header `X-API-Key` obrigatório
- **JWT**: Suporte a tokens JWT do FamaChat
- **Rate Limiting**: 10 requests/minuto por task

### 13.2 Proteções

- CORS configurado (origem específica em produção)
- Validação de entrada com Pydantic
- Queries parametrizadas (SQLAlchemy)
- Sanitização de logs (sem dados sensíveis)

### 13.3 Boas Práticas

```python
# Nunca logar dados sensíveis
logger.info("Processando", config_id=config_id)  # OK
logger.info("Processando", api_key=api_key)      # ERRADO

# Sempre validar entrada
from pydantic import BaseModel, validator

class ClassifyRequest(BaseModel):
    config_id: int
    campaign_ids: list[str] | None = None

    @validator('config_id')
    def validate_config_id(cls, v):
        if v <= 0:
            raise ValueError('config_id deve ser positivo')
        return v
```

---

## 14. Guia de Uso

### 14.1 Cenário: Classificar Campanhas

```python
# 1. Carregar dados de insights
insights = await data_service.get_campaign_insights(config_id)

# 2. Extrair features
features = feature_engineer.compute_campaign_features(insights, campaign_info)

# 3. Classificar
result = classifier.classify(features, avg_cpl, avg_ctr)

# 4. Salvar
await ml_repo.save_classification(result)
```

### 14.2 Cenário: Detectar Anomalias

```python
# 1. Obter dados diários
df = await data_service.get_daily_metrics(config_id, campaign_id)

# 2. Detectar
anomalies = detector.detect_anomalies(df, "campaign", campaign_id)

# 3. Filtrar críticas
critical = [a for a in anomalies if a.severity == "CRITICAL"]

# 4. Salvar e notificar
for anomaly in critical:
    await ml_repo.save_anomaly(anomaly)
    await notify_critical_anomaly(anomaly)
```

### 14.3 Cenário: Gerar Previsões

```python
# 1. Obter série temporal
df = await data_service.get_time_series(config_id, campaign_id, metric="cpl")

# 2. Prever próximos 7 dias
forecast = forecaster.forecast_cpl(df, "campaign", campaign_id, horizon_days=7)

# 3. Salvar
for f in forecast.forecasts:
    await ml_repo.save_forecast(f)
```

### 14.4 Cenário: Chat com Agente

```python
# 1. Criar sessão
thread_id = f"user_{user_id}_session_{datetime.now().timestamp()}"

# 2. Enviar mensagem
response = await agent_service.chat(
    message="Quais campanhas precisam de atenção?",
    config_id=config_id,
    thread_id=thread_id
)

# 3. Resposta inclui:
# - Texto explicativo em português
# - Dados das campanhas problemáticas
# - Recomendações de ação
```

---

## 15. Troubleshooting

### 15.1 API não responde

```bash
# Verificar logs
docker logs famachat-ml-api -f

# Verificar health
curl http://localhost:8000/api/v1/health

# Verificar banco
docker exec -it famachat-ml-api python -c "from app.db.session import check_database_connection; import asyncio; print(asyncio.run(check_database_connection()))"
```

### 15.2 Tasks Celery não executam

```bash
# Verificar worker
docker logs famachat-ml-worker -f

# Verificar Redis
redis-cli -p 6380 PING

# Ver tasks pendentes
docker exec famachat-ml-worker celery -A app.tasks.celery_app inspect active
```

### 15.3 Modelo não classifica corretamente

```python
# Verificar se modelo está treinado
classifier = CampaignClassifier(model_path="models_storage/classifier.pkl")
print(f"Modelo treinado: {classifier.is_fitted}")

# Forçar classificação por regras
result = classifier.classify_by_rules(features, avg_cpl, avg_ctr)
```

### 15.4 Agente não responde

```bash
# Verificar API key do LLM
echo $AGENT_ANTHROPIC_API_KEY

# Testar manualmente
curl https://api.anthropic.com/v1/messages \
     -H "x-api-key: $AGENT_ANTHROPIC_API_KEY" \
     -H "content-type: application/json" \
     -d '{"model":"claude-sonnet-4-20250514","max_tokens":100,"messages":[{"role":"user","content":"Teste"}]}'
```

---

## 16. Referências Técnicas

### 16.1 Arquivos Principais

| Arquivo | Linhas | Descrição |
|---------|--------|-----------|
| `app/ml/models/classification/campaign_classifier.py` | 464 | Classificador XGBoost |
| `app/ml/models/anomaly/anomaly_detector.py` | 478 | Detector de anomalias |
| `app/ml/models/timeseries/forecaster.py` | 396 | Forecaster de séries |
| `app/services/feature_engineering.py` | 407 | Engenharia de features |
| `app/services/recommendation_service.py` | 384 | Serviço de recomendações |
| `app/agent/service.py` | ~500 | Serviço do agente IA |
| `app/agent/graph/builder.py` | 163 | Construtor do grafo |
| `app/tasks/scheduled_tasks.py` | 710 | Tasks agendadas |

### 16.2 Dependências Principais

```
fastapi==0.109.0
scikit-learn==1.4.0
xgboost==2.0.3
pandas==2.2.0
langgraph>=0.2.0
langchain-anthropic>=0.2.0
celery==5.3.6
sqlalchemy==2.0.25
```

### 16.3 Links Úteis

- **FastAPI Docs**: https://fastapi.tiangolo.com
- **XGBoost**: https://xgboost.readthedocs.io
- **LangGraph**: https://langchain-ai.github.io/langgraph
- **Celery**: https://docs.celeryq.dev

---

## Changelog

### v1.0.0 (Janeiro 2026)
- Implementação inicial do sistema ML
- 4 modelos de ML (Classificação, Anomalia, Previsão, Recomendação)
- Agente IA com LangGraph
- 40+ endpoints REST
- 8 jobs agendados
- Integração completa com FamaChat

---

**Documento gerado em:** Janeiro 2026
**Próxima revisão:** Abril 2026
