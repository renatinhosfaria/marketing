# FamaChat ML - Documentação Completa do Agente de IA

**Versão:** 1.0.0
**Última Atualização:** Janeiro 2026
**Autor:** Equipe FamaChat

---

## Sumário

1. [Visão Geral](#1-visão-geral)
2. [Arquitetura do Sistema](#2-arquitetura-do-sistema)
3. [Stack Tecnológico](#3-stack-tecnológico)
4. [Agente LangGraph](#4-agente-langgraph)
5. [Tools do Agente](#5-tools-do-agente)
6. [API REST](#6-api-rest)
7. [Modelos de Machine Learning](#7-modelos-de-machine-learning)
8. [Sistema de Persistência](#8-sistema-de-persistência)
9. [Tarefas Agendadas (Celery)](#9-tarefas-agendadas-celery)
10. [Configuração e Deploy](#10-configuração-e-deploy)
11. [Monitoramento e Observabilidade](#11-monitoramento-e-observabilidade)
12. [Integração com FamaChat Principal](#12-integração-com-famachat-principal)
13. [Segurança](#13-segurança)
14. [Guia de Uso](#14-guia-de-uso)
15. [Troubleshooting](#15-troubleshooting)

---

## 1. Visão Geral

### 1.1 O que é o FamaChat ML?

O **FamaChat ML** é um microserviço Python especializado em Machine Learning que complementa o sistema FamaChat principal. Ele implementa um **Agente de IA Conversacional** focado em otimização de campanhas de Facebook Ads para o mercado imobiliário.

### 1.2 Principais Funcionalidades

| Funcionalidade | Descrição |
|----------------|-----------|
| **Análise Inteligente de Campanhas** | Classificação automática por tiers de performance (HIGH_PERFORMER, MODERATE, LOW, UNDERPERFORMER) |
| **Detecção de Anomalias** | Identificação em tempo real de comportamentos atípicos (CPL alto, frequência excessiva, zero leads) |
| **Previsões Temporais** | Forecast de CPL, leads e gastos para 7-30 dias |
| **Recomendações Automatizadas** | Sugestões acionáveis baseadas em regras de negócio e ML |
| **Agente Conversacional** | Interface natural em português para análise interativa de dados |
| **Cálculo de ROI** | Projeções de retorno sobre investimento |

### 1.3 Proposta de Valor

O agente permite que gestores de tráfego pago façam perguntas em linguagem natural como:

- *"Qual campanha está com o melhor CPL?"*
- *"Tem alguma anomalia crítica que eu preciso resolver?"*
- *"Compare minhas top 3 campanhas"*
- *"Qual campanha devo pausar?"*
- *"Previsão de leads para próxima semana"*

E recebam respostas contextualizadas, com métricas reais e recomendações acionáveis.

---

## 2. Arquitetura do Sistema

### 2.1 Diagrama de Alto Nível

```
┌─────────────────────────────────────────────────────────────────────┐
│                          FamaChat Principal                         │
│                       (Node.js / Express.js)                        │
└───────────────────────────────┬─────────────────────────────────────┘
                                │ HTTP/REST + JWT
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                          FamaChat ML                                │
│                          (Python/FastAPI)                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐  │
│  │  API REST    │  │  LangGraph   │  │  Celery Workers          │  │
│  │  (FastAPI)   │◄─┤  Agent       │  │  (Background Tasks)      │  │
│  └──────────────┘  └──────────────┘  └──────────────────────────┘  │
│         │                  │                     │                  │
│         ▼                  ▼                     ▼                  │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │              Camada de Serviços (ML Services)                │  │
│  │  Classification │ Recommendation │ Anomaly │ Forecast │ Data │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                │                                    │
└────────────────────────────────┼────────────────────────────────────┘
                                 │
          ┌──────────────────────┼──────────────────────┐
          ▼                      ▼                      ▼
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│   PostgreSQL     │  │     Redis        │  │   LLM Provider   │
│  (Compartilhado) │  │  (Cache/Broker)  │  │ (Claude/OpenAI)  │
└──────────────────┘  └──────────────────┘  └──────────────────┘
```

### 2.2 Estrutura de Diretórios

```
famachat-ml/
├── app/
│   ├── agent/                      # Agente IA LangGraph
│   │   ├── config.py               # Configurações do agente
│   │   ├── service.py              # Serviço principal (chat, stream)
│   │   ├── graph/                  # Grafo LangGraph
│   │   │   ├── state.py            # Definição do estado
│   │   │   ├── builder.py          # Construtor do grafo
│   │   │   ├── nodes.py            # 6 nós de processamento
│   │   │   └── edges.py            # 4 transições condicionais
│   │   ├── tools/                  # 20 tools do agente
│   │   │   ├── classification_tools.py
│   │   │   ├── recommendation_tools.py
│   │   │   ├── anomaly_tools.py
│   │   │   ├── forecast_tools.py
│   │   │   ├── campaign_tools.py
│   │   │   ├── analysis_tools.py
│   │   │   └── base.py
│   │   ├── prompts/                # System prompts
│   │   │   └── system.py           # Prompt principal (PT-BR)
│   │   ├── llm/                    # Factory de LLM
│   │   │   └── provider.py         # Claude/GPT factory
│   │   └── memory/                 # Persistência
│   │       └── checkpointer.py     # PostgreSQL Checkpointer
│   ├── api/v1/                     # API REST
│   │   ├── router.py               # Router principal
│   │   └── agent/                  # Endpoints do agente
│   │       ├── router.py           # 10+ endpoints
│   │       └── schemas.py          # Pydantic schemas
│   ├── ml/                         # Modelos ML
│   │   └── models/
│   │       ├── classification/     # XGBoost Classifier
│   │       ├── anomaly/            # Isolation Forest
│   │       ├── recommendation/     # Rule Engine
│   │       └── timeseries/         # ARIMA/Prophet
│   ├── services/                   # Lógica de negócio
│   │   ├── classification_service.py
│   │   ├── recommendation_service.py
│   │   ├── anomaly_service.py
│   │   └── data_service.py
│   ├── db/                         # Database
│   │   ├── session.py              # Engine + Sessions
│   │   ├── models/                 # SQLAlchemy Models
│   │   └── repositories/           # Repositórios
│   ├── tasks/                      # Celery
│   │   ├── celery_app.py           # Configuração
│   │   └── scheduled_tasks.py      # 8 jobs agendados
│   ├── core/                       # Utilitários
│   │   ├── security.py             # Auth (JWT + API Key)
│   │   └── logging.py              # structlog
│   ├── config.py                   # Configurações gerais
│   └── main.py                     # Entry point FastAPI
├── migrations/                     # Alembic migrations
├── models_storage/                 # Modelos serializados
├── logs/                           # Logs da aplicação
├── docker-compose.yml              # 5 serviços
├── Dockerfile                      # Multi-stage build
├── requirements.txt                # 35+ dependências
└── .env.example                    # Template de configuração
```

---

## 3. Stack Tecnológico

### 3.1 Tecnologias Principais

| Camada | Tecnologia | Versão | Propósito |
|--------|------------|--------|-----------|
| **Framework Web** | FastAPI | 0.109.0 | API REST de alta performance |
| **ASGI Server** | Uvicorn | 0.27.0 | Servidor assíncrono |
| **ORM** | SQLAlchemy | 2.0.25 | Acesso ao banco de dados |
| **Async DB** | asyncpg | 0.29.0 | Driver PostgreSQL assíncrono |
| **Task Queue** | Celery | 5.3.6 | Processamento em background |
| **Message Broker** | Redis | 5.0.1 | Broker + Cache |
| **Validação** | Pydantic | 2.7+ | Schemas e validação |
| **Logging** | structlog | 24.1.0 | Logs estruturados |

### 3.2 Machine Learning

| Biblioteca | Versão | Uso |
|------------|--------|-----|
| **scikit-learn** | 1.4.0 | Algoritmos base, métricas |
| **XGBoost** | 2.0.3 | Classificação de campanhas |
| **LightGBM** | 4.3.0 | Classificação alternativa |
| **statsmodels** | 0.14.1 | ARIMA para time series |
| **pandas** | 2.2.0 | Manipulação de dados |
| **numpy** | 1.26.3 | Computação numérica |
| **scipy** | 1.12.0 | Funções estatísticas |
| **joblib** | 1.3.2 | Serialização de modelos |

### 3.3 LangGraph / LangChain

| Biblioteca | Versão | Uso |
|------------|--------|-----|
| **langgraph** | 0.2+ | Framework de agentes stateful |
| **langchain-core** | 0.3+ | Abstrações core |
| **langchain-anthropic** | 0.2+ | Integração Claude |
| **langchain-openai** | 0.2+ | Integração GPT |
| **langgraph-checkpoint-postgres** | 1.0+ | Persistência de estado |
| **tiktoken** | 0.5+ | Tokenização |

---

## 4. Agente LangGraph

### 4.1 Conceito

O agente utiliza **LangGraph**, um framework da LangChain para construção de agentes com estado persistente. Diferente de chains simples, o LangGraph permite:

- **Estado Persistente**: Conversas continuam de onde pararam
- **Fluxo Condicional**: Decisões baseadas no contexto
- **Tool Calling**: Execução de ferramentas especializadas
- **Checkpointing**: Estado salvo automaticamente

### 4.2 Fluxo do Grafo

```
                            START
                              │
                              ▼
                    ┌─────────────────────┐
                    │   classify_intent   │
                    │  (detecta intenção) │
                    └─────────┬───────────┘
                              │
                    ┌─────────▼───────────┐
                    │   route_by_intent   │
                    │     (decisão)       │
                    └─────────┬───────────┘
                   ╱          │            ╲
        ┌─────────┘           │             └─────────┐
        ▼                     │                       ▼
┌───────────────┐             │              ┌───────────────┐
│  gather_data  │             │              │   call_model  │◄────┐
│(busca dados ML)             │              │  (chama LLM)  │     │
└───────┬───────┘             │              └───────┬───────┘     │
        │                     │                      │             │
        ▼                     │             ┌────────▼────────┐    │
┌───────────────┐             │             │ should_continue │    │
│check_data_qual│             │             │    (decisão)    │    │
└───────┬───────┘             │             └────────┬────────┘    │
        │                     │                ╱     │      ╲      │
   ┌────┴────┐                │         ┌────┘      │       └───┐ │
   ▼         ▼                │         ▼           ▼           ▼ │
call_model  handle_error      │    call_tools  generate_response  │
                              │         │           │           │ │
                              │         └───────────┼───────────┘ │
                              │                     │ after_tools │
                              │                     └─────────────┘
                              │
                              ▼
                             END
```

### 4.3 Estado do Agente (AgentState)

```python
class AgentState(TypedDict):
    # Mensagens da conversa (acumula automaticamente)
    messages: Annotated[Sequence[dict], add_messages]

    # Contexto da sessão
    config_id: int          # ID da conta Facebook Ads
    user_id: int            # ID do usuário autenticado
    thread_id: str          # ID único da conversa

    # Dados coletados durante análise
    classifications: Optional[list[dict]]    # Tiers de campanhas
    recommendations: Optional[list[dict]]    # Recomendações ativas
    anomalies: Optional[list[dict]]          # Anomalias detectadas
    forecasts: Optional[list[dict]]          # Previsões

    # Estado da análise atual
    current_intent: Optional[str]            # analyze|compare|recommend|forecast|troubleshoot|general
    selected_campaigns: list[str]            # Campanhas em foco
    analysis_result: Optional[dict]          # Resultado da análise

    # Metadados de execução
    tool_calls_count: int                    # Contador de tool calls
    last_error: Optional[str]                # Último erro (se houver)
```

### 4.4 Nós do Grafo

| Nó | Função | Descrição |
|----|--------|-----------|
| `classify_intent` | Classificação | Detecta a intenção do usuário analisando palavras-chave |
| `gather_data` | Coleta | Busca classificações, recomendações, anomalias e previsões relevantes |
| `call_model` | Processamento | Invoca o LLM (Claude/GPT) com contexto e tools |
| `call_tools` | Execução | Executa as tools solicitadas pelo modelo |
| `generate_response` | Formatação | Prepara a resposta final para o usuário |
| `handle_error` | Erro | Trata erros e gera mensagem amigável |

### 4.5 Transições Condicionais

| Edge | Origem | Decisão |
|------|--------|---------|
| `route_by_intent` | classify_intent | Decide se precisa buscar dados ML ou ir direto ao modelo |
| `check_data_quality` | gather_data | Verifica se houve erro na coleta de dados |
| `should_continue` | call_model | Decide se executa tools, gera resposta ou trata erro |
| `after_tools` | call_tools | Decide se volta ao modelo ou gera resposta |

### 4.6 Detecção de Intenção

O agente detecta automaticamente a intenção do usuário baseado em palavras-chave:

```python
intent_keywords = {
    "analyze": ["analise", "análise", "como está", "desempenho", "performance", "métricas"],
    "compare": ["compare", "comparar", "versus", "vs", "diferença", "melhor", "pior"],
    "recommend": ["recomend", "sugest", "o que fazer", "próximos passos", "ação"],
    "forecast": ["previsão", "prever", "futuro", "projeção", "estimar"],
    "troubleshoot": ["problema", "erro", "anomalia", "queda", "piorou", "crítico"],
}
```

### 4.7 System Prompt

O agente é instruído em português brasileiro com as seguintes diretrizes:

```
Você é um especialista em gestão de tráfego pago para Facebook Ads.
Seu papel é analisar campanhas, identificar oportunidades de otimização
e fornecer recomendações acionáveis baseadas em dados.

## Capacidades:
- Analisar classificações de performance (HIGH_PERFORMER, MODERATE, LOW, UNDERPERFORMER)
- Identificar anomalias (CPL alto, spend zerado, frequência alta)
- Interpretar previsões de CPL e leads
- Recomendar ações de otimização (escalar, pausar, ajustar budget)
- Comparar campanhas e identificar padrões
- Calcular ROI e projeções de receita

## Formato de Resposta:
- 📊 para métricas
- ✅ para ações positivas
- ⚠️ para alertas médios
- 🔴 para problemas críticos
- 📈 para tendências positivas
- 📉 para tendências negativas
- 💡 para sugestões
```

---

## 5. Tools do Agente

O agente possui **20 tools** organizadas em 6 categorias:

### 5.1 Tools de Classificação

| Tool | Descrição | Parâmetros |
|------|-----------|------------|
| `get_classifications` | Lista todas as classificações ativas | `config_id`, `limit`, `active_only` |
| `get_campaign_tier` | Retorna o tier de uma campanha específica | `config_id`, `campaign_id` |
| `get_high_performers` | Lista campanhas HIGH_PERFORMER | `config_id`, `limit` |
| `get_underperformers` | Lista campanhas UNDERPERFORMER | `config_id`, `limit` |

### 5.2 Tools de Recomendação

| Tool | Descrição | Parâmetros |
|------|-----------|------------|
| `get_recommendations` | Lista recomendações ativas | `config_id`, `active_only` |
| `get_recommendations_by_type` | Filtra por tipo (BUDGET_INCREASE, etc) | `config_id`, `type` |
| `get_high_priority_recommendations` | Lista recomendações urgentes | `config_id` |

### 5.3 Tools de Anomalia

| Tool | Descrição | Parâmetros |
|------|-----------|------------|
| `get_anomalies` | Lista anomalias detectadas | `config_id`, `days` |
| `get_critical_anomalies` | Lista anomalias críticas | `config_id` |
| `get_anomalies_by_type` | Filtra por tipo de anomalia | `config_id`, `type` |

### 5.4 Tools de Previsão

| Tool | Descrição | Parâmetros |
|------|-----------|------------|
| `get_forecasts` | Lista previsões disponíveis | `config_id`, `days_ahead` |
| `predict_campaign_cpl` | Previsão de CPL para campanha | `config_id`, `campaign_id`, `days` |
| `predict_campaign_leads` | Previsão de leads para campanha | `config_id`, `campaign_id`, `days` |

### 5.5 Tools de Campanha

| Tool | Descrição | Parâmetros |
|------|-----------|------------|
| `get_campaign_details` | Métricas completas de uma campanha | `config_id`, `campaign_id` |
| `list_campaigns` | Lista todas as campanhas | `config_id`, `status` |

### 5.6 Tools de Análise

| Tool | Descrição | Parâmetros |
|------|-----------|------------|
| `compare_campaigns` | Comparação lado-a-lado | `config_id`, `campaign_ids`, `days` |
| `analyze_trends` | Análise de tendências | `config_id`, `campaign_id`, `days` |
| `get_account_summary` | Resumo geral da conta | `config_id`, `days` |
| `calculate_roi` | Cálculo de ROI | `config_id`, `campaign_id`, `average_ticket`, `conversion_rate` |
| `get_top_campaigns` | Ranking por métrica | `config_id`, `metric`, `days`, `limit` |

### 5.7 Exemplo de Output de Tool

```python
# get_classifications output
{
    "total": 15,
    "by_tier": {
        "HIGH_PERFORMER": 3,
        "MODERATE": 7,
        "LOW": 3,
        "UNDERPERFORMER": 2
    },
    "classifications": [
        {
            "campaign_id": "123456",
            "campaign_name": "Campanha Leads Apartamentos",
            "tier": "HIGH_PERFORMER",
            "confidence": 92.5,
            "cpl_7d": 28.50,
            "leads_7d": 45,
            "spend_7d": 1282.50,
            "is_valid": True,
            "classified_at": "2026-01-19T02:00:00"
        }
    ],
    "summary": "Total de 15 campanhas classificadas: 3 high performers..."
}
```

---

## 6. API REST

### 6.1 Endpoints do Agente

#### Chat Completo
```http
POST /api/v1/agent/chat
Authorization: Bearer {jwt_token}
Content-Type: application/json

{
    "message": "Qual campanha está com o melhor CPL?",
    "config_id": 1,
    "thread_id": "optional-uuid"
}
```

**Resposta:**
```json
{
    "success": true,
    "thread_id": "550e8400-e29b-41d4-a716-446655440000",
    "response": "📊 Analisei suas campanhas e a com melhor CPL é...",
    "intent": "analyze",
    "tool_calls_count": 2
}
```

#### Chat com Streaming (SSE)
```http
POST /api/v1/agent/chat/stream
Authorization: Bearer {jwt_token}
Content-Type: application/json

{
    "message": "Compare minhas top 3 campanhas",
    "config_id": 1
}
```

**Eventos SSE:**
```
data: {"type": "stream_start", "thread_id": "...", "timestamp": 1705632000000}

data: {"type": "node_start", "node": "classify_intent", "timestamp": ...}

data: {"type": "intent_classified", "intent": "compare", "timestamp": ...}

data: {"type": "node_end", "node": "classify_intent", "duration_ms": 15}

data: {"type": "data_gathered", "data_counts": {"classifications": 15, ...}}

data: {"type": "tool_start", "tool": "compare_campaigns", "input_preview": "..."}

data: {"type": "tool_end", "tool": "compare_campaigns", "success": true, "duration_ms": 120}

data: {"type": "text", "content": "📊 Comparando suas top 3 campanhas..."}

data: {"type": "done", "total_duration_ms": 2500}
```

#### Listar Conversas
```http
GET /api/v1/agent/conversations?config_id=1&limit=20&offset=0
Authorization: Bearer {jwt_token}
```

#### Histórico da Conversa
```http
GET /api/v1/agent/conversations/{thread_id}
Authorization: Bearer {jwt_token}
```

#### Limpar Conversa
```http
DELETE /api/v1/agent/conversations/{thread_id}
Authorization: Bearer {jwt_token}
```

#### Enviar Feedback
```http
POST /api/v1/agent/feedback
Authorization: Bearer {jwt_token}

{
    "message_id": 123,
    "rating": 5,
    "feedback_text": "Resposta muito útil!"
}
```

#### Sugestões de Perguntas
```http
GET /api/v1/agent/suggestions/{config_id}
Authorization: Bearer {jwt_token}
```

#### Status do Agente
```http
GET /api/v1/agent/status
```

### 6.2 Endpoints de Health Check

```http
GET /api/v1/health           # Check básico (sem auth)
GET /api/v1/health/detailed  # Check detalhado com dependências
```

### 6.3 Códigos de Status

| Código | Significado |
|--------|-------------|
| 200 | Sucesso |
| 201 | Criado com sucesso |
| 400 | Requisição inválida |
| 401 | Não autorizado |
| 403 | Proibido |
| 404 | Não encontrado |
| 429 | Rate limit excedido |
| 500 | Erro interno |

---

## 7. Modelos de Machine Learning

### 7.1 Classificação de Campanhas (XGBoost)

#### Objetivo
Classificar campanhas em tiers de performance para identificar quais escalar, otimizar ou pausar.

#### Features de Entrada (10)

| Feature | Descrição | Fórmula |
|---------|-----------|---------|
| `cpl_ratio` | CPL relativo à média | `campaign_cpl / account_avg_cpl` |
| `ctr_ratio` | CTR relativo à média | `campaign_ctr / account_avg_ctr` |
| `leads_7d_normalized` | Leads normalizados | `leads_7d / max_leads_7d` |
| `cpl_trend` | Tendência do CPL | `(cpl_7d - cpl_30d) / cpl_30d` |
| `leads_trend` | Tendência de leads | `(leads_7d - leads_30d) / leads_30d` |
| `cpl_volatility` | Volatilidade do CPL | `std(cpl_daily) / avg(cpl_daily)` |
| `conversion_rate_7d` | Taxa de conversão | `leads_7d / clicks_7d` |
| `days_with_leads_ratio` | Dias com leads | `days_with_leads / 7` |
| `frequency_score` | Score de frequência | `1 - (frequency - 1) / 5` |
| `consistency_score` | Score de consistência | `1 - cpl_volatility` |

#### Tiers de Saída

| Tier | Critérios | Ação Recomendada |
|------|-----------|------------------|
| **HIGH_PERFORMER** | CPL baixo, leads consistentes, tendência positiva | Escalar budget |
| **MODERATE** | Performance na média, estável | Otimizar criativos |
| **LOW** | Performance abaixo da média | Investigar e ajustar |
| **UNDERPERFORMER** | CPL alto, poucos leads, tendência negativa | Pausar ou reestruturar |

### 7.2 Detecção de Anomalias (Isolation Forest)

#### Tipos de Anomalias Detectadas

| Tipo | Descrição | Severidade |
|------|-----------|------------|
| `CPL_HIGH` | CPL > 1.5x média histórica | MEDIUM |
| `CPL_VERY_HIGH` | CPL > 2x média histórica | HIGH |
| `ZERO_LEADS` | 0 leads em 3+ dias consecutivos | HIGH |
| `FREQUENCY_HIGH` | Frequência > 5 | MEDIUM |
| `SPEND_ZERO` | Gasto = 0 com campanha ativa | CRITICAL |
| `PERFORMANCE_DROP` | Queda > 50% em performance | HIGH |

### 7.3 Engine de Recomendações

O sistema de recomendações utiliza regras de negócio para gerar ações específicas:

| Tipo | Trigger | Recomendação |
|------|---------|--------------|
| `BUDGET_INCREASE` | HIGH_PERFORMER com ROI > 200% | Aumentar budget em 50% |
| `BUDGET_DECREASE` | LOW com CPL > 2x média | Reduzir budget em 30% |
| `PAUSE_CAMPAIGN` | UNDERPERFORMER + sem leads 7 dias | Pausar campanha |
| `REFRESH_CREATIVE` | Frequência > 4 | Renovar criativos |
| `AUDIENCE_REVIEW` | CTR < 0.5% | Revisar segmentação |
| `SCALE_CAMPAIGN` | HIGH_PERFORMER consistente | Escalar gradualmente |

### 7.4 Previsões de Séries Temporais

#### Métodos Suportados

| Método | Uso | Requisitos |
|--------|-----|------------|
| **ARIMA** | Previsão de CPL e leads | 14+ dias de dados |
| **Moving Average** | Fallback simples | 7+ dias de dados |
| **Prophet** | Sazonalidade (opcional) | 30+ dias de dados |

#### Métricas Previstas

- **CPL**: Custo por Lead (7-30 dias)
- **Leads**: Volume de leads (7-30 dias)
- **Spend**: Gasto projetado (7-30 dias)

---

## 8. Sistema de Persistência

### 8.1 Tabelas do Agente

#### agent_conversations
```sql
CREATE TABLE agent_conversations (
    id SERIAL PRIMARY KEY,
    thread_id VARCHAR(255) UNIQUE NOT NULL,
    config_id INTEGER NOT NULL,
    user_id INTEGER NOT NULL,
    title VARCHAR(255),
    message_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);
```

#### agent_messages
```sql
CREATE TABLE agent_messages (
    id SERIAL PRIMARY KEY,
    conversation_id INTEGER REFERENCES agent_conversations(id) ON DELETE CASCADE,
    role VARCHAR(20) NOT NULL,  -- user, assistant, tool
    content TEXT NOT NULL,
    tool_calls JSONB,
    tool_results JSONB,
    tokens_used INTEGER,
    created_at TIMESTAMP DEFAULT NOW()
);
```

#### agent_checkpoints
```sql
CREATE TABLE agent_checkpoints (
    thread_id VARCHAR(255),
    thread_ts TIMESTAMP,
    checkpoint BYTEA NOT NULL,
    metadata JSONB,
    PRIMARY KEY (thread_id, thread_ts)
);
```

#### agent_feedback
```sql
CREATE TABLE agent_feedback (
    id SERIAL PRIMARY KEY,
    message_id INTEGER REFERENCES agent_messages(id) ON DELETE CASCADE UNIQUE,
    user_id INTEGER NOT NULL,
    rating INTEGER CHECK (rating >= 1 AND rating <= 5),
    feedback_text TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);
```

### 8.2 Tabelas de ML

| Tabela | Descrição |
|--------|-----------|
| `ml_trained_models` | Registro de modelos treinados |
| `ml_features` | Features extraídas (normalizadas) |
| `ml_campaign_classifications` | Classificações de campanhas |
| `ml_recommendations` | Recomendações geradas |
| `ml_anomalies` | Anomalias detectadas |
| `ml_forecasts` | Previsões de séries temporais |
| `ml_training_jobs` | Jobs de treinamento |
| `ml_predictions` | Previsões individuais |

### 8.3 Acesso Read-Only ao FamaChat

O microserviço acessa tabelas do FamaChat principal em modo **somente leitura**:

- `sistema_facebook_ads_configs`
- `sistema_facebook_ads_campaigns`
- `sistema_facebook_ads_adsets`
- `sistema_facebook_ads_ads`
- `sistema_facebook_ads_insights_history`
- `sistema_facebook_ads_insights_today`

---

## 9. Tarefas Agendadas (Celery)

### 9.1 Arquitetura Celery

```
┌─────────────────┐      ┌─────────────┐      ┌─────────────────┐
│  Celery Beat    │─────▶│    Redis    │◀─────│  Celery Worker  │
│  (Scheduler)    │      │   (Broker)  │      │  (Executor)     │
└─────────────────┘      └─────────────┘      └─────────────────┘
```

### 9.2 Jobs Agendados

| Job | Horário | Fila | Descrição |
|-----|---------|------|-----------|
| `daily_pipeline` | 02:00 | ml | Pipeline completo: features → classificação → recomendações → previsões |
| `daily_model_retraining` | 05:00 | training | Retreina modelos com dados novos |
| `daily_classification` | 06:00 | ml | Reclassifica todas as campanhas |
| `daily_recommendations` | 07:00 | ml | Gera novas recomendações |
| `validate_predictions` | 08:00 | ml | Valida previsões anteriores |
| `hourly_anomaly_detection` | *:30 | ml | Detecta anomalias a cada hora |
| `batch_predictions` | */4h | ml | Gera previsões em batch |

### 9.3 Filas Celery

| Fila | Propósito | Concorrência |
|------|-----------|--------------|
| `default` | Tasks gerais | 2 |
| `training` | Treinamento de modelos | 1 |
| `ml` | Processamento ML | 2 |

### 9.4 Monitoramento com Flower

Acesse o dashboard Flower em: `http://localhost:5555`

- Visualização de tasks em tempo real
- Histórico de execuções
- Métricas de performance
- Status dos workers

---

## 10. Configuração e Deploy

### 10.1 Variáveis de Ambiente

```env
# ==================== Database ====================
DATABASE_URL=postgresql://user:pass@host:5432/famachat

# ==================== Redis ====================
REDIS_URL=redis://localhost:6380/0

# ==================== LLM ====================
AGENT_LLM_PROVIDER=anthropic  # ou openai
AGENT_ANTHROPIC_API_KEY=sk-ant-...
AGENT_OPENAI_API_KEY=sk-proj-...
AGENT_LLM_MODEL=claude-sonnet-4-20250514
AGENT_TEMPERATURE=0.3
AGENT_MAX_TOKENS=4096

# ==================== Segurança ====================
ML_API_KEY=sua-api-key-secreta
JWT_SECRET=seu-jwt-secret

# ==================== Performance ====================
AGENT_TIMEOUT_SECONDS=60
AGENT_MAX_TOOL_CALLS=10
AGENT_RATE_LIMIT_PER_MINUTE=20
AGENT_RATE_LIMIT_PER_DAY=500

# ==================== Persistência ====================
AGENT_CHECKPOINT_ENABLED=true
AGENT_CONVERSATION_TTL_DAYS=30

# ==================== ML Thresholds ====================
THRESHOLD_CPL_LOW=0.7
THRESHOLD_CPL_HIGH=1.3
THRESHOLD_CTR_GOOD=1.2
THRESHOLD_FREQUENCY_HIGH=3.0

# ==================== Logging ====================
LOG_LEVEL=INFO
ENVIRONMENT=production
```

### 10.2 Docker Compose

O projeto inclui 5 serviços Docker:

| Serviço | Porta | Recursos |
|---------|-------|----------|
| **famachat-ml-api** | 8000 | 0.5-2 CPU, 512MB-2GB RAM |
| **famachat-ml-worker** | - | 0.5-2 CPU, 1-3GB RAM |
| **famachat-ml-beat** | - | 0.25 CPU, 256MB RAM |
| **famachat-ml-redis** | 6380 | 0.5 CPU, 512MB RAM |
| **famachat-ml-flower** | 5555 | 0.25 CPU, 256MB RAM |

### 10.3 Comandos de Deploy

```bash
# Build da imagem
docker-compose build

# Iniciar serviços
docker-compose up -d

# Ver logs
docker-compose logs -f famachat-ml-api

# Reiniciar worker
docker-compose restart famachat-ml-worker

# Parar tudo
docker-compose down
```

### 10.4 Desenvolvimento Local

```bash
# Instalar dependências
pip install -r requirements.txt

# Rodar migrations
alembic upgrade head

# Iniciar API
uvicorn app.main:app --reload --port 8000

# Iniciar worker (em outro terminal)
celery -A app.tasks.celery_app worker --loglevel=info

# Iniciar beat (em outro terminal)
celery -A app.tasks.celery_app beat --loglevel=info
```

---

## 11. Monitoramento e Observabilidade

### 11.1 Logging (structlog)

Todos os logs são estruturados em JSON:

```json
{
    "timestamp": "2026-01-19T10:30:00.000Z",
    "level": "info",
    "logger": "app.agent.service",
    "message": "Chat processado",
    "thread_id": "550e8400-e29b-41d4-a716-446655440000",
    "user_id": 123,
    "config_id": 1,
    "intent": "analyze",
    "tool_calls_count": 3,
    "duration_ms": 2500
}
```

### 11.2 Health Checks

**Endpoint Básico:**
```json
GET /api/v1/health
{
    "status": "healthy",
    "timestamp": "2026-01-19T10:30:00Z"
}
```

**Endpoint Detalhado:**
```json
GET /api/v1/health/detailed
{
    "status": "healthy",
    "database": {"status": "connected", "latency_ms": 5},
    "redis": {"status": "connected", "latency_ms": 2},
    "celery": {"status": "online", "workers": 2},
    "llm": {"provider": "anthropic", "model": "claude-sonnet-4"}
}
```

### 11.3 Métricas do Agente

```http
GET /api/v1/agent/status
{
    "status": "online",
    "llm_provider": "anthropic",
    "model": "claude-sonnet-4-20250514",
    "version": "1.0.0"
}
```

---

## 12. Integração com FamaChat Principal

### 12.1 Fluxo de Comunicação

```
┌────────────────────┐
│   FamaChat Web     │
│   (React Frontend) │
└─────────┬──────────┘
          │ HTTP/REST
          ▼
┌────────────────────┐
│  FamaChat Backend  │
│   (Node.js/Express)│
│                    │
│   ┌────────────┐   │
│   │ JWT Auth   │   │
│   └────────────┘   │
└─────────┬──────────┘
          │ HTTP/REST + JWT
          ▼
┌────────────────────┐
│   FamaChat ML      │
│   (Python/FastAPI) │
│                    │
│   ┌────────────┐   │
│   │ Valida JWT │   │
│   └────────────┘   │
└────────────────────┘
```

### 12.2 Autenticação

O token JWT é gerado pelo FamaChat principal e validado pelo ML:

```python
# Validação do JWT
@router.post("/chat")
async def chat(
    request: ChatRequest,
    current_user: dict = Depends(get_current_user),
):
    # current_user contém: id, role, config_id
    ...
```

### 12.3 Banco de Dados Compartilhado

Ambos os serviços acessam o mesmo PostgreSQL:

- FamaChat principal: **leitura/escrita** em suas tabelas
- FamaChat ML: **somente leitura** em tabelas do principal + **leitura/escrita** em tabelas de ML

---

## 13. Segurança

### 13.1 Autenticação

| Método | Uso | Header |
|--------|-----|--------|
| **JWT** | Endpoints do agente | `Authorization: Bearer {token}` |
| **API Key** | Endpoints ML internos | `X-API-Key: {key}` |

### 13.2 Rate Limiting

| Limite | Valor | Escopo |
|--------|-------|--------|
| Por minuto | 20 requisições | Por usuário |
| Por dia | 500 requisições | Por usuário |

### 13.3 Proteções Implementadas

- **CORS**: Restrito a domínios autorizados
- **Input Validation**: Pydantic em todas as entradas
- **SQL Injection**: Prevenido pelo SQLAlchemy ORM
- **Secrets**: Variáveis de ambiente, nunca hardcoded
- **Token Encryption**: JWT com secret forte

---

## 14. Guia de Uso

### 14.1 Exemplos de Perguntas

#### Análise de Performance
```
"Como está a performance das minhas campanhas?"
"Qual campanha está com o melhor CPL?"
"Mostre um resumo da conta dos últimos 7 dias"
```

#### Identificação de Problemas
```
"Tem alguma anomalia crítica?"
"Qual campanha devo pausar?"
"Por que minha campanha X não está performando?"
```

#### Comparações
```
"Compare a campanha A com a campanha B"
"Compare minhas top 3 campanhas"
"Qual campanha tem mais leads?"
```

#### Previsões
```
"Previsão de CPL para a próxima semana"
"Quantos leads devo esperar nos próximos 7 dias?"
"Qual a tendência do meu CPL?"
```

#### Recomendações
```
"O que você recomenda que eu faça?"
"Quais campanhas devo escalar?"
"Próximos passos para melhorar os resultados"
```

### 14.2 Exemplo de Conversa

**Usuário:** Qual campanha devo escalar?

**Agente:**
```
📊 Analisei suas campanhas e recomendo escalar a campanha "Leads Apartamentos Centro":

📈 Métricas Atuais:
• CPL: R$ 28,50 (43% abaixo da média da conta)
• Leads 7 dias: 45 leads
• Investimento: R$ 1.282,50
• CTR: 1.8% (acima da média)
• Classificação: HIGH_PERFORMER (confiança: 92%)

✅ Por que escalar:
• Performance consistente nos últimos 30 dias
• CPL abaixo da média com volume alto de leads
• CTR saudável indica boa aceitação do público
• Sem anomalias detectadas

💡 Recomendação:
Aumente o budget em 30-50% gradualmente (a cada 3 dias) para evitar
perda de otimização do algoritmo do Facebook.

⚠️ Atenção:
Monitore a frequência após escalar. Se passar de 4, considere
renovar os criativos.
```

---

## 15. Troubleshooting

### 15.1 Problemas Comuns

#### Erro: "API Key inválida"
```
Causa: ANTHROPIC_API_KEY ou OPENAI_API_KEY não configurada
Solução: Verificar variável de ambiente no .env
```

#### Erro: "Timeout ao processar mensagem"
```
Causa: LLM demorando muito para responder
Solução: Aumentar AGENT_TIMEOUT_SECONDS ou usar modelo mais rápido
```

#### Erro: "Sem dados para análise"
```
Causa: Conta sem dados de insights ou campanhas inativas
Solução: Verificar se há dados no Facebook Ads e sincronização ativa
```

#### Erro: "Rate limit excedido"
```
Causa: Muitas requisições em curto período
Solução: Aguardar ou aumentar AGENT_RATE_LIMIT_PER_MINUTE
```

### 15.2 Logs Úteis

```bash
# Ver logs da API
docker-compose logs -f famachat-ml-api

# Ver logs do worker
docker-compose logs -f famachat-ml-worker

# Ver erros específicos
grep "ERROR" logs/famachat-ml.log

# Ver requisições lentas
grep "duration_ms" logs/famachat-ml.log | jq 'select(.duration_ms > 5000)'
```

### 15.3 Comandos de Debug

```bash
# Testar conexão com banco
python -c "from app.db.session import sync_engine; print(sync_engine.connect())"

# Testar Redis
redis-cli -p 6380 ping

# Testar LLM
curl http://localhost:8000/api/v1/agent/status

# Forçar reprocessamento
celery -A app.tasks.celery_app call app.tasks.scheduled_tasks.daily_pipeline
```

---

## Apêndice A: Referências Rápidas

### Endpoints Principais

| Endpoint | Método | Descrição |
|----------|--------|-----------|
| `/api/v1/agent/chat` | POST | Chat completo |
| `/api/v1/agent/chat/stream` | POST | Chat com streaming |
| `/api/v1/agent/conversations` | GET | Listar conversas |
| `/api/v1/agent/suggestions/{id}` | GET | Sugestões |
| `/api/v1/health` | GET | Health check |

### Tiers de Classificação

| Tier | Ação |
|------|------|
| HIGH_PERFORMER | Escalar |
| MODERATE | Otimizar |
| LOW | Investigar |
| UNDERPERFORMER | Pausar |

### Severidade de Anomalias

| Severidade | Urgência |
|------------|----------|
| CRITICAL | Ação imediata |
| HIGH | Ação em 24h |
| MEDIUM | Monitorar |
| LOW | Informativo |

---

## Apêndice B: Changelog

### v1.0.0 (Janeiro 2026)
- Implementação inicial do agente LangGraph
- 20 tools para análise de campanhas
- Integração com Claude e GPT
- Sistema de classificação XGBoost
- Detecção de anomalias Isolation Forest
- Previsões com ARIMA
- API REST completa
- Celery para jobs em background
- Persistência com PostgreSQL checkpointer

---

**Documento mantido pela equipe FamaChat**
**Contato:** suporte@famachat.com
