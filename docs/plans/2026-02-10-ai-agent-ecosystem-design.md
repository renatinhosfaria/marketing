# Design: Ecossistema Multi-Agente de IA para Facebook Ads

**Data:** 2026-02-10
**Status:** Aprovado (design)
**Autor:** Brainstorming colaborativo (humano + IA)

---

## Indice

1. [Visao Geral da Arquitetura](#1-visao-geral-da-arquitetura)
2. [Design de Estado e Memoria](#2-design-de-estado-e-memoria)
3. [Tools e Integracao com APIs](#3-tools-e-integracao-com-apis)
4. [Streaming, Frontend e UX](#4-streaming-frontend-e-ux)
5. [Estrutura de Arquivos e Organizacao do Codigo](#5-estrutura-de-arquivos-e-organizacao-do-codigo)
6. [Resiliencia, Testes e Observabilidade](#6-resiliencia-testes-e-observabilidade)

---

## 1. Visao Geral da Arquitetura

### Nome do Sistema

**FamaChat AI Agent** — Ecossistema multi-agente especializado em trafego pago Facebook Ads.

### Padrao Arquitetural

Supervisor com Subgraphs Hierarquicos + Command pattern para handoffs dinamicos + Send API para fan-out paralelo.

### Fluxo Principal

```
Usuario (Frontend Next.js)
    | POST /api/v1/agent/chat
    v
SSE Stream <---- Agent API (FastAPI, porta 8002)
    |                |
    |                v
    |         +-------------+
    |         |  Supervisor  | <- Haiku 3.5 (Structured Output)
    |         |  (Router)    |
    |         +------+-------+
    |                | Send() fan-out paralelo
    |         +------+----------+----------+
    |         v      v          v          v
    |    +--------+ +--------+ +--------+ +--------+
    |    | Health | | Perfor.| | Creat. | | Forec. |  ...
    |    |Monitor | | Analyst| | Spec.  | | Scient.|
    |    +----+---+ +----+---+ +----+---+ +----+---+
    |         +----------+---------+----------+
    |                    v Fan-in (reducer)
    |            +---------------+
    |            |  Synthesizer  | <- Sonnet 4.5
    |            |  (Resposta)   |
    |            +---------------+
    v
Resposta final + Cards de atividade dos agentes
```

### 6 Agentes Especializados

| Agente | Tipo | Modelo Default | Subgraph? | Funcao |
|--------|------|---------------|-----------|--------|
| Monitor de Saude & Anomalias | Read-only | Sonnet | Sim (3 nos internos) | Detecta anomalias (IsolationForest/Z-score/IQR) e classifica saude das campanhas |
| Analista de Performance & Impacto | Read-only | Sonnet | Sim | Analisa metricas, compara periodos, mede impacto causal de mudancas |
| Especialista em Criativos | Read-only | Sonnet | Sim | Analisa fadiga criativa, compara anuncios, recomenda creative refresh |
| Especialista em Audiencias | Read-only | Sonnet | Nao (no simples) | Analisa segmentacao, saturacao de publico, performance por audiencia |
| Cientista de Previsao | Read-only | Sonnet | Nao (no simples) | Gera previsoes (Prophet + Ensemble) de CPL, Leads, Spend |
| Gerente de Operacoes | Write (acoes) | Sonnet | Sim (interrupt obrigatorio) | Executa acoes: altera budgets, pausa campanhas, aplica recomendacoes |

### Comunicacao

SSE com `stream_mode=["messages", "updates", "custom"]` e `subgraphs=True`.

### Decisoes Arquiteturais

| Decisao | Escolha | Motivo |
|---------|---------|--------|
| Padrao multi-agente | Supervisor + Subgraphs + Command | Isolamento de complexidade ML, seguranca em operacoes de escrita, handoffs ageis |
| Roteamento do Supervisor | Classificacao Estruturada + Send API (fan-out) | Paralelismo real, determinismo de negocio, eficiencia de tokens |
| Protocolo frontend | SSE (Server-Sent Events) | Nativo do LangGraph, compativel com Next.js, simplifica interrupts |
| Estrategia de LLM | Configuravel por agente (Opção C) com defaults tiered (Opcao B) | Haiku para roteamento rapido, Sonnet para analise, flexivel via config |
| Provider de LLM | Multi-provider (Anthropic principal + OpenAI fallback) | Resiliencia operacional com `with_fallbacks()` |
| Integracao de tools | Hibrida: ML via HTTP, FB Ads/DB via import direto | ML isolado (CPU-heavy), I/O-bound com baixa latencia |
| Memoria de longo prazo | PostgresStore + pgvector com embeddings | Busca semantica desde o dia 1, evita migracao futura |

---

## 2. Design de Estado e Memoria

### 2.1 — State Schemas

#### SupervisorState (Global)

O estado compartilhado do grafo principal. Usa `add_messages` como reducer para acumular
historico e `operator.add` para fan-in dos reports dos agentes.

```python
from typing import Annotated, List, Optional, Literal
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langgraph.managed import RemainingSteps
from langchain_core.messages import AnyMessage
import operator


class AgentReport(TypedDict):
    agent_id: str
    status: Literal["running", "completed", "error"]
    summary: str              # Resumo textual para o Synthesizer
    data: Optional[dict]      # Dados estruturados (metricas, scores)
    confidence: float         # 0.0 - 1.0


class UserContext(TypedDict):
    user_id: str
    account_id: str
    account_name: str
    timezone: str             # Default: America/Sao_Paulo


class SupervisorState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    user_context: UserContext
    routing_decision: Optional[dict]
    agent_reports: Annotated[List[AgentReport], operator.add]  # Fan-in via reducer
    pending_actions: List[dict]
    synthesis: Optional[str]
    remaining_steps: RemainingSteps  # Controle de recursao
```

**Principio-chave:** O `agent_reports` usa `operator.add` como reducer. Quando multiplos agentes
completam em paralelo, seus reports sao **acumulados** automaticamente sem sobrescrever.

#### Subgraph States (Privados)

Dados brutos de metricas (dataframes) **nunca** entram no State. Sao carregados dentro do no,
processados, e apenas agregacoes/referencias sao persistidas no checkpoint.

```python
class HealthSubgraphState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    metrics_ref: Optional[str]          # Referencia a dados em cache (nao o dataframe)
    anomaly_results: Optional[dict]     # Scores e anomalias detectadas
    classifications: Optional[dict]     # Tiers das entidades analisadas
    diagnosis: Optional[str]            # Diagnostico final (sobe ao pai)


class OperationsSubgraphState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    proposed_action: Optional[dict]     # Acao proposta com dry_run
    approval_status: Literal["pending", "approved", "rejected", None]
    execution_result: Optional[dict]    # Resultado da API do Facebook


class CreativeSubgraphState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    ad_creatives: List[dict]            # Metadados dos anuncios
    fatigue_analysis: Optional[dict]    # Resultado da analise de fadiga
    preview_urls: List[str]             # URLs de preview para o frontend
    recommendation: Optional[str]       # Recomendacao final


class PerformanceSubgraphState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    metrics_data: Optional[dict]        # Metricas agregadas
    comparison: Optional[dict]          # Resultado de comparacao entre periodos
    impact_analysis: Optional[dict]     # Resultado de analise causal
    report: Optional[str]               # Relatorio final
```

#### Input/Output Schemas (Contratos entre Supervisor e Especialistas)

```python
class HealthInput(TypedDict):
    messages: List[AnyMessage]

class HealthOutput(TypedDict):
    agent_reports: List[AgentReport]

# Na compilacao do Subgraph:
health_app = workflow.compile(
    input_schema=HealthInput,
    output_schema=HealthOutput,
)
```

### 2.2 — Memoria de Longo Prazo (Store)

**Infraestrutura:** PostgresStore + pgvector com embeddings `text-embedding-3-small` (1536 dims).

```python
from langgraph.store.postgres import PostgresStore
from langchain.embeddings import init_embeddings

store = PostgresStore.from_conn_string(
    DATABASE_URL,
    index={
        "dims": 1536,
        "embed": init_embeddings("openai:text-embedding-3-small"),
        "fields": ["insight_text", "context"],
    }
)
```

#### Namespace Hierarquico

```
(user_id, "profile")                               -> Preferencias globais do usuario
(user_id, account_id, "patterns")                   -> Padroes da conta de ads
(user_id, account_id, "action_history")             -> Historico de acoes executadas
(user_id, account_id, campaign_id, "insights")      -> Insights granulares por campanha
```

#### Exemplos de Memorias Persistidas

```python
# O Analista descobre um padrao -> salva via tool save_insight
store.put(
    ("user_123", "acc_456", "patterns"),
    key="weekend_cpl_spike",
    value={
        "insight_text": "CPL aumenta 35% aos sabados para campanhas de conversao",
        "context": "Padrao observado em 4 dos ultimos 6 fins de semana",
        "confidence": 0.85,
        "discovered_by": "performance_analyst",
        "discovered_at": "2026-02-10",
    }
)

# O Forecaster busca padroes relevantes antes de projetar
relevant = store.search(
    ("user_123", "acc_456", "patterns"),
    query="previsao de custo para o fim de semana",
    limit=5,
)
```

#### Acesso no LangGraph

Agentes acessam o Store via `Runtime` injetado automaticamente:

```python
from langgraph.runtime import Runtime

async def analyst_node(state: State, runtime: Runtime[AgentConfig]):
    user_id = runtime.context.user_id
    memories = await runtime.store.asearch(
        (user_id, state["user_context"]["account_id"], "patterns"),
        query=state["messages"][-1].content,
        limit=5,
    )
    # Incorpora memorias no prompt do agente
```

### 2.3 — Checkpointer (Memoria de Curto Prazo)

**Producao:** `AsyncPostgresSaver` com a mesma instancia PostgreSQL do projeto.

```python
from langgraph.checkpoint.postgres import AsyncPostgresSaver

checkpointer = AsyncPostgresSaver.from_conn_string(DATABASE_URL)
await checkpointer.asetup()  # Cria tabelas na inicializacao

store = PostgresStore.from_conn_string(DATABASE_URL, index={...})
await store.asetup()  # Cria tabelas + indice pgvector

graph = supervisor_builder.compile(
    checkpointer=checkpointer,
    store=store,
)
```

**Thread management:**
- Cada conversa = 1 `thread_id` unico
- `get_state_history()` para debug e time-travel
- `update_state()` para o usuario corrigir acoes propostas antes de executar

---

## 3. Tools e Integracao com APIs

### 3.1 — Estrategia de Integracao (Hibrida)

```
+-----------------------------------------------------+
|                   Agent Runtime                      |
|                                                      |
|  +--Tools ML--+    HTTP (async)    +-----------+    |
|  | (read-only)|  ----------------> |  ML API   |    |
|  +------------+  httpx.AsyncClient |  :8001    |    |
|                                    +-----------+    |
|                                                      |
|  +--Tools FB--+    Import direto   +-----------+    |
|  | (read+write)  ----------------> | FB Ads    |    |
|  +------------+  services/client   | Services  |    |
|                                    +-----------+    |
|                                                      |
|  +--Tools DB--+    SQLAlchemy      +-----------+    |
|  | (read-only)|  ----------------> | PostgreSQL|    |
|  +------------+  async session     |           |    |
|                                    +-----------+    |
|                                                      |
|  +--Tools Mem-+    Runtime.store   +-----------+    |
|  | (memoria)  |  ----------------> | Postgres  |    |
|  +------------+  (injetado)        | Store     |    |
|                                    +-----------+    |
+-----------------------------------------------------+
```

**Regra:** ML (CPU-heavy) via HTTP. Facebook Ads e DB (I/O-bound) via import direto. Memoria via Store injetado pelo Runtime.

### 3.2 — Tools Compartilhadas (Disponíveis a Todos os Agentes)

```python
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt import InjectedStore
from langgraph.store.base import BaseStore
from typing import Annotated, Literal
from uuid import uuid4


@tool
async def save_insight(
    insight_text: str,
    context: str,
    category: Literal["performance_pattern", "audience_preference", "creative_learning"],
    store: Annotated[BaseStore, InjectedStore],
    config: RunnableConfig,
) -> str:
    """Salva um insight descoberto para memoria de longo prazo."""
    user_id = config["configurable"]["user_id"]
    account_id = config["configurable"]["account_id"]
    await store.aput(
        (user_id, account_id, "insights", category),
        key=str(uuid4()),
        value={
            "insight_text": insight_text,
            "context": context,
            "discovered_at": "now()",
        },
    )
    return f"Insight salvo: {insight_text[:80]}..."


@tool
async def recall_insights(
    query: str,
    store: Annotated[BaseStore, InjectedStore],
    config: RunnableConfig,
) -> list:
    """Busca insights relevantes na memoria de longo prazo (busca semantica)."""
    user_id = config["configurable"]["user_id"]
    account_id = config["configurable"]["account_id"]
    results = await store.asearch(
        (user_id, account_id, "patterns"),
        query=query,
        limit=5,
    )
    return [item.value for item in results]
```

### 3.3 — Tools por Agente

#### 1. Monitor de Saude & Anomalias (ML via HTTP)

```python
@tool
async def detect_anomalies(
    entity_type: Literal["campaign", "adset", "ad"],
    entity_ids: List[str],
    metrics: List[str] = ["cpl", "ctr", "cpc", "frequency"],
    config: RunnableConfig = None,
) -> dict:
    """Executa deteccao de anomalias (IsolationForest + Z-score + IQR)."""
    account_id = config.get("configurable", {}).get("account_id")
    async with httpx.AsyncClient(base_url=ML_API_URL, timeout=30.0) as client:
        resp = await client.post("/api/v1/anomalies", json={
            "entity_type": entity_type,
            "entity_ids": entity_ids,
            "metrics": metrics,
            "account_id": account_id,
        })
        resp.raise_for_status()
        return resp.json()


@tool
async def get_classifications(
    entity_type: Literal["campaign", "adset", "ad"],
    config: RunnableConfig = None,
) -> dict:
    """Busca classificacoes atuais (HIGH_PERFORMER, MODERATE, LOW, UNDERPERFORMER)."""
    account_id = config.get("configurable", {}).get("account_id")
    async with httpx.AsyncClient(base_url=ML_API_URL, timeout=10.0) as client:
        resp = await client.get(
            "/api/v1/classifications",
            params={"account_id": account_id, "entity_type": entity_type},
        )
        resp.raise_for_status()
        return resp.json()


@tool
async def classify_entity(
    entity_type: Literal["campaign", "adset", "ad"],
    entity_id: str,
    config: RunnableConfig = None,
) -> dict:
    """Classifica uma entidade especifica em tempo real."""
    async with httpx.AsyncClient(base_url=ML_API_URL, timeout=15.0) as client:
        resp = await client.post("/api/v1/classifications", json={
            "entity_type": entity_type,
            "entity_id": entity_id,
        })
        resp.raise_for_status()
        return resp.json()


@tool
async def get_anomaly_history(
    entity_id: str,
    days: int = 30,
    config: RunnableConfig = None,
) -> list:
    """Retorna historico de anomalias detectadas nos ultimos N dias."""
    async with httpx.AsyncClient(base_url=ML_API_URL, timeout=10.0) as client:
        resp = await client.get(
            "/api/v1/anomalies",
            params={"entity_id": entity_id, "days": days},
        )
        resp.raise_for_status()
        return resp.json()
```

#### 2. Analista de Performance & Impacto (DB direto + ML via HTTP)

```python
@tool
async def get_campaign_insights(
    entity_type: Literal["campaign", "adset", "ad"],
    entity_id: str,
    date_start: str,
    date_end: str,
    config: RunnableConfig = None,
) -> list:
    """Busca metricas detalhadas (spend, leads, CPL, CTR, CPC, impressions) por periodo."""
    account_id = config.get("configurable", {}).get("account_id")
    query = """
        SELECT date, spend, leads, cpl, ctr, cpc, impressions
        FROM facebook_ads_insights_history
        WHERE entity_id = $1 AND account_id = $2
          AND date BETWEEN $3 AND $4
        ORDER BY date ASC
    """
    async with db_pool.acquire() as conn:
        rows = await conn.fetch(query, entity_id, account_id, date_start, date_end)
        return [dict(row) for row in rows]


@tool
async def compare_periods(
    entity_id: str,
    period_a_start: str,
    period_a_end: str,
    period_b_start: str,
    period_b_end: str,
    metrics: List[str] = ["spend", "leads", "cpl", "ctr"],
    config: RunnableConfig = None,
) -> dict:
    """Compara metricas entre dois periodos (ex: semana atual vs anterior)."""
    # DB direto -> query comparativa com aggregations
    pass  # Implementacao busca ambos periodos e calcula diffs


@tool
async def analyze_causal_impact(
    entity_id: str,
    intervention_date: str,
    metric: str = "cpl",
    config: RunnableConfig = None,
) -> dict:
    """Analisa impacto causal de uma mudanca na campanha."""
    async with httpx.AsyncClient(base_url=ML_API_URL, timeout=30.0) as client:
        resp = await client.post("/api/v1/impact", json={
            "entity_id": entity_id,
            "intervention_date": intervention_date,
            "metric": metric,
        })
        resp.raise_for_status()
        return resp.json()


@tool
async def get_insights_summary(config: RunnableConfig = None) -> dict:
    """Resumo de KPIs agregados da conta (total spend, leads, CPL medio)."""
    account_id = config.get("configurable", {}).get("account_id")
    # DB direto -> aggregation query
    pass
```

#### 3. Especialista em Criativos (DB + Facebook API)

```python
@tool
async def get_ad_creatives(
    campaign_id: Optional[str] = None,
    adset_id: Optional[str] = None,
    config: RunnableConfig = None,
) -> list:
    """Lista anuncios com metadados (formato, copy, thumbnail URL)."""
    # DB direto -> facebook_ads_ads + joins
    pass


@tool
async def detect_creative_fatigue(
    ad_ids: List[str],
    window_days: int = 14,
    config: RunnableConfig = None,
) -> dict:
    """Detecta fadiga criativa: queda de CTR + aumento de frequency ao longo do tempo."""
    # DB direto -> analise de tendencia em insights diarios
    pass


@tool
async def compare_creatives(
    ad_ids: List[str],
    metric: str = "ctr",
    config: RunnableConfig = None,
) -> dict:
    """Compara performance entre criativos do mesmo adset/campanha."""
    # DB direto -> query comparativa
    pass


@tool
async def get_ad_preview_url(
    ad_id: str,
    config: RunnableConfig = None,
) -> str:
    """Retorna URL de preview do anuncio para renderizar no frontend."""
    # Facebook API direto -> ad preview endpoint
    pass
```

#### 4. Especialista em Audiencias (DB)

```python
@tool
async def get_adset_audiences(
    campaign_id: Optional[str] = None,
    config: RunnableConfig = None,
) -> list:
    """Dados de segmentacao dos adsets (targeting, idade, genero, interesses)."""
    pass


@tool
async def detect_audience_saturation(
    adset_ids: List[str],
    window_days: int = 14,
    config: RunnableConfig = None,
) -> dict:
    """Analisa saturacao: frequency crescente + CTR decrescente = publico esgotado."""
    pass


@tool
async def get_audience_performance(
    adset_ids: List[str],
    config: RunnableConfig = None,
) -> list:
    """Performance por audiencia: CPL, CTR, Leads por adset."""
    pass
```

#### 5. Cientista de Previsao (ML via HTTP)

```python
@tool
async def generate_forecast(
    entity_id: str,
    metric: Literal["cpl", "leads", "spend"],
    horizon_days: int = 7,
    config: RunnableConfig = None,
) -> dict:
    """Gera previsao (Prophet + Ensemble) para N dias."""
    async with httpx.AsyncClient(base_url=ML_API_URL, timeout=30.0) as client:
        resp = await client.post("/api/v1/predictions", json={
            "entity_id": entity_id,
            "metric": metric,
            "horizon_days": horizon_days,
        })
        resp.raise_for_status()
        return resp.json()


@tool
async def get_forecast_history(
    entity_id: str,
    config: RunnableConfig = None,
) -> list:
    """Previsoes anteriores com acuracia real (previsto vs realizado)."""
    async with httpx.AsyncClient(base_url=ML_API_URL, timeout=10.0) as client:
        resp = await client.get(f"/api/v1/forecasts/{entity_id}")
        resp.raise_for_status()
        return resp.json()


@tool
async def validate_forecast(
    forecast_id: str,
    config: RunnableConfig = None,
) -> dict:
    """Compara previsao passada com resultado real (MAPE, MAE)."""
    async with httpx.AsyncClient(base_url=ML_API_URL, timeout=10.0) as client:
        resp = await client.get(f"/api/v1/forecasts/{forecast_id}/validate")
        resp.raise_for_status()
        return resp.json()
```

#### 6. Gerente de Operacoes (Facebook API + interrupt)

```python
from langgraph.types import interrupt


@tool
async def update_campaign_budget(
    campaign_id: str,
    new_daily_budget: float,
    reason: str,
    config: RunnableConfig = None,
) -> str:
    """Atualiza orcamento diario. Requer aprovacao humana."""
    # 1. Busca dados atuais para mostrar diff
    current_budget = 100.00  # await fb_service.get_campaign(campaign_id)
    diff_pct = ((new_daily_budget - current_budget) / current_budget) * 100

    # 2. INTERRUPCAO: pausa aqui ate aprovacao humana
    approval = interrupt({
        "type": "budget_change",
        "details": {
            "campaign_id": campaign_id,
            "current_value": current_budget,
            "new_value": new_daily_budget,
            "diff": f"{diff_pct:+.1f}%",
            "reason": reason,
        },
    })

    # 3. Retomada apos decisao do usuario
    if approval.get("approved"):
        final_budget = approval.get("new_budget_override", new_daily_budget)
        # await fb_service.update_budget(campaign_id, final_budget)
        return f"Orcamento atualizado: R${current_budget} -> R${final_budget}"

    return f"Acao cancelada pelo usuario. Motivo: {approval.get('reason', 'N/A')}"


@tool
async def update_campaign_status(
    campaign_id: str,
    new_status: Literal["ACTIVE", "PAUSED"],
    reason: str,
    config: RunnableConfig = None,
) -> str:
    """Pausa ou ativa uma campanha. Sempre requer aprovacao humana."""
    approval = interrupt({
        "type": "status_change",
        "details": {
            "campaign_id": campaign_id,
            "new_status": new_status,
            "reason": reason,
        },
    })

    if approval.get("approved"):
        # await fb_service.update_status(campaign_id, new_status)
        return f"Campanha {new_status.lower()} com sucesso."

    return "Acao cancelada pelo usuario."


@tool
async def get_recommendations(
    entity_type: Literal["campaign", "adset", "ad"],
    config: RunnableConfig = None,
) -> list:
    """Busca recomendacoes geradas pelo sistema ML."""
    account_id = config.get("configurable", {}).get("account_id")
    async with httpx.AsyncClient(base_url=ML_API_URL, timeout=10.0) as client:
        resp = await client.get(
            "/api/v1/recommendations",
            params={"account_id": account_id, "entity_type": entity_type},
        )
        resp.raise_for_status()
        return resp.json()


@tool
async def apply_recommendation(
    recommendation_id: str,
    config: RunnableConfig = None,
) -> str:
    """Aplica uma recomendacao especifica (requer aprovacao humana)."""
    # 1. Busca detalhes da recomendacao via ML API
    # 2. interrupt() para aprovacao
    # 3. Executa via Facebook API
    pass
```

### 3.4 — Padroes de Seguranca nas Tools

| Padrao | Onde | Como |
|--------|------|------|
| Context Injection | Todas as tools | `config["configurable"]` injeta user_id, account_id — LLM nunca ve |
| InjectedStore | Tools de memoria | `Annotated[BaseStore, InjectedStore]` injeta Store transparente ao LLM |
| Interrupt obrigatorio | Tools de escrita | `interrupt()` pausa para aprovacao humana |
| Edit on Resume | Tools de escrita | `approval.get("new_budget_override")` permite usuario editar valor |
| Dry Run | Tools de escrita | Simula antes de executar (opcional) |
| Rate limiting | Tools HTTP | Respeitam rate limit da ML API e Facebook API |
| Error handling | Tools HTTP | try/except com mensagem amigavel em vez de stack trace |
| Idempotencia | Tools de escrita | Verificam estado atual antes de aplicar mudancas |

---

## 4. Streaming, Frontend e UX ("Sala de Controle")

### 4.1 — Endpoint SSE (Backend FastAPI)

```python
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from langgraph.types import Command
import json

router = APIRouter(prefix="/api/v1/agent")


@router.post("/chat")
async def chat_stream(request: Request, body: ChatRequest):
    """Endpoint principal de chat. Retorna SSE stream."""
    config = {
        "configurable": {
            "thread_id": body.thread_id,
            "user_id": body.user_id,
            "account_id": body.account_id,
        }
    }

    # Se for um resume (aprovacao de interrupt), passa Command
    if body.resume_payload:
        input_data = Command(resume=body.resume_payload)
    else:
        input_data = {
            "messages": [HumanMessage(content=body.message)],
            "user_context": {
                "user_id": body.user_id,
                "account_id": body.account_id,
            },
        }

    async def event_generator():
        async for namespace, mode, chunk in graph.astream(
            input_data,
            config=config,
            stream_mode=["messages", "updates", "custom"],
            subgraphs=True,
        ):
            agent_source = namespace[-1] if namespace else "supervisor"

            if mode == "messages":
                msg_chunk, metadata = chunk
                if msg_chunk.content:
                    yield sse_event("message", {
                        "content": msg_chunk.content,
                        "agent": metadata.get("langgraph_node", "supervisor"),
                    })

            elif mode == "updates":
                if "__interrupt__" in chunk:
                    interrupt_data = chunk["__interrupt__"][0].value
                    yield sse_event("interrupt", {
                        "type": interrupt_data["type"],
                        "details": interrupt_data["details"],
                        "thread_id": body.thread_id,
                    })
                else:
                    for node_name in chunk:
                        yield sse_event("agent_status", {
                            "agent": node_name,
                            "source": agent_source,
                            "status": "completed",
                        })

            elif mode == "custom":
                yield sse_event("agent_progress", {
                    "agent": agent_source,
                    **chunk,
                })

        yield sse_event("done", {"thread_id": body.thread_id})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def sse_event(event_type: str, data: dict) -> str:
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"
```

### 4.2 — Protocolo de Eventos SSE

| Evento | Origem | Payload | Acao no Frontend |
|--------|--------|---------|------------------|
| `message` | LLM tokens | `{content, agent}` | Renderiza texto no chat (efeito digitacao) |
| `agent_status` | Transicao de nos | `{agent, source, status}` | Acende/apaga card do agente na sidebar |
| `agent_progress` | `StreamWriter` custom | `{agent, status, progress}` | Barra de progresso no card do agente |
| `interrupt` | Tool de escrita | `{type, details, thread_id}` | Exibe widget de aprovacao inline |
| `done` | Fim do stream | `{thread_id}` | Finaliza indicadores de loading |

### 4.3 — Uso do StreamWriter nos Nos (Custom Events)

```python
from langgraph.config import get_stream_writer

def anomaly_detection_node(state: HealthSubgraphState):
    writer = get_stream_writer()

    writer({"agent": "health_monitor", "status": "fetching_data", "progress": 10})
    data = fetch_metrics(...)

    writer({"agent": "health_monitor", "status": "running_isolation_forest", "progress": 50})
    anomalies = run_isolation_forest(data)

    writer({"agent": "health_monitor", "status": "diagnosing", "progress": 90})
    diagnosis = format_diagnosis(anomalies)

    return {"anomaly_results": anomalies, "diagnosis": diagnosis}
```

### 4.4 — Arquitetura do Frontend (Next.js)

```
frontend/
├── app/app/ai-agent/
│   ├── page.tsx                    # Pagina principal do chat
│   └── layout.tsx                  # Layout com sidebar de agentes
│
├── components/ai-agent/
│   ├── chat-container.tsx          # Container principal do chat
│   ├── message-list.tsx            # Lista de mensagens scrollable
│   ├── message-bubble.tsx          # Bolha individual (user/assistant)
│   ├── message-input.tsx           # Input de texto + send button
│   ├── streaming-text.tsx          # Renderiza tokens em tempo real
│   │
│   ├── agent-sidebar.tsx           # Sidebar com cards dos agentes
│   ├── agent-card.tsx              # Card individual de um agente
│   ├── agent-progress-bar.tsx      # Barra de progresso dentro do card
│   │
│   ├── approval-widget.tsx         # Widget de aprovacao (interrupt)
│   ├── budget-change-card.tsx      # Card para mudanca de budget
│   ├── status-change-card.tsx      # Card para pausa/ativacao
│   │
│   └── conversation-history.tsx    # Lista de threads anteriores
│
├── hooks/
│   ├── use-agent-chat.ts           # Hook principal: SSE + state management
│   ├── use-agent-stream.ts         # Gerencia conexao SSE e parsing
│   └── use-thread-history.ts       # Gerencia historico de conversas
│
├── types/
│   └── ai-agent.ts                 # Types: Message, AgentStatus, InterruptPayload
│
└── lib/
    └── agent-api.ts                # Client HTTP para o Agent API
```

### 4.5 — Hook Principal: use-agent-chat.ts

```typescript
interface AgentStatus {
  id: string;
  name: string;
  status: 'idle' | 'running' | 'completed' | 'error';
  progress?: { message: string; percent: number };
}

interface InterruptPayload {
  type: string;
  details: Record<string, any>;
  threadId: string;
}

export function useAgentChat(accountId: string) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [agents, setAgents] = useState<Record<string, AgentStatus>>({});
  const [interrupt, setInterrupt] = useState<InterruptPayload | null>(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [threadId] = useState(() => crypto.randomUUID());

  // Funcao generica para processar stream (usada no envio e no resume)
  const processStream = useCallback(async (body: any) => {
    setIsStreaming(true);
    setInterrupt(null);

    const response = await fetch('/api/v1/agent/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ...body, account_id: accountId }),
    });

    const reader = response.body!.getReader();
    const decoder = new TextDecoder();
    let assistantBuffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const events = parseSSE(decoder.decode(value, { stream: true }));

      for (const event of events) {
        switch (event.type) {
          case 'message':
            assistantBuffer += event.data.content;
            updateStreamingMessage(assistantBuffer);
            break;
          case 'agent_status':
            updateAgentStatus(event.data.agent, event.data.status);
            break;
          case 'agent_progress':
            updateAgentProgress(event.data.agent, event.data);
            break;
          case 'interrupt':
            setInterrupt(event.data);
            setIsStreaming(false);
            return;
          case 'done':
            setIsStreaming(false);
            finalizeMessage(assistantBuffer);
            break;
        }
      }
    }
  }, [accountId]);

  const sendMessage = useCallback(async (content: string) => {
    setMessages(prev => [...prev, { role: 'user', content }]);
    setAgents({});
    await processStream({ message: content, thread_id: threadId });
  }, [processStream, threadId]);

  // Resume apos aprovacao — reabre stream para receber continuacao
  const resumeWithApproval = useCallback(async (
    approved: boolean,
    overrides?: Record<string, any>
  ) => {
    await processStream({
      thread_id: threadId,
      resume_payload: { approved, ...overrides },
    });
  }, [processStream, threadId]);

  return { messages, agents, interrupt, isStreaming, sendMessage, resumeWithApproval };
}
```

### 4.6 — Widget de Aprovacao (Interrupt)

```typescript
interface ApprovalWidgetProps {
  interrupt: InterruptPayload;
  onApprove: (overrides?: Record<string, any>) => void;
  onReject: (reason?: string) => void;
}

export function ApprovalWidget({ interrupt, onApprove, onReject }: ApprovalWidgetProps) {
  const [editedValue, setEditedValue] = useState<number | null>(null);

  if (interrupt.type === 'budget_change') {
    return (
      <Card className="border-amber-500 bg-amber-50">
        <CardHeader>
          <AlertTriangle className="text-amber-600" />
          <span>Aprovacao Necessaria</span>
        </CardHeader>
        <CardContent>
          <p>Campanha: <strong>{interrupt.details.campaign_name}</strong></p>
          <p>Orcamento atual: R$ {interrupt.details.current_value}</p>
          <p>Novo orcamento: R$ {interrupt.details.new_value}
            <span className="text-sm">({interrupt.details.diff})</span>
          </p>
          <Input
            type="number"
            placeholder="Ou defina outro valor..."
            onChange={(e) => setEditedValue(parseFloat(e.target.value))}
          />
        </CardContent>
        <CardFooter>
          <Button variant="success" onClick={() =>
            onApprove(editedValue ? { new_budget_override: editedValue } : undefined)
          }>
            Aprovar
          </Button>
          <Button variant="destructive" onClick={() => onReject()}>
            Rejeitar
          </Button>
        </CardFooter>
      </Card>
    );
  }
  // ... outros tipos de interrupt
}
```

### 4.7 — Layout da Interface

```
+-------------------------------------------------------------+
|  FamaChat                              [Conta: MinhaEmpresa]|
+----------------------------------+--------------------------+
|                                  |    AGENTES ATIVOS        |
|   Chat Principal                 |                          |
|                                  |  +--------------------+  |
|   User: "Meu CPL subiu 40%      |  | Monitor Saude      |  |
|          essa semana"            |  | xxxxxxxx-- 80%     |  |
|                                  |  | Rodando IsolForest |  |
|   Agent: "Detectei anomalias    |  +--------------------+  |
|       em 3 campanhas. O CPL     |  +--------------------+  |
|       medio subiu de R$12       |  | Analista Perf.     |  |
|       para R$16.80..."          |  | xxxxxxxxxx 100%    |  |
|                                  |  | Concluido          |  |
|   +--APROVACAO NECESSARIA-----+  |  +--------------------+  |
|   |                           |  |  +--------------------+  |
|   | Pausar campanha           |  |  | Forecaster         |  |
|   | "Promo Verao 2026"        |  |  | xxxxxx---- 60%     |  |
|   |                           |  |  | Gerando projecao   |  |
|   | Status: ACTIVE -> PAUSED  |  |  +--------------------+  |
|   |                           |  |  +--------------------+  |
|   | [Aprovar]     [Rejeitar]  |  |  | Gerente Ops        |  |
|   +---------------------------+  |  | Aguardando         |  |
|                                  |  | aprovacao...       |  |
|   +---------------------------+  |  +--------------------+  |
|   | Digite sua mensagem...    |  |                          |
|   +---------------------------+  |  Threads Anteriores      |
|                                  |  - Analise semanal (2h)  |
|                                  |  - Budget review (ontem) |
+----------------------------------+--------------------------+
|  Dica: O agente lembra que CPL sobe aos sabados nesta conta |
+-------------------------------------------------------------+
```

### 4.8 — Generative UI

O `message-bubble.tsx` suporta renderizacao condicional baseada no tipo de dado:

- Se tool_name === `generate_forecast` -> renderiza `<RechartsLineChart />`
- Se tool_name === `get_ad_preview` -> renderiza `<AdPreviewCard />` com imagem/video
- Se tool_name === `compare_periods` -> renderiza `<ComparisonTable />`
- Texto padrao -> renderiza Markdown com `react-markdown`

---

## 5. Estrutura de Arquivos e Organizacao do Codigo

### 5.1 — Layout do Modulo Agent

```
projects/agent/
├── __init__.py
│
├── config.py                          # AgentSettings (Pydantic BaseSettings)
│
├── graph/                             # Grafo principal (SuperGraph)
│   ├── __init__.py
│   ├── supervisor.py                  # No Supervisor: classificacao + Send() dispatch
│   ├── synthesizer.py                 # No Synthesizer: fan-in + resposta final
│   ├── state.py                       # SupervisorState, AgentReport, UserContext
│   ├── routing.py                     # RoutingDecision schema + logica de classificacao
│   └── builder.py                     # Compilacao do grafo principal
│
├── subgraphs/                         # Subgraphs dos 6 especialistas
│   ├── __init__.py
│   ├── health_monitor/                # Monitor de Saude & Anomalias
│   │   ├── __init__.py
│   │   ├── graph.py                   # StateGraph: fetch -> detect -> diagnose
│   │   ├── state.py                   # HealthSubgraphState, HealthInput/Output
│   │   └── nodes.py                   # fetch_metrics, anomaly_detection, diagnose
│   │
│   ├── performance_analyst/           # Analista de Performance & Impacto
│   │   ├── __init__.py
│   │   ├── graph.py                   # StateGraph: analyze -> compare -> report
│   │   ├── state.py                   # PerformanceSubgraphState
│   │   └── nodes.py
│   │
│   ├── creative_specialist/           # Especialista em Criativos
│   │   ├── __init__.py
│   │   ├── graph.py                   # StateGraph: fetch_ads -> fatigue -> recommend
│   │   ├── state.py                   # CreativeSubgraphState
│   │   └── nodes.py
│   │
│   ├── audience_specialist/           # Especialista em Audiencias (no simples)
│   │   ├── __init__.py
│   │   └── node.py                    # Funcao unica (sem subgraph)
│   │
│   ├── forecast_scientist/            # Cientista de Previsao (no simples)
│   │   ├── __init__.py
│   │   └── node.py                    # Funcao unica (sem subgraph)
│   │
│   └── operations_manager/            # Gerente de Operacoes
│       ├── __init__.py
│       ├── graph.py                   # StateGraph: propose -> interrupt -> execute
│       ├── state.py                   # OperationsSubgraphState
│       └── nodes.py
│
├── tools/                             # Tools organizadas por dominio
│   ├── __init__.py
│   ├── shared.py                      # save_insight, recall_insights
│   ├── health_tools.py
│   ├── performance_tools.py
│   ├── creative_tools.py
│   ├── audience_tools.py
│   ├── forecast_tools.py
│   └── operations_tools.py
│
├── llm/                               # Gerenciamento de LLMs
│   ├── __init__.py
│   └── provider.py                    # get_model(role, config) com fallback
│
├── prompts/                           # System prompts por agente
│   ├── __init__.py
│   ├── supervisor.py
│   ├── synthesizer.py
│   ├── health_monitor.py
│   ├── performance_analyst.py
│   ├── creative_specialist.py
│   ├── audience_specialist.py
│   ├── forecast_scientist.py
│   └── operations_manager.py
│
├── memory/                            # Configuracao de memoria
│   ├── __init__.py
│   ├── store.py                       # PostgresStore factory + index config
│   └── checkpointer.py               # AsyncPostgresSaver factory
│
└── api/                               # Endpoints REST do Agent
    ├── __init__.py
    ├── router.py                      # APIRouter: /chat, /threads, /health
    ├── schemas.py                     # ChatRequest, ChatResponse, ResumePayload
    ├── stream.py                      # SSE event generator + parsing
    └── dependencies.py                # FastAPI deps: get_graph, get_store, auth
```

### 5.2 — Entry Point do Agent API

```python
# app/agent_main.py

from fastapi import FastAPI
from projects.agent.api.router import router as agent_router
from projects.agent.memory.store import init_store
from projects.agent.memory.checkpointer import init_checkpointer
from shared.infrastructure.logging.structlog_config import setup_logging
from shared.observability.tracing import setup_tracing
from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):
    store = await init_store()
    checkpointer = await init_checkpointer()
    app.state.store = store
    app.state.checkpointer = checkpointer
    yield
    # Shutdown: cleanup de conexoes


app = FastAPI(
    title="FamaChat Agent API",
    version="1.0.0",
    lifespan=lifespan,
)

setup_logging()
setup_tracing(app, service_name="agent-api")
app.include_router(agent_router, prefix="/api/v1/agent")
```

### 5.3 — Compilacao do Grafo (builder.py)

```python
# projects/agent/graph/builder.py

from langgraph.graph import StateGraph, START, END
from langgraph.types import RetryPolicy

from projects.agent.graph.state import SupervisorState
from projects.agent.graph.supervisor import supervisor_node
from projects.agent.graph.synthesizer import synthesizer_node
from projects.agent.subgraphs.health_monitor.graph import build_health_graph
from projects.agent.subgraphs.performance_analyst.graph import build_performance_graph
from projects.agent.subgraphs.creative_specialist.graph import build_creative_graph
from projects.agent.subgraphs.audience_specialist.node import audience_node
from projects.agent.subgraphs.forecast_scientist.node import forecast_node
from projects.agent.subgraphs.operations_manager.graph import build_operations_graph

LLM_RETRY = RetryPolicy(max_attempts=2, initial_interval=2.0)
HTTP_RETRY = RetryPolicy(max_attempts=3, initial_interval=1.0, backoff_factor=2.0)


def build_supervisor_graph():
    """Constroi o SuperGraph principal."""
    health_subgraph = build_health_graph()
    performance_subgraph = build_performance_graph()
    creative_subgraph = build_creative_graph()
    operations_subgraph = build_operations_graph()

    builder = StateGraph(SupervisorState)

    # Nos
    builder.add_node("supervisor", supervisor_node, retry=LLM_RETRY)
    builder.add_node("health_monitor", health_subgraph, retry=HTTP_RETRY)
    builder.add_node("performance_analyst", performance_subgraph, retry=HTTP_RETRY)
    builder.add_node("creative_specialist", creative_subgraph, retry=HTTP_RETRY)
    builder.add_node("audience_specialist", audience_node, retry=HTTP_RETRY)
    builder.add_node("forecast_scientist", forecast_node, retry=HTTP_RETRY)
    builder.add_node("operations_manager", operations_subgraph)
    builder.add_node("synthesizer", synthesizer_node, retry=LLM_RETRY)

    # Edges
    builder.add_edge(START, "supervisor")

    # Supervisor -> Fan-out via Send() (retornado pelo supervisor_node)
    builder.add_conditional_edges(
        "supervisor",
        lambda state: state,
        [
            "health_monitor",
            "performance_analyst",
            "creative_specialist",
            "audience_specialist",
            "forecast_scientist",
            "operations_manager",
        ],
    )

    # Fan-in: todos os agentes -> Synthesizer
    builder.add_edge("health_monitor", "synthesizer")
    builder.add_edge("performance_analyst", "synthesizer")
    builder.add_edge("creative_specialist", "synthesizer")
    builder.add_edge("audience_specialist", "synthesizer")
    builder.add_edge("forecast_scientist", "synthesizer")
    builder.add_edge("operations_manager", "synthesizer")

    builder.add_edge("synthesizer", END)

    return builder


def compile_graph(checkpointer, store):
    """Compila o grafo com persistencia."""
    builder = build_supervisor_graph()
    return builder.compile(checkpointer=checkpointer, store=store)
```

### 5.4 — Supervisor Node (Classificacao + Send)

```python
# projects/agent/graph/supervisor.py

from langgraph.types import Send
from langchain_core.messages import AIMessage
from projects.agent.graph.routing import RoutingDecision
from projects.agent.llm.provider import get_model


def supervisor_node(state: SupervisorState):
    """Classifica intencao e despacha agentes em paralelo via Send()."""
    if state["remaining_steps"] <= 3:
        return {"agent_reports": [{
            "agent_id": "supervisor",
            "status": "completed",
            "summary": "Limite de processamento atingido.",
            "confidence": 0.5,
        }]}

    last_message = state["messages"][-1]
    model = get_model("supervisor").with_structured_output(RoutingDecision)

    decision = model.invoke(last_message.content)

    if not decision.selected_agents:
        return {"messages": [AIMessage(content="Como posso ajudar com seus anuncios?")]}

    return [
        Send(node=agent_id, arg={"messages": [last_message]})
        for agent_id in decision.selected_agents
    ]
```

### 5.5 — Routing Schema

```python
# projects/agent/graph/routing.py

from pydantic import BaseModel, Field
from typing import Literal, List

AgentType = Literal[
    "health_monitor",
    "performance_analyst",
    "creative_specialist",
    "forecast_scientist",
    "operations_manager",
    "audience_specialist",
]


class RoutingDecision(BaseModel):
    reasoning: str = Field(description="Por que estes agentes foram escolhidos")
    selected_agents: List[AgentType] = Field(
        description="Lista de agentes a serem ativados em paralelo"
    )
    urgency: Literal["low", "medium", "high"] = Field(
        description="Nivel de urgencia percebido"
    )
```

### 5.6 — Configuracao do Agent

```python
# projects/agent/config.py

from pydantic_settings import BaseSettings
from typing import Optional


class AgentSettings(BaseSettings):
    # LLM Models
    supervisor_model: str = "claude-3-5-haiku-latest"
    analyst_model: str = "claude-sonnet-4-5-20250929"
    synthesizer_model: str = "claude-sonnet-4-5-20250929"
    operations_model: str = "claude-sonnet-4-5-20250929"
    default_provider: str = "anthropic"

    # API Keys
    anthropic_api_key: str
    openai_api_key: Optional[str] = None

    # ML API
    ml_api_url: str = "http://marketing-api:8000"
    ml_api_timeout: int = 30

    # Memory
    store_embedding_model: str = "openai:text-embedding-3-small"
    store_embedding_dims: int = 1536

    # Streaming
    sse_keepalive_interval: int = 15

    # Safety
    max_budget_change_pct: float = 50.0
    auto_approve_threshold: float = 0.0

    # LangSmith
    langsmith_tracing: bool = False
    langsmith_project: str = "famachat-agent"

    class Config:
        env_prefix = "AGENT_"
        env_file = ".env"


agent_settings = AgentSettings()
```

### 5.7 — Docker Compose

Novo servico no `docker-compose.yml`:

```yaml
agent-api:
  build:
    context: .
    dockerfile: Dockerfile
  container_name: marketing-agent-api
  ports:
    - "8002:8001"
  environment:
    - DATABASE_URL=${DATABASE_URL}
    - REDIS_URL=${REDIS_URL}
    - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
    - OPENAI_API_KEY=${OPENAI_API_KEY}
    - AGENT_ML_API_URL=http://marketing-api:8000
    - LANGSMITH_TRACING=${LANGSMITH_TRACING:-false}
    - LANGSMITH_API_KEY=${LANGSMITH_API_KEY:-}
  command: >
    uvicorn app.agent_main:app
    --host 0.0.0.0 --port 8001
    --workers 1 --loop uvloop
  depends_on:
    - marketing-api
    - redis
    - postgres
  healthcheck:
    test: ["CMD", "curl", "-f", "http://localhost:8001/api/v1/agent/health"]
    interval: 30s
    timeout: 10s
    retries: 3
```

### 5.8 — Dependencias Adicionais (requirements.txt)

```
# LangGraph + LangChain
langgraph>=0.3.0
langgraph-checkpoint-postgres>=0.1.0
langgraph-store-postgres>=0.1.0
langchain-core>=0.3.0
langchain-anthropic>=0.3.0
langchain-openai>=0.3.0

# Embeddings e pgvector
pgvector>=0.3.0

# HTTP client async
httpx>=0.26.0
```

### 5.9 — Mapa de Dependencias entre Modulos

```
                    app/agent_main.py
                         |
                         v
                projects/agent/api/
                    |         |
              router.py   dependencies.py
                    |         |
                    v         v
            projects/agent/graph/builder.py
                    |
          +---------+-----------------------+
          v         v                       v
      graph/     subgraphs/             memory/
   supervisor   (6 agentes)        store + checkpointer
   synthesizer      |
   state            v
   routing      tools/          llm/         prompts/
                (7 modulos)   provider.py   (8 modulos)
                    |
          +---------+----------+
          v         v          v
      ML API    FB Services   PostgreSQL
      (HTTP)    (import)      (async session)
```

---

## 6. Resiliencia, Testes e Observabilidade

### 6.1 — Estrategia de Resiliencia

#### Retry Policies por Tipo de No

```python
from langgraph.types import RetryPolicy

# Nos que chamam APIs externas: retry agressivo
HTTP_RETRY = RetryPolicy(
    max_attempts=3,
    initial_interval=1.0,
    backoff_factor=2.0,
)

# Nos que chamam LLM: retry moderado
LLM_RETRY = RetryPolicy(
    max_attempts=2,
    initial_interval=2.0,
)
```

#### Fallback de LLM (Provider-Level)

```python
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain.chat_models import init_chat_model


def get_model(role: str, config: dict = None):
    """Retorna o modelo para o papel, com fallback automatico."""
    config = config or {}
    configurable = config.get("configurable", {})

    model_map = {
        "supervisor": configurable.get("supervisor_model", agent_settings.supervisor_model),
        "analyst": configurable.get("analyst_model", agent_settings.analyst_model),
        "synthesizer": configurable.get("synthesizer_model", agent_settings.synthesizer_model),
        "operations": configurable.get("operations_model", agent_settings.operations_model),
    }

    model_name = model_map.get(role, agent_settings.analyst_model)
    primary = init_chat_model(model_name, model_provider="anthropic")

    if agent_settings.openai_api_key:
        fallback = init_chat_model("gpt-4o", model_provider="openai")
        return primary.with_fallbacks([fallback])

    return primary
```

#### Graceful Degradation no Synthesizer

```python
async def synthesizer_node(state: SupervisorState, config: RunnableConfig):
    reports = state.get("agent_reports", [])

    # Se nenhum agente respondeu, resposta de fallback
    if not reports:
        return {"messages": [AIMessage(
            content="Desculpe, nao consegui analisar seus dados neste momento. "
                    "Por favor, tente novamente."
        )]}

    # Se alguns agentes falharam, sintetiza com os que responderam
    successful = [r for r in reports if r["status"] == "completed"]
    failed = [r for r in reports if r["status"] == "error"]

    model = get_model("synthesizer", config)
    prompt = build_synthesis_prompt(successful, failed, state["messages"][-1].content)

    response = await model.ainvoke(prompt)
    return {"messages": [response], "synthesis": response.content}
```

#### Recursion Limit com Degradacao Proativa

```python
def supervisor_node(state: SupervisorState):
    if state["remaining_steps"] <= 3:
        return {"agent_reports": [{
            "agent_id": "supervisor",
            "status": "completed",
            "summary": "Limite de processamento atingido. Respondendo com dados disponiveis.",
            "confidence": 0.5,
        }]}
    # ... logica normal
```

### 6.2 — Estrategia de Testes

#### Organizacao

```
tests/
├── unit/agent/
│   ├── graph/
│   │   ├── test_supervisor.py         # Classificacao + routing
│   │   ├── test_synthesizer.py        # Sintese com reports variados
│   │   ├── test_state.py              # Reducers e schemas
│   │   └── test_routing.py            # RoutingDecision schema
│   │
│   ├── subgraphs/
│   │   ├── test_health_monitor.py     # Nos individuais do subgraph
│   │   ├── test_performance.py
│   │   ├── test_creative.py
│   │   ├── test_audience.py
│   │   ├── test_forecast.py
│   │   └── test_operations.py         # Fluxo interrupt/resume
│   │
│   ├── tools/
│   │   ├── test_health_tools.py       # Mock HTTP -> ML API
│   │   ├── test_performance_tools.py  # Mock DB queries
│   │   ├── test_creative_tools.py
│   │   ├── test_audience_tools.py
│   │   ├── test_forecast_tools.py
│   │   ├── test_operations_tools.py   # Interrupt dentro de tool
│   │   └── test_shared_tools.py       # save_insight, recall_insights
│   │
│   ├── llm/
│   │   └── test_provider.py           # Fallback, model selection
│   │
│   └── api/
│       ├── test_chat_endpoint.py      # SSE stream
│       └── test_schemas.py            # Validacao de payloads
│
├── integration/agent/
│   ├── test_full_graph.py             # Grafo completo com LLM mockado
│   ├── test_memory_store.py           # PostgresStore + busca semantica
│   └── test_interrupt_resume.py       # Fluxo completo de aprovacao
```

#### Teste Unitario — No Individual

```python
@pytest.mark.asyncio
async def test_supervisor_routes_to_health_on_anomaly():
    """Testa que o Supervisor aciona o Monitor de Saude para anomalias."""
    mock_model = AsyncMock()
    mock_model.invoke.return_value = RoutingDecision(
        reasoning="CPL alto indica anomalia",
        selected_agents=["health_monitor", "performance_analyst"],
        urgency="high",
    )

    with patch("projects.agent.graph.supervisor.get_model", return_value=mock_model):
        result = supervisor_node({
            "messages": [HumanMessage(content="Meu CPL subiu 40%")],
            "user_context": {"user_id": "u1", "account_id": "a1"},
            "remaining_steps": 100,
        })

    assert len(result) == 2
    assert any(s.node == "health_monitor" for s in result)
    assert any(s.node == "performance_analyst" for s in result)
```

#### Teste Unitario — Subgraph

```python
@pytest.mark.asyncio
async def test_health_monitor_detects_anomaly():
    """Testa que o subgraph detecta anomalia e retorna diagnostico."""
    graph = build_health_graph().compile(checkpointer=MemorySaver())

    mock_anomalies = {
        "anomalies": [
            {"entity_id": "camp_123", "metric": "cpl", "score": 0.95,
             "severity": "HIGH", "explanation": "CPL 40% acima da media"}
        ]
    }

    with patch("projects.agent.tools.health_tools.detect_anomalies") as mock:
        mock.return_value = mock_anomalies
        result = await graph.ainvoke(
            {"messages": [HumanMessage(content="CPL subiu muito")]},
            config={"configurable": {"thread_id": "test-1", "account_id": "acc_456"}},
        )

    assert result["diagnosis"] is not None
    assert "HIGH" in result["diagnosis"] or "anomalia" in result["diagnosis"].lower()
```

#### Teste de Integracao — Interrupt/Resume

```python
@pytest.mark.asyncio
async def test_budget_change_requires_approval():
    """Testa fluxo completo: proposta -> interrupt -> aprovacao -> execucao."""
    from langgraph.types import Command

    graph = compile_graph(checkpointer=MemorySaver(), store=InMemoryStore())
    config = {"configurable": {"thread_id": "test-ops-1",
                                "user_id": "u1", "account_id": "a1"}}

    # 1. Usuario pede para aumentar budget
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="Aumente o budget da campanha X para R$200")]},
        config=config,
    )

    # Deve ter interrompido
    state = graph.get_state(config)
    assert state.next

    # 2. Usuario aprova
    result = await graph.ainvoke(
        Command(resume={"approved": True}),
        config=config,
    )

    final_message = result["messages"][-1].content
    assert "atualizado" in final_message.lower() or "sucesso" in final_message.lower()


@pytest.mark.asyncio
async def test_budget_change_with_override():
    """Testa que o usuario pode editar o valor antes de aprovar."""
    # Similar ao acima, mas com:
    # Command(resume={"approved": True, "new_budget_override": 150.00})
    pass
```

### 6.3 — Observabilidade

#### LangSmith (Tracing de LLM)

```bash
# Ativacao via env vars:
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=ls_...
LANGSMITH_PROJECT=famachat-agent
```

#### OpenTelemetry (Tracing de Infra)

Integra com a stack existente (Prometheus + Grafana + Tempo + OTel Collector).

#### Metricas Prometheus Customizadas

```python
from prometheus_client import Counter, Histogram, Gauge

# Contadores
agent_requests_total = Counter(
    "agent_requests_total", "Total de requests ao Agent API",
    ["endpoint", "status"],
)

agent_dispatches_total = Counter(
    "agent_dispatches_total", "Total de despachos para agentes",
    ["agent_id"],
)

agent_interrupts_total = Counter(
    "agent_interrupts_total", "Total de interrupts disparados",
    ["interrupt_type", "resolution"],
)

# Histogramas
agent_response_duration = Histogram(
    "agent_response_duration_seconds", "Tempo total de resposta",
    ["routing_urgency"],
    buckets=[1, 2, 5, 10, 20, 30, 60],
)

agent_subgraph_duration = Histogram(
    "agent_subgraph_duration_seconds", "Tempo por subgraph",
    ["agent_id"],
    buckets=[0.5, 1, 2, 5, 10, 20],
)

# Gauges
agent_active_streams = Gauge(
    "agent_active_streams", "Streams SSE ativos",
)

agent_store_memories = Gauge(
    "agent_store_memories_total", "Total de memorias no Store",
    ["namespace_type"],
)
```

#### Logging Estruturado

```python
import structlog

logger = structlog.get_logger()

async def supervisor_node(state, config):
    thread_id = config["configurable"]["thread_id"]

    logger.info("supervisor.classify_start",
                thread_id=thread_id,
                message_preview=state["messages"][-1].content[:100])

    decision = await router_model.ainvoke(...)

    logger.info("supervisor.dispatch",
                thread_id=thread_id,
                selected_agents=decision.selected_agents,
                urgency=decision.urgency)
```

#### Dashboard Grafana Sugerido

```
+-------------------------------------------------------------+
|               FamaChat Agent — Dashboard                    |
+-----------------+------------------+------------------------+
|  Requests/min   |  Latencia p50    |  Streams Ativos        |
|     45          |    4.2s          |     12                 |
+-----------------+------------------+------------------------+
|  Despachos por Agente (ultimas 24h)                         |
|  xxxxxxxxxxxx Health Monitor      320                       |
|  xxxxxxxxxx   Performance         280                       |
|  xxxxxx       Creative            180                       |
|  xxxxx        Forecast            150                       |
|  xxxx         Audience            120                       |
|  xx           Operations           60                       |
+-------------------------------------------------------------+
|  Interrupts: 45 disparados | 38 aprovados | 7 rejeitados    |
+-------------------------------------------------------------+
|  Latencia por Subgraph (p95)                                |
|  Health: 3.2s | Perf: 2.1s | Forecast: 8.5s | Ops: 1.2s   |
+-------------------------------------------------------------+
|  Erros: 3 timeouts ML API | 1 LLM fallback ativado         |
+-------------------------------------------------------------+
```

---

## Resumo das Decisoes Arquiteturais

| # | Decisao | Escolha | Alternativas Descartadas |
|---|---------|---------|--------------------------|
| 1 | Agentes especializados | 6 agentes (Saude, Performance, Criativos, Audiencias, Previsao, Operacoes) | 7 agentes originais (muita sobreposicao) |
| 2 | Padrao multi-agente | Supervisor + Subgraphs + Command | Supervisor puro (gargalo), Swarm (sem controle) |
| 3 | Roteamento | Classificacao Estruturada + Send API | Tool-based routing (instavel para paralelo) |
| 4 | Estado | add_messages + operator.add para fan-in | Overwrite simples (perda de dados paralelos) |
| 5 | Memoria longo prazo | PostgresStore + pgvector (Opcao B) | Key-value simples (sem busca semantica) |
| 6 | Protocolo frontend | SSE | WebSocket (complexidade desnecessaria) |
| 7 | UX streaming | Hibrida: chat + cards de agente + custom events | Supervisor-only (caixa preta) |
| 8 | LLM strategy | Configuravel por agente com defaults tiered | Modelo unico (custo alto, sem otimizacao) |
| 9 | Provider | Anthropic + OpenAI fallback | Provider unico (SPOF) |
| 10 | Integracao tools | Hibrida: ML via HTTP, FB/DB via import | Tudo HTTP (latencia) ou tudo direto (acoplamento) |

---

## Proximos Passos (Implementacao)

1. **Setup infraestrutura**: Criar modulo `projects/agent/` com estrutura de diretórios
2. **State + Memory**: Implementar schemas, checkpointer, store
3. **LLM Provider**: Implementar `get_model()` com fallback
4. **Tools**: Implementar tools compartilhadas + por agente
5. **Subgraphs**: Implementar cada agente como subgraph isolado
6. **SuperGraph**: Montar supervisor + routing + synthesizer
7. **API**: Endpoint SSE `/chat` com FastAPI
8. **Frontend**: Pagina de chat + sidebar + approval widgets
9. **Testes**: Unit + integration
10. **Observabilidade**: Metricas + logging + LangSmith
