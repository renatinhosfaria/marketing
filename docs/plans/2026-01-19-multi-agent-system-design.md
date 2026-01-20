# Design: Sistema Multi-Agente para FamaChat ML

**Data:** 2026-01-19
**Status:** Aprovado
**Autor:** Equipe FamaChat

---

## 1. Visão Geral

### 1.1 Objetivo

Implementar um sistema multi-agente hierárquico usando LangGraph para melhorar a **escalabilidade** do agente de IA, permitindo análises paralelas de campanhas Facebook Ads.

### 1.2 Decisões de Design

| Aspecto | Decisão |
|---------|---------|
| **Objetivo principal** | Escalabilidade - análises paralelas |
| **Orquestração** | Agente Supervisor (Orchestrator) |
| **Contexto** | Compartilhado entre subagentes via checkpointer |
| **Framework** | LangGraph Hierárquico com Send() |

---

## 2. Arquitetura Geral

```
                              ┌─────────────────────────────────────┐
                              │         ORCHESTRATOR AGENT          │
                              │   • Interpreta intenção do usuário  │
                              │   • Decide quais especialistas      │
                              │   • Dispara em paralelo via Send()  │
                              │   • Sintetiza resposta final        │
                              └──────────────────┬──────────────────┘
                                                 │
                 ┌───────────────┬───────────────┼───────────────┬───────────────┐
                 │               │               │               │               │
                 ▼               ▼               ▼               ▼               ▼
        ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
        │CLASSIFICATION│ │ ANOMALY    │ │  FORECAST  │ │RECOMMENDATION│ │  CAMPAIGN  │
        │   AGENT     │ │   AGENT    │ │   AGENT    │ │    AGENT    │ │   AGENT    │
        │             │ │            │ │            │ │             │ │            │
        │ 4 tools     │ │ 3 tools    │ │ 3 tools    │ │ 3 tools     │ │ 2 tools    │
        └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘
                 │               │               │               │               │
                 └───────────────┴───────────────┼───────────────┴───────────────┘
                                                 ▼
                              ┌─────────────────────────────────────┐
                              │           ANALYSIS AGENT            │
                              │  (consolidação de insights - 6 tools)│
                              └─────────────────────────────────────┘
```

### 2.1 Fluxo de Execução

1. **Usuário** envia pergunta
2. **Orchestrator** identifica intenção e seleciona subagentes
3. **Dispatch paralelo** via `Send()` - subagentes executam simultaneamente
4. **Collect results** - aguarda todos completarem
5. **Synthesis** - gera resposta unificada
6. **Resposta** entregue ao usuário

---

## 3. Orchestrator Agent

### 3.1 Grafo do Orchestrator

```
                                    START
                                      │
                                      ▼
                            ┌─────────────────┐
                            │  parse_request  │
                            └────────┬────────┘
                                     │
                                     ▼
                            ┌─────────────────┐
                            │  plan_execution │
                            └────────┬────────┘
                                     │
                                     ▼
                            ┌─────────────────┐
                            │ dispatch_agents │ ◄── Send() paralelo
                            └────────┬────────┘
                                     │
                    ┌────────────────┼────────────────┐
                    ▼                ▼                ▼
            ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
            │  Subagent A │  │  Subagent B │  │  Subagent C │
            └──────┬──────┘  └──────┬──────┘  └──────┬──────┘
                   └────────────────┼────────────────┘
                                    ▼
                            ┌─────────────────┐
                            │ collect_results │
                            └────────┬────────┘
                                     │
                                     ▼
                            ┌─────────────────┐
                            │   synthesize    │
                            └────────┬────────┘
                                     │
                                     ▼
                                    END
```

### 3.2 Estado do Orchestrator

```python
class OrchestratorState(TypedDict):
    # Conversa
    messages: Annotated[Sequence[BaseMessage], add_messages]
    thread_id: str
    config_id: int
    user_id: int

    # Planejamento
    user_intent: str
    required_agents: list[str]
    execution_plan: dict

    # Resultados dos subagentes
    agent_results: dict[str, AgentResult]

    # Resposta final
    synthesized_response: str
    confidence_score: float
```

### 3.3 Mapeamento Intenção → Agentes

```python
INTENT_TO_AGENTS = {
    "analyze_performance": ["classification", "campaign"],
    "find_problems": ["anomaly", "classification"],
    "get_recommendations": ["recommendation", "classification"],
    "predict_future": ["forecast"],
    "compare_campaigns": ["analysis", "classification"],
    "full_report": ["classification", "anomaly", "recommendation", "forecast"],
    "troubleshoot": ["anomaly", "recommendation", "campaign"],
}
```

---

## 4. Subagentes Especialistas

### 4.1 Estrutura Comum

```
                    START
                      │
                      ▼
              ┌──────────────┐
              │ receive_task │
              └──────┬───────┘
                     │
                     ▼
              ┌──────────────┐
              │  call_model  │
              └──────┬───────┘
                     │
              ┌──────┴──────┐
              ▼             ▼
        ┌──────────┐  ┌───────────┐
        │call_tools│  │  respond  │
        └────┬─────┘  └─────┬─────┘
             │              │
             └──────┬───────┘
                    ▼
                   END
```

### 4.2 Os 6 Subagentes

| Agente | Especialidade | Tools | Quando Acionado |
|--------|---------------|-------|-----------------|
| **ClassificationAgent** | Tiers de performance | 4 | Análise de performance |
| **AnomalyAgent** | Problemas e alertas | 3 | Troubleshooting |
| **ForecastAgent** | Previsões futuras | 3 | Projeções |
| **RecommendationAgent** | Ações sugeridas | 3 | "O que fazer?" |
| **CampaignAgent** | Dados de campanhas | 2 | Detalhes, listagens |
| **AnalysisAgent** | Análises avançadas | 6 | Comparações, ROI |

### 4.3 Tools por Subagente

**ClassificationAgent:**
- `get_classifications`
- `get_campaign_tier`
- `get_high_performers`
- `get_underperformers`

**AnomalyAgent:**
- `get_anomalies`
- `get_critical_anomalies`
- `get_anomalies_by_type`

**ForecastAgent:**
- `get_forecasts`
- `predict_campaign_cpl`
- `predict_campaign_leads`

**RecommendationAgent:**
- `get_recommendations`
- `get_recommendations_by_type`
- `get_high_priority_recommendations`

**CampaignAgent:**
- `get_campaign_details`
- `list_campaigns`

**AnalysisAgent:**
- `compare_campaigns`
- `analyze_trends`
- `get_account_summary`
- `calculate_roi`
- `get_top_campaigns`

---

## 5. Mecanismo de Dispatch Paralelo

### 5.1 Uso do Send()

```python
from langgraph.constants import Send

def dispatch_agents(state: OrchestratorState) -> list[Send]:
    sends = []

    for agent_name in state["required_agents"]:
        sends.append(
            Send(
                node=f"{agent_name}_agent",
                arg={
                    "task": state["execution_plan"][agent_name],
                    "config_id": state["config_id"],
                    "messages": state["messages"],
                    "thread_id": state["thread_id"],
                }
            )
        )

    return sends
```

### 5.2 Timeouts por Subagente

| Subagente | Timeout |
|-----------|---------|
| classification | 30s |
| anomaly | 30s |
| forecast | 45s |
| recommendation | 30s |
| campaign | 20s |
| analysis | 45s |

---

## 6. Síntese de Respostas

### 6.1 Estratégia

```
┌─────────────────────────────────────────────────────────────────┐
│                        RESULTADOS DOS AGENTES                   │
├─────────────────┬─────────────────┬─────────────────────────────┤
│ Classification  │    Anomaly      │      Recommendation         │
└────────┬────────┴────────┬────────┴──────────────┬──────────────┘
         └─────────────────┼───────────────────────┘
                           ▼
                 ┌─────────────────┐
                 │    SYNTHESIZE   │
                 │ • Prioriza info │
                 │ • Remove duplic.│
                 │ • Formata output│
                 └────────┬────────┘
                          ▼
                 RESPOSTA UNIFICADA
```

### 6.2 Priorização

```python
PRIORITY_ORDER = {
    "anomaly": 1,         # Problemas primeiro
    "recommendation": 2,   # Ações a tomar
    "classification": 3,   # Contexto de performance
    "forecast": 4,         # Projeções futuras
    "campaign": 5,         # Dados específicos
    "analysis": 6,         # Análises complementares
}
```

---

## 7. Estrutura de Arquivos

```
famachat-ml/
└── app/
    └── agent/
        ├── __init__.py
        ├── config.py
        ├── service.py
        │
        ├── orchestrator/                # NOVO
        │   ├── __init__.py
        │   ├── graph.py
        │   ├── state.py
        │   ├── nodes/
        │   │   ├── __init__.py
        │   │   ├── parse_request.py
        │   │   ├── plan_execution.py
        │   │   ├── dispatch.py
        │   │   ├── collect_results.py
        │   │   └── synthesize.py
        │   └── prompts.py
        │
        ├── subagents/                   # NOVO
        │   ├── __init__.py
        │   ├── base.py
        │   ├── state.py
        │   │
        │   ├── classification/
        │   │   ├── __init__.py
        │   │   ├── agent.py
        │   │   ├── graph.py
        │   │   └── prompts.py
        │   │
        │   ├── anomaly/
        │   ├── forecast/
        │   ├── recommendation/
        │   ├── campaign/
        │   └── analysis/
        │
        ├── tools/                       # EXISTENTE
        ├── graph/                       # EXISTENTE (deprecar)
        ├── memory/
        ├── llm/
        └── prompts/
```

---

## 8. Configurações

### 8.1 Variáveis de Ambiente

```env
# Multi-Agent System
AGENT_MULTI_AGENT_ENABLED=true
AGENT_ORCHESTRATOR_TIMEOUT=120
AGENT_MAX_PARALLEL_SUBAGENTS=4

# Subagent Timeouts
AGENT_TIMEOUT_CLASSIFICATION=30
AGENT_TIMEOUT_ANOMALY=30
AGENT_TIMEOUT_FORECAST=45
AGENT_TIMEOUT_RECOMMENDATION=30
AGENT_TIMEOUT_CAMPAIGN=20
AGENT_TIMEOUT_ANALYSIS=45

# LLM Config
AGENT_SUBAGENT_LLM_MODEL=claude-sonnet-4-20250514
AGENT_ORCHESTRATOR_LLM_MODEL=claude-sonnet-4-20250514

# Synthesis
AGENT_SYNTHESIS_MAX_TOKENS=4096
AGENT_SYNTHESIS_TEMPERATURE=0.3

# Retry
AGENT_SUBAGENT_MAX_RETRIES=2
AGENT_SUBAGENT_RETRY_DELAY=1.0
```

---

## 9. API e Endpoints

### 9.1 Endpoints Existentes (Mantidos)

```
POST /api/v1/agent/chat
POST /api/v1/agent/chat/stream
GET  /api/v1/agent/conversations
GET  /api/v1/agent/status
```

### 9.2 Novos Endpoints

```
GET  /api/v1/agent/subagents              # Lista subagentes
GET  /api/v1/agent/subagents/{name}/status # Status específico
POST /api/v1/agent/chat/detailed          # Resposta com detalhes
```

### 9.3 Eventos SSE (Streaming)

```
stream_start      → Início do processamento
intent_detected   → Intenção identificada
agents_planned    → Subagentes selecionados
subagent_start    → Subagente X iniciou
subagent_end      → Subagente X terminou
synthesis_start   → Iniciando síntese
text              → Chunk de resposta
done              → Finalizado
```

---

## 10. Plano de Implementação

### Fase 1: Infraestrutura Base
- [ ] Criar estrutura de diretórios
- [ ] Implementar SubagentState e OrchestratorState
- [ ] Implementar BaseSubagent
- [ ] Configurações e variáveis de ambiente

### Fase 2: Subagentes Especialistas
- [ ] CampaignAgent + testes
- [ ] ClassificationAgent + testes
- [ ] AnomalyAgent + testes
- [ ] RecommendationAgent + testes
- [ ] ForecastAgent + testes
- [ ] AnalysisAgent + testes

### Fase 3: Orchestrator
- [ ] parse_request node
- [ ] plan_execution node
- [ ] dispatch_agents com Send()
- [ ] collect_results node
- [ ] synthesize node
- [ ] Grafo completo + testes integração

### Fase 4: API e Integração
- [ ] Novos endpoints
- [ ] Streaming com eventos de subagentes
- [ ] Feature flag para migração gradual
- [ ] Testes E2E

### Fase 5: Migração e Rollout
- [ ] Deploy com feature flag OFF
- [ ] Testes em staging
- [ ] Rollout gradual (10% → 50% → 100%)
- [ ] Monitoramento e ajustes
- [ ] Deprecar agente legado

---

## 11. Métricas de Sucesso

| Métrica | Baseline (Legado) | Meta (Multi-Agent) |
|---------|-------------------|---------------------|
| Latência P50 | ~3s | ≤ 3s |
| Latência P95 | ~8s | ≤ 6s |
| Taxa de erro | < 1% | < 1% |
| Qualidade resposta | - | ≥ 4.0/5.0 |
| Cobertura de intenções | 70% | ≥ 90% |

---

## 12. Rollback Plan

```bash
# Em caso de problemas:

# 1. Via variável de ambiente
AGENT_MULTI_AGENT_ENABLED=false

# 2. Via restart
pm2 restart famachat-ml --env AGENT_MULTI_AGENT_ENABLED=false
```

---

## Resumo

| Componente | Quantidade |
|------------|------------|
| Orchestrator | 1 |
| Subagentes | 6 |
| Tools | 21 |
| Endpoints novos | 3 |
| Arquivos novos | ~35 |
| Fases | 5 |

---

**Documento aprovado em:** 2026-01-19
