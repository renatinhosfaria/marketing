# Orchestrator Agent

O Orchestrator e o componente central do sistema multi-agente do Marketing.

## Arquitetura

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
        │CLASSIFICATION│ │  ANOMALY   │ │  FORECAST  │ │RECOMMENDATION│ │  ANALYSIS  │
        └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘
```

## Fluxo de Execucao

1. **parse_request**: Detecta intenção do usuário
2. **plan_execution**: Seleciona subagentes necessários
3. **dispatch_agents**: Dispara subagentes em paralelo via `Send()`
4. **subagent_***: Execução paralela dos especialistas
5. **collect_results**: Agrega resultados
6. **synthesize**: Gera resposta unificada

## Configuracao

```env
AGENT_MULTI_AGENT_ENABLED=true
AGENT_ORCHESTRATOR_TIMEOUT=120
AGENT_MAX_PARALLEL_SUBAGENTS=4
```

## Uso

```python
from app.agent.orchestrator import get_orchestrator, create_initial_orchestrator_state
from langchain_core.messages import HumanMessage

# Criar estado
state = create_initial_orchestrator_state(
    config_id=1,
    user_id=1,
    thread_id="thread-123",
    messages=[HumanMessage(content="Como esta a performance?")]
)

# Executar
orchestrator = get_orchestrator()
result = await orchestrator.ainvoke(state)

# Resposta
print(result["synthesized_response"])
```

## Intencoes Suportadas

| Intencao | Subagentes |
|----------|------------|
| analyze_performance | classification, campaign |
| find_problems | anomaly, classification |
| get_recommendations | recommendation, classification |
| predict_future | forecast |
| compare_campaigns | analysis, classification |
| full_report | classification, anomaly, recommendation, forecast |
| troubleshoot | anomaly, recommendation, campaign |

## Rollback

Para desabilitar o sistema multi-agente:

```bash
# Via variavel de ambiente
AGENT_MULTI_AGENT_ENABLED=false

# Via restart do servico
pm2 restart marketing --env AGENT_MULTI_AGENT_ENABLED=false
```
