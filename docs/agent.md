# Agente IA

## Modo de operacao

O agente pode operar em modo single-agent (grafo monolitico) ou multi-agent (orquestrador). A selecao e feita por `AGENT_MULTI_AGENT_ENABLED` em `.env` e lida em `projects/agent/config.py`.

## Streaming

O endpoint `POST /api/v1/agent/chat/stream` usa SSE para streaming de tokens em tempo real.

## Multi-agent

Endpoints em `/api/v1/agent/multi-agent/*` coordenam subagentes de classification, anomaly, forecast, recommendation, campaign e analysis. Implementacao principal em `projects/agent/service.py` e `projects/agent/orchestrator/`.

## Persistencia

Conversas e mensagens sao persistidas em tabelas do agente e checkpoints via `projects/agent/memory/checkpointer.py`.
