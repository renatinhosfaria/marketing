# Agente IA

## Arquitetura

O agente opera com sistema multi-agente orquestrado. O OrchestratorAgent coordena 6 subagentes especializados (classification, anomaly, forecast, recommendation, campaign, analysis) em paralelo para fornecer respostas completas. Implementacao principal em `projects/agent/service.py` e `projects/agent/orchestrator/`.

## Streaming

O endpoint `POST /api/v1/agent/chat/stream` usa SSE para streaming de eventos em tempo real, incluindo deteccao de intencao, execucao de subagentes e sintese da resposta.

## Endpoints

- `POST /api/v1/agent/chat` - Chat com resposta completa
- `POST /api/v1/agent/chat/stream` - Chat com streaming SSE
- `POST /api/v1/agent/analyze` - Analise rapida
- `GET /api/v1/agent/subagents` - Lista subagentes disponiveis
- `POST /api/v1/agent/chat/detailed` - Chat com detalhes dos subagentes

## Grafo do Orchestrator

```
START → load_memory → parse_request → plan_execution → dispatch → [subagentes paralelos] → collect_results → synthesize → persist_memory → END
```

- `load_memory`: carrega sumario de conversa, entidades do usuario e contexto RAG cross-thread
- `persist_memory`: persiste sumario, embeddings vetoriais e extrai entidades da resposta

## Memoria Avancada

O agente possui 4 camadas de memoria, cada uma habilitavel independentemente:

| Camada | Config | Descricao |
|--------|--------|-----------|
| Summarization | `AGENT_SUMMARIZATION_ENABLED=true` | Resume conversas longas automaticamente |
| Vector Store | `AGENT_VECTOR_STORE_ENABLED=true` | Busca semantica em historico (requer pgvector) |
| Entity Memory | `AGENT_ENTITY_MEMORY_ENABLED=true` | Extrai e persiste entidades mencionadas |
| Cross-thread | `AGENT_CROSS_THREAD_ENABLED=true` | Aprendizado entre conversas do mesmo usuario |

### Summarization Memory (Phase 1)

Quando uma conversa excede `AGENT_SUMMARIZATION_THRESHOLD` mensagens (default: 20), as mensagens antigas sao resumidas pelo LLM e persistidas em `agent_conversation_summaries`. O sumario e injetado como SystemMessage em conversas futuras na mesma thread.

Servico: `projects/agent/memory/summarization.py`

### Vector Store / RAG (Phase 2)

Armazena embeddings vetoriais (OpenAI `text-embedding-3-small`, 1536 dims) das respostas sintetizadas em `agent_memory_embeddings` com indice HNSW (pgvector). Busca por similaridade cosseno permite injetar contexto de conversas passadas.

Servico: `projects/agent/memory/embeddings.py`

### Entity Memory (Phase 3)

Extrai entidades estruturadas (campanhas, metricas, preferencias, thresholds, insights) via LLM e persiste em `agent_user_entities`. Entidades sao indexadas por `user_id` e ordenadas por frequencia de mencao.

Servico: `projects/agent/memory/entities.py`

### Cross-thread Memory (Phase 4)

Servico unificado `UserMemoryService` combina Entity Memory + Vector Store para fornecer contexto personalizado entre threads diferentes do mesmo usuario.

Servico: `projects/agent/memory/user_memory.py`

### Configuracoes Completas

| Variavel | Default | Descricao |
|----------|---------|-----------|
| `AGENT_SUMMARIZATION_ENABLED` | `true` | Habilita sumarizacao |
| `AGENT_SUMMARIZATION_THRESHOLD` | `20` | Mensagens para triggerar sumarizacao |
| `AGENT_SUMMARIZATION_KEEP_RECENT` | `10` | Mensagens recentes mantidas intactas |
| `AGENT_SUMMARIZATION_MAX_TOKENS` | `600` | Max tokens do sumario |
| `AGENT_VECTOR_STORE_ENABLED` | `false` | Habilita vector store |
| `AGENT_EMBEDDING_MODEL` | `text-embedding-3-small` | Modelo de embeddings |
| `AGENT_RAG_TOP_K` | `3` | Resultados RAG no contexto |
| `AGENT_RAG_MIN_SIMILARITY` | `0.75` | Similaridade minima |
| `AGENT_ENTITY_MEMORY_ENABLED` | `false` | Habilita entity memory |
| `AGENT_ENTITY_MAX_PER_USER` | `50` | Max entidades por usuario |
| `AGENT_CROSS_THREAD_ENABLED` | `false` | Habilita cross-thread |
| `AGENT_CROSS_THREAD_MAX_RESULTS` | `3` | Max resultados cross-thread |

## Persistencia

Conversas e mensagens sao persistidas em tabelas do agente e checkpoints via `projects/agent/memory/checkpointer.py`.

### Tabelas de Memoria

| Tabela | Migracao | Descricao |
|--------|----------|-----------|
| `agent_conversation_summaries` | 013 | Sumarios de conversas longas |
| `agent_memory_embeddings` | 014 | Embeddings vetoriais (pgvector) |
| `agent_user_entities` | 015 | Entidades extraidas de conversas |
