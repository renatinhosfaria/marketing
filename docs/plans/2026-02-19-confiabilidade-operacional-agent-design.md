# Design — Confiabilidade Operacional do Agent (2–4 semanas)

## Contexto
Este design define a evolução de confiabilidade operacional do Agent API mantendo a stack principal atual (`FastAPI`, `LangGraph`, `Postgres`, `Redis`, `Celery`) e aceitando apenas componentes leves adicionais dentro desse ecossistema.

## Objetivos acordados
- Horizonte: 2–4 semanas
- Escopo: disponibilidade de chat/SSE, escalabilidade horizontal, latência e resiliência a dependências
- Capacidade alvo: carga atual +2x
- SLO alvo:
  - disponibilidade do chat/SSE `>= 99,9%`
  - erro de stream `< 1%`
  - latência `p95 < 8s`

## Abordagens consideradas
1. Hardening incremental local
- Menor risco de implementação
- Ganho limitado em escala horizontal e recuperação

2. Confiabilidade distribuída sobre Redis (recomendada)
- Coordenação crítica sai de memória local e vai para Redis
- Melhor equilíbrio entre risco, prazo e ganho estrutural

3. Desacoplamento pesado orientado a eventos
- Maior robustez de longo prazo
- Custo e risco acima do horizonte deste ciclo

## Arquitetura alvo
### Princípio
Preservar a experiência atual do produto e substituir mecanismos frágeis in-memory por coordenação distribuída, com foco em reconexão SSE sem perda.

### Componentes lógicos
1. Gateway Chat
- valida tenant/thread/account
- aplica admissão e limite distribuído

2. Orquestrador de Sessão SSE
- cria `stream_session_id`
- mantém estado/heartbeat/TTL da sessão
- entrega eventos SSE ao cliente

3. Executor LangGraph
- mantém supervisor/subgrafos/synthesizer
- publica eventos tipados com `event_id` monotônico

4. Camada de Resiliência HTTP
- timeout por operação
- retry bounded com jitter
- circuit breaker por `tenant+dependency+endpoint`

5. Persistência Conversacional
- checkpointer/store atuais preservados
- sessão de stream desacoplada do estado de conversa

### Fluxo de dados
1. Cliente chama `POST /api/v1/agent/chat`.
2. Gateway valida entrada e cria sessão distribuída (`stream_session_id`).
3. Endpoint SSE inicia rápido (reduz “time to first byte”).
4. Executor roda o grafo e publica eventos em `Redis Stream` por sessão.
5. SSE consome e transmite (`message_delta`, `agent_status`, `tool_result`, `interrupt`, `done`, `error`).
6. Em reconexão, cliente envia `Last-Event-ID`; backend faz replay incremental sem reiniciar execução.

## Estratégia de erro e degradação
### Classes de falha
1. Desconexão do cliente
- encerra conexão atual
- mantém sessão curta para reconnect

2. Falha transitória de dependência
- retry bounded
- retorna erro estruturado da tool sem matar sessão inteira

3. Falha persistente de dependência
- circuit breaker `open`
- resposta parcial com transparência e próximos passos

4. Falha interna de subgrafo
- converter para `agent_report(status=error)` consistente

### Princípios
- Falhar com resposta útil
- Evitar cascata de falha
- Preservar continuidade da sessão sempre que possível

## SLO, métricas e alertas
### Métricas obrigatórias
- `agent_stream_error_rate`
- `agent_reconnect_success_rate`
- `agent_first_event_latency_seconds`
- `agent_time_to_done_seconds`
- `agent_session_orphan_count`
- `agent_dependency_circuit_open_total`

### Alertas
- erro de stream acima de 1% (janela 15 min)
- reconnect success abaixo de 98%
- `p95` acima de 8s por 3 janelas consecutivas

## Plano de implementação (2–4 semanas)
### Semana 1 — Baseline + coordenação distribuída
- instrumentar métricas faltantes
- mover limites críticos de concorrência para Redis com TTL
- validar regressão funcional zero

### Semana 2 — SSE resiliente com replay
- implementar `stream_session_id`
- publicar/consumir eventos por `Redis Stream`
- suportar cursor/replay (`Last-Event-ID`) no backend e frontend
- finalização idempotente (`done/error`)

### Semana 3 — Resiliência externa
- middleware interno para timeout/retry/circuit breaker
- padronizar erros de tools
- garantir síntese parcial em indisponibilidade de ML/FB API

### Semana 4 — Hardening + rollout
- teste de carga +2x
- testes de caos leves (timeout, restart, reconnect)
- rollout gradual por feature flag (10% > 50% > 100%)

## Estratégia de testes
### Automatizados
- unitário de sessão/cursor/replay
- integração HTTP para `/chat` com reconexão
- integração de falha ML/FB sem quebra total do stream
- testes de corrida para concorrência distribuída

### Não funcionais
- carga progressiva (50% → 200%)
- validação de SLO antes de 100% do rollout

## Operação e governança
- gates objetivos por etapa antes de ampliar rollout
- rollback rápido por feature flag
- runbook com resposta por tipo de incidente (rede, dependência, sessão órfã)

## Riscos e mitigação
1. Ordenação/replay incorreto de eventos
- `event_id` monotônico + replay por cursor

2. Sessões órfãs
- heartbeat + TTL + reaper periódico

3. Complexidade operacional crescente
- dashboard dedicado + alarmes + runbooks

## Fora de escopo (YAGNI)
- novo event bus fora do Redis
- reescrever supervisor/subgrafos
- migrar para arquitetura totalmente orientada a filas neste ciclo

## Critérios de aceite finais
- SLO alvo atendido na janela de validação
- carga +2x sustentada sem regressão crítica
- reconexão SSE com replay sem perda perceptível para o usuário
