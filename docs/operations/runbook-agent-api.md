# Runbook — FamaChat Agent API

Guia de resposta a incidentes do `marketing-agent` (porta 8001).

---

## Alertas e Respostas

### 1. Taxa de erro de stream > 1% (janela 15 min)

**Alerta**: `rate(agent_stream_errors_total[5m]) / rate(agent_requests_total[5m]) > 0.01`

**Causas e ações**:

| Causa | Diagnóstico | Ação |
|-------|-------------|------|
| ML API indisponível | `agent_dependency_circuit_open_total{dependency="ml_api"} == 1` | Ver seção 3 abaixo |
| FB API indisponível | `agent_dependency_circuit_open_total{dependency="fb_api"} == 1` | Ver seção 4 abaixo |
| LLM timeout | `grep "graph_error" agent_logs` | Aumentar `AGENT_LLM_TIMEOUT` e `AGENT_SUPERVISOR_TIMEOUT` |
| Redis down | `grep "redis_semaphore.connect_failed"` | Ver seção 5 abaixo |

**Rollback de emergência**: desabilitar features por variável de ambiente sem redeploy:
```bash
# Desabilitar SSE replay (reduz pressão no Redis)
docker service update --env-add AGENT_ENABLE_SSE_REPLAY=false marketing_marketing-agent

# Desabilitar semaphores Redis (voltar para in-memory, single-worker)
docker service update --env-add AGENT_ENABLE_REDIS_COORDINATION=false marketing_marketing-agent

# Desabilitar circuit breakers (bypass fail-fast)
docker service update --env-add AGENT_ENABLE_CIRCUIT_BREAKER=false marketing_marketing-agent

# Desabilitar retry com jitter (reduz latência em cascata)
docker service update --env-add AGENT_ENABLE_RETRY_JITTER=false marketing_marketing-agent
```

---

### 2. Latência p95 > 8s por 3 janelas consecutivas

**Alerta**: `histogram_quantile(0.95, agent_time_to_done_seconds) > 8`

**Diagnóstico**:
```bash
# Verificar latência por subgraph
docker service logs marketing_marketing-agent 2>&1 | grep "subgraph_duration"

# Verificar se circuit breaker ML está aberto
curl http://localhost:8001/api/v1/agent/health
```

**Ações**:
1. Verificar saúde da ML API: `curl http://localhost:8000/health`
2. Aumentar timeout se modelos estiverem lentos: `AGENT_LLM_TIMEOUT=90`
3. Se LLM externo (OpenAI/Anthropic) com latência alta, verificar status da API

---

### 3. ML API indisponível (circuit breaker aberto)

**Sintoma**: `agent_dependency_circuit_open_total{dependency="ml_api"} == 1`

**Diagnóstico**:
```bash
# Ver logs do container ML
docker service logs marketing_marketing-ml

# Checar endpoint de saúde
curl http://marketing-ml:8000/health

# Ver se modelos estão treinados
curl http://marketing-ml:8000/api/v1/models
```

**Ações**:
1. Se ML API está down, aguardar recuperação — o circuit breaker abrirá automaticamente (HALF_OPEN após 60s)
2. Se workers Celery não processaram tarefas de treinamento: `docker service logs marketing_marketing-celery-worker`
3. O Agent API continua respondendo com síntese parcial durante indisponibilidade da ML API

**Reset manual do circuit breaker** (apenas em emergência):
```bash
# Reiniciar o container do agent reseta o estado em-memória
docker service update --force marketing_marketing-agent
```

---

### 4. FB API indisponível (circuit breaker fb_api aberto)

**Sintoma**: `agent_dependency_circuit_open_total{dependency="fb_api"} == 1`

**Diagnóstico**:
```bash
# Ver logs de erro na FB API
docker service logs marketing_marketing-agent 2>&1 | grep "fb_api"

# Verificar status da Facebook Graph API
# https://developers.facebook.com/status/
```

**Ações**:
1. Se for instabilidade temporária da Graph API, aguardar — o CB fecha automaticamente
2. Operações de escrita (alterar budget/status) retornam erro estruturado ao usuário
3. Operações de leitura (insights, campanhas) usam dados em cache no banco local

---

### 5. Redis indisponível

**Sintoma**: logs com `redis_semaphore.connect_failed` ou `sse_session_manager.connect_failed`

**Impacto sem Redis**:
- Semaphores: fallback automático para `asyncio.Semaphore` in-memory (funciona em single-worker)
- SSE replay: desabilitado automaticamente (reconexão reinicia stream)
- Celery tasks: `reap_orphan_sse_sessions` retorna erro (não crítico)

**Diagnóstico**:
```bash
docker service logs marketing_marketing-redis
docker exec -it $(docker ps -q -f name=marketing-redis) redis-cli ping
```

**Ações**:
1. Se Redis reiniciou, o Agent API reconnecta automaticamente no próximo startup
2. Para forçar reconnect: `docker service update --force marketing_marketing-agent`
3. Se Redis está irreparavelmente down, desabilitar features que dependem dele:
```bash
docker service update \
  --env-add AGENT_ENABLE_REDIS_COORDINATION=false \
  --env-add AGENT_ENABLE_SSE_REPLAY=false \
  marketing_marketing-agent
```

---

### 6. Sessões SSE órfãs acima do normal

**Alerta**: `agent_session_orphan_count > 50` (ajustar threshold conforme baseline)

**O que são**: sessões onde o cliente desconectou sem o servidor chamar `close_session()`. Expiram automaticamente via TTL (300s). São inofensivas mas indicam desconexões abruptas.

**Diagnóstico**:
```bash
# Ver contagem atual
docker exec -it $(docker ps -q -f name=marketing-redis) redis-cli \
  --scan --pattern "agent:sse:meta:*" | wc -l

# Ver distribuição de status
docker exec -it $(docker ps -q -f name=marketing-redis) redis-cli \
  --scan --pattern "agent:sse:meta:*" | \
  xargs -I{} redis-cli hget {} status | sort | uniq -c
```

**Ações**:
1. Se órfãs > 100 e crescendo: investigar se o cliente está reconectando em loop
2. Verificar logs de `stream.graph_error` — exceptions no grafo causam disconnect abrupto
3. TTL de 300s as remove automaticamente; nenhuma ação urgente necessária

---

### 7. Secrets inválidos em produção

**Sintoma**: `RuntimeError: Secrets inseguros detectados em producao` no startup

**Ações**:
```bash
# Gerar hash da API key
python3 -c "import hashlib; print(hashlib.sha256(b'SUA_API_KEY').hexdigest())"

# Configurar no Swarm
docker secret create agent_api_key_hash <(echo -n "HASH_GERADO")
docker service update --secret-add agent_api_key_hash marketing_marketing-agent

# Configurar approval token secret
docker service update \
  --env-add AGENT_APPROVAL_TOKEN_SECRET=$(openssl rand -hex 32) \
  marketing_marketing-agent
```

---

## Comandos Úteis

```bash
# Ver métricas Prometheus em tempo real
curl -s http://localhost:8001/metrics | grep agent_

# Verificar estado dos circuit breakers
curl http://localhost:8001/api/v1/agent/health

# Logs estruturados filtrados por nível
docker service logs marketing_marketing-agent 2>&1 | \
  python3 -c "import sys, json; [print(json.dumps(json.loads(l), indent=2)) for l in sys.stdin if l.strip()]" 2>/dev/null | \
  grep -A5 '"level": "error"'

# Contar erros por tipo nas últimas 100 linhas
docker service logs --tail 100 marketing_marketing-agent 2>&1 | grep '"level":"error"' | wc -l
```

---

## Escalação

| Situação | Ação |
|----------|------|
| Erro de stream > 5% por > 30 min | Desabilitar features via feature flags, notificar engenharia |
| Redis completamente down > 5 min | Desabilitar SSE replay + semaphores Redis, abrir ticket de infra |
| LLM externo (OpenAI/Anthropic) down | Verificar status page, sem ação de infra disponível |
| Falha de migration de DB | NÃO iniciar o serviço sem migration; rodar `alembic upgrade head` |
