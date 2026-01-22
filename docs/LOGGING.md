# Sistema de Logging Detalhado - FamaChat ML

## Visão Geral

O FamaChat ML possui um sistema de logging estruturado e detalhado que oferece visibilidade completa das operações do agente de IA, permitindo debug eficiente e análise de comportamento.

### Características Principais

- **Trace Context**: Cada requisição possui um `trace_id` único para rastreamento end-to-end
- **Span Hierarchy**: Hierarquia de operações (orquestrador → subagentes → tools)
- **Logs Estruturados**: Formato JSON para parsing e análise programática
- **Correlação Backend ↔ ML**: Trace IDs compartilhados entre Node.js e Python
- **Debug Facilitado**: Filtrar por trace_id e visualizar execução completa

### Níveis de Log

| Nível | Uso | Exemplo |
|-------|-----|---------|
| `DEBUG` | Informação detalhada de desenvolvimento | Parâmetros de função, estado interno |
| `INFO` | Eventos normais do sistema | Início/fim de requisições, decisões do agente |
| `WARNING` | Situações anormais não-críticas | Timeouts, fallbacks, retries |
| `ERROR` | Erros que afetam operação | Exceções, falhas de API |
| `CRITICAL` | Erros críticos do sistema | Falha total, corrupção de dados |

## Configuração

### Variáveis de Ambiente

```bash
# Nível de log (DEBUG, INFO, WARNING, ERROR, CRITICAL)
LOG_LEVEL=INFO

# Habilitar logging detalhado do agente
AGENT_DETAILED_LOGGING=true

# Logs de prompts completos (cuidado com tamanho)
AGENT_LOG_FULL_PROMPTS=false

# Logs de respostas completas
AGENT_LOG_FULL_RESPONSES=false

# Formato de log (json | text)
LOG_FORMAT=json

# Saída de log (stdout | file | both)
LOG_OUTPUT=stdout

# Diretório de logs (se LOG_OUTPUT=file ou both)
LOG_DIR=/var/log/famachat-ml
```

### Recomendações por Ambiente

#### Development
```bash
LOG_LEVEL=DEBUG
AGENT_DETAILED_LOGGING=true
AGENT_LOG_FULL_PROMPTS=true
AGENT_LOG_FULL_RESPONSES=true
LOG_FORMAT=text
LOG_OUTPUT=stdout
```

#### Staging
```bash
LOG_LEVEL=INFO
AGENT_DETAILED_LOGGING=true
AGENT_LOG_FULL_PROMPTS=false
AGENT_LOG_FULL_RESPONSES=false
LOG_FORMAT=json
LOG_OUTPUT=both
```

#### Production
```bash
LOG_LEVEL=INFO
AGENT_DETAILED_LOGGING=false
AGENT_LOG_FULL_PROMPTS=false
AGENT_LOG_FULL_RESPONSES=false
LOG_FORMAT=json
LOG_OUTPUT=file
```

## Estrutura de Logs

### Trace Context

Cada log entry contém:

```json
{
  "timestamp": "2026-01-22T14:30:45.123Z",
  "level": "INFO",
  "logger": "orchestrator",
  "message": "Starting orchestrator execution",
  "trace_id": "abc-123-def-456",
  "span_id": "span-001",
  "parent_span_id": null,
  "user_id": 42,
  "correlation_id": "req-789"
}
```

### Campos Padrão

| Campo | Tipo | Descrição |
|-------|------|-----------|
| `timestamp` | ISO 8601 | Timestamp UTC |
| `level` | string | Nível do log |
| `logger` | string | Nome do logger (orchestrator, classification_agent, etc.) |
| `message` | string | Mensagem legível |
| `trace_id` | string | ID único da requisição (compartilhado com backend) |
| `span_id` | string | ID único da operação atual |
| `parent_span_id` | string | ID do span pai (null para root) |
| `user_id` | int | ID do usuário (se disponível) |
| `correlation_id` | string | ID de correlação adicional |

### Eventos Principais

#### 1. Início de Requisição
```json
{
  "event": "request_started",
  "trace_id": "abc-123",
  "span_id": "span-root",
  "query": "Como estão as campanhas?",
  "config_id": 1
}
```

#### 2. Detecção de Intent
```json
{
  "event": "intent_detected",
  "trace_id": "abc-123",
  "span_id": "span-root",
  "intent": "analyze_campaigns",
  "confidence": 0.95,
  "required_agents": ["classification", "anomaly"]
}
```

#### 3. Execução de Subagente
```json
{
  "event": "subagent_started",
  "trace_id": "abc-123",
  "span_id": "span-001",
  "parent_span_id": "span-root",
  "agent": "classification_agent",
  "timeout": 30
}
```

#### 4. Tool Call
```json
{
  "event": "tool_call",
  "trace_id": "abc-123",
  "span_id": "span-002",
  "parent_span_id": "span-001",
  "tool_name": "list_campaign_tiers",
  "parameters": {"config_id": 1},
  "duration_ms": 145
}
```

#### 5. Subagente Completo
```json
{
  "event": "subagent_completed",
  "trace_id": "abc-123",
  "span_id": "span-001",
  "agent": "classification_agent",
  "duration_ms": 1234,
  "status": "success",
  "result_size": 512
}
```

#### 6. Síntese Final
```json
{
  "event": "synthesis_completed",
  "trace_id": "abc-123",
  "span_id": "span-root",
  "insights_count": 8,
  "prioritized_count": 5,
  "duration_ms": 456
}
```

#### 7. Requisição Completa
```json
{
  "event": "request_completed",
  "trace_id": "abc-123",
  "span_id": "span-root",
  "total_duration_ms": 4567,
  "status": "success",
  "subagents_executed": 3,
  "total_tokens": 2345
}
```

## Consultando Logs

### Comandos Básicos

```bash
# Ver logs em tempo real
docker-compose logs -f famachat-ml-api

# Últimas 100 linhas
docker-compose logs --tail=100 famachat-ml-api

# Logs de um período específico
docker-compose logs --since="2026-01-22T14:00:00" famachat-ml-api

# Filtrar por trace_id
docker-compose logs famachat-ml-api | grep "trace_id.*abc-123"

# Logs apenas de erros
docker-compose logs famachat-ml-api | grep '"level":"ERROR"'
```

### Análise com jq

```bash
# Extrair apenas eventos de tool calls
docker-compose logs famachat-ml-api | jq 'select(.event=="tool_call")'

# Agrupar por agent e contar
docker-compose logs famachat-ml-api | \
  jq -s 'group_by(.agent) | map({agent: .[0].agent, count: length})'

# Duração média por subagente
docker-compose logs famachat-ml-api | \
  jq -s '[.[] | select(.event=="subagent_completed")] |
    group_by(.agent) |
    map({agent: .[0].agent, avg_ms: (map(.duration_ms) | add / length)})'

# Top 10 tools mais usados
docker-compose logs famachat-ml-api | \
  jq -s '[.[] | select(.event=="tool_call")] |
    group_by(.tool_name) |
    map({tool: .[0].tool_name, count: length}) |
    sort_by(.count) | reverse | .[0:10]'

# Requisições com erro
docker-compose logs famachat-ml-api | \
  jq 'select(.level=="ERROR" or .status=="error")'

# Timeline de uma requisição específica
docker-compose logs famachat-ml-api | \
  jq 'select(.trace_id=="abc-123") |
    {timestamp, event, agent, duration_ms}'
```

### Debug de Requisição Específica

Para debugar uma requisição específica, use o `trace_id`:

```bash
# 1. Extrair trace_id da requisição
curl -H "X-Trace-ID: my-trace-123" http://localhost:8000/api/v1/agent/query

# 2. Buscar todos os logs dessa requisição
docker-compose logs famachat-ml-api | grep "my-trace-123" > trace.log

# 3. Analisar timeline
cat trace.log | jq '{timestamp, event, agent, message, duration_ms}'

# 4. Ver hierarquia de spans
cat trace.log | jq 'select(.span_id) | {span_id, parent_span_id, event, agent}'

# 5. Verificar erros
cat trace.log | jq 'select(.level=="ERROR")'
```

## Métricas

### Performance

Use logs para calcular métricas de performance:

```bash
# P50, P95, P99 de duração total
docker-compose logs famachat-ml-api | \
  jq -s '[.[] | select(.event=="request_completed").total_duration_ms] |
    sort |
    {
      p50: .[length/2],
      p95: .[length*0.95|floor],
      p99: .[length*0.99|floor]
    }'

# Taxa de erro por agente
docker-compose logs famachat-ml-api | \
  jq -s '[.[] | select(.event=="subagent_completed")] |
    group_by(.agent) |
    map({
      agent: .[0].agent,
      total: length,
      errors: [.[] | select(.status=="error")] | length,
      error_rate: ([.[] | select(.status=="error")] | length) / length * 100
    })'

# Tempo médio por tool
docker-compose logs famachat-ml-api | \
  jq -s '[.[] | select(.event=="tool_call")] |
    group_by(.tool_name) |
    map({
      tool: .[0].tool_name,
      avg_ms: (map(.duration_ms) | add / length),
      count: length
    }) |
    sort_by(.avg_ms) | reverse'
```

### Custos (Tokens)

```bash
# Total de tokens por requisição
docker-compose logs famachat-ml-api | \
  jq -s '[.[] | select(.event=="request_completed")] |
    {
      total_requests: length,
      total_tokens: map(.total_tokens) | add,
      avg_tokens: (map(.total_tokens) | add / length)
    }'

# Tokens por agente
docker-compose logs famachat-ml-api | \
  jq -s '[.[] | select(.event=="subagent_completed" and .tokens)] |
    group_by(.agent) |
    map({
      agent: .[0].agent,
      total_tokens: map(.tokens) | add,
      avg_tokens: (map(.tokens) | add / length)
    })'
```

## Correlação Backend ↔ ML

O FamaChat backend envia o `X-Trace-ID` header para o microserviço ML, permitindo correlação entre logs.

### No Backend (Node.js)

```javascript
// server/routes/ai-agent.ts
const traceId = req.headers['x-trace-id'] || generateTraceId();
logger.info('Calling ML service', { trace_id: traceId });

const response = await fetch('http://ml-service:8000/api/v1/agent/query', {
  headers: {
    'X-Trace-ID': traceId,
    'X-API-Key': ML_API_KEY
  }
});
```

### No ML (Python)

```python
# app/api/v1/endpoints/agent.py
trace_id = request.headers.get('X-Trace-ID')
logger.info('Request received', extra={'trace_id': trace_id})
```

### Correlação

```bash
# 1. Backend: buscar trace_id no log do Node.js
grep "abc-123" /var/www/famachat/logs/famachat-backend-out.log

# 2. ML: buscar mesmo trace_id no log do Python
docker-compose logs famachat-ml-api | grep "abc-123"

# 3. Combinar timeline
{
  backend: grep "abc-123" backend.log | jq '.timestamp, .message',
  ml: grep "abc-123" ml.log | jq '.timestamp, .event'
} | sort_by(.timestamp)
```

## Troubleshooting

### Problema: Logs não aparecem

**Solução:**
```bash
# Verificar se container está rodando
docker-compose ps

# Verificar stdout do container
docker logs famachat-ml-api

# Verificar variável LOG_OUTPUT
docker-compose exec famachat-ml-api env | grep LOG_OUTPUT
```

### Problema: Logs muito grandes

**Solução:**
```bash
# Desabilitar logs detalhados
AGENT_LOG_FULL_PROMPTS=false
AGENT_LOG_FULL_RESPONSES=false

# Aumentar nível de log
LOG_LEVEL=WARNING

# Rotacionar logs (Docker)
# Editar docker-compose.yml:
services:
  famachat-ml-api:
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
```

### Problema: Não consigo encontrar trace_id

**Solução:**
```bash
# 1. Buscar por correlation_id ou user_id
docker-compose logs famachat-ml-api | grep "user_id.*42"

# 2. Buscar por timestamp aproximado
docker-compose logs --since="2026-01-22T14:30:00" \
  --until="2026-01-22T14:31:00" famachat-ml-api

# 3. Buscar por query similar
docker-compose logs famachat-ml-api | grep -i "como estão as campanhas"
```

### Problema: Span hierarchy confusa

**Solução:**
```bash
# Visualizar hierarquia de spans para um trace_id
cat trace.log | jq -r '
  select(.span_id) |
  "  " * (.parent_span_id | if . then 1 else 0 end) +
  (.event // .message) +
  " [" + .agent + "]"
'

# Exemplo de output:
# request_started [orchestrator]
#   subagent_started [classification_agent]
#     tool_call [classification_agent]
#     tool_call [classification_agent]
#   subagent_completed [classification_agent]
#   synthesis_completed [orchestrator]
# request_completed [orchestrator]
```

### Problema: Performance lenta

**Solução:**
```bash
# Identificar bottlenecks
docker-compose logs famachat-ml-api | \
  jq -s '[.[] | select(.duration_ms)] |
    sort_by(.duration_ms) | reverse | .[0:10] |
    .[] | {event, agent, tool_name, duration_ms}'

# Verificar timeouts
docker-compose logs famachat-ml-api | grep -i timeout
```

## Referências

- **Código de Logging**: `app/core/logging.py`
- **Trace Context**: `app/core/tracing.py`
- **Configuração**: `app/config.py`
- **Testes**: `tests/core/test_logging.py`, `tests/core/test_tracing.py`
- **Documentação Multi-Agente**: `app/agent/orchestrator/README.md`
