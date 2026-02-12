# Infraestrutura e Deploy

## Ambientes

| Ambiente | Arquivo | Proxy | Comando |
|----------|---------|-------|---------|
| **Producao** (VPS) | `marketing-stack.yml` | Traefik (Swarm) | `docker stack deploy -c marketing-stack.yml marketing` |
| **Desenvolvimento** (local) | `docker-compose.yml` | Nenhum (portas diretas) | `docker compose up -d` |

**NAO rode `docker compose up` na VPS de producao.** Isso cria containers duplicados
que competem com o Swarm pelo mesmo banco e filas Celery.

## Producao — Docker Swarm + Traefik

### Proxy Reverso (Traefik)

Traefik roda como servico externo na rede `network_public`, compartilhado entre stacks.
Configuracao via labels nos servicos do `marketing-stack.yml`.

Roteamento por prioridade:

| Rota | Servico | Prioridade |
|------|---------|------------|
| `/api/v1/agent` | marketing-agent | 10 |
| `/api/v1/facebook-ads` | marketing-fb-ads | 10 |
| `/flower/` | marketing-flower | 10 |
| `/api/` | marketing-api | 5 |
| `/` (catch-all) | marketing-frontend | 1 |

TLS automatico via Let's Encrypt (certresolver `letsencryptresolver`).

### Deploy e Redeploy

```bash
./scripts/redeploy.sh          # Rebuild + deploy completo
./scripts/redeploy.sh backend  # Apenas backend
./scripts/redeploy.sh frontend # Apenas frontend
```

### Monitoramento

```bash
docker service ls | grep marketing           # Status dos servicos
docker service logs marketing_marketing-api  # Logs de um servico
docker service ps marketing_marketing-api    # Tasks/replicas
```

## Desenvolvimento — Docker Compose

Stack em `docker-compose.yml` com portas mapeadas diretamente no host:

| Servico | Host:Container |
|---------|---------------|
| Frontend | 8000:3001 |
| ML API | 8001:8000 |
| FB Ads API | 8003:8002 |
| Agent API | 8008:8001 |
| Redis | 8007:6379 |
| Flower | 5555:5555 |

## Healthchecks

Cada servico possui healthcheck configurado (API, FB Ads, Agent, Redis, Flower).

```bash
# Producao
docker service ls | grep marketing

# Desenvolvimento
docker compose ps
docker compose logs <servico>
```

## Volumes

- `models_storage` para modelos ML
- `logs` para logs
- `redis_ml_data` e `celerybeat_schedule`
