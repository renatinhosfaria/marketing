# Infraestrutura e Deploy

## Proxy Reverso — Traefik (producao)

Traefik e o proxy reverso em producao, gerenciado pelo Docker Swarm via `marketing-stack.yml`.

Configuracao via labels nos servicos. Roteamento por prioridade:

| Rota | Servico | Prioridade |
|------|---------|------------|
| `/api/v1/agent` | marketing-agent | 10 |
| `/api/v1/facebook-ads` | marketing-fb-ads | 10 |
| `/flower/` | marketing-flower | 10 |
| `/api/` | marketing-api | 5 |
| `/` (catch-all) | marketing-frontend | 1 |

TLS automatico via Let's Encrypt (certresolver `letsencryptresolver`).

## Docker Compose (desenvolvimento)

Stack em `docker-compose.yml` com portas mapeadas diretamente no host. Subir com:

```bash
docker compose up -d
```

Portas de desenvolvimento:

| Servico | Host:Container |
|---------|---------------|
| Frontend | 8000:3001 |
| ML API | 8001:8000 |
| FB Ads API | 8003:8002 |
| Agent API | 8008:8001 |
| Redis | 8007:6379 |
| Flower | 5555:5555 |

## Docker Swarm (producao)

Deploy via:

```bash
docker stack deploy -c marketing-stack.yml marketing
```

Traefik roda como servico externo na rede `network_public`.

## Healthchecks

Cada servico possui healthcheck configurado (API, FB Ads, Agent, Redis, Flower).
Use `docker compose ps` e `docker compose logs <servico>` para diagnostico.

## Volumes

- `models_storage` para modelos ML
- `logs` para logs
- `redis_ml_data` e `celerybeat_schedule`
