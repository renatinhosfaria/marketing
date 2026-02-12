# Infraestrutura e Deploy

## Proxy Reverso — Traefik

Traefik e o unico proxy reverso do projeto, usado tanto em desenvolvimento (`docker-compose.yml`) quanto em producao (`marketing-stack.yml` / Docker Swarm).

Configuracao via labels nos servicos Docker. Roteamento por prioridade:

| Rota | Servico | Prioridade |
|------|---------|------------|
| `/api/v1/agent` | marketing-agent | 10 |
| `/api/v1/facebook-ads` | marketing-fb-ads | 10 |
| `/flower/` | marketing-flower | 10 |
| `/api/` | marketing-api | 5 |
| `/` (catch-all) | marketing-frontend | 1 |

TLS automatico via Let's Encrypt (certresolver `letsencryptresolver`).

## Docker Compose (desenvolvimento)

Stack em `docker-compose.yml` inclui Traefik como servico. Subir com:

```bash
docker compose up -d
```

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
- `traefik_letsencrypt` para certificados TLS
