# Runbooks

## Operacao diaria

Checklist diaria sugerida:

1. verificar saude dos endpoints de ML e Facebook Ads;
2. validar fila Celery no Flower;
3. acompanhar dashboards de erro/latencia no Grafana;
4. checar sincronizacao recente de Facebook Ads;
5. revisar jobs periodicos executados pelo beat.

Comandos uteis:

```bash
bash scripts/healthcheck.sh
docker compose ps
docker compose logs marketing-api --tail=200
docker compose logs marketing-worker --tail=200
```

## Incidentes comuns

- API indisponivel:
- validar `DATABASE_URL`, conectividade com banco e readiness.

- Worker sem consumir fila:
- validar `REDIS_URL`, status do worker e fila correta (`training`, `ml`, `default`).

- OAuth Facebook falhando:
- conferir `FACEBOOK_APP_ID`, `FACEBOOK_APP_SECRET`, callback URL e status do token.

- Degradacao de latencia:
- inspecionar traces, pool de conexoes e backlog de tarefas Celery.
