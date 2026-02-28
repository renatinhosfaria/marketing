# Runbooks

## Operacao diaria

Checklist diaria sugerida:

1. verificar saude dos endpoints de ML e Facebook Ads;
2. validar fila Celery no Flower;
3. acompanhar dashboards de erro/latencia no Grafana;
4. checar sincronizacao recente de Facebook Ads;
5. revisar jobs periodicos executados pelo beat;
6. validar resposta do `agent/query` para um prompt padrao.

Comandos uteis:

```bash
bash scripts/healthcheck.sh
docker compose ps
docker compose logs marketing-api --tail=200
docker compose logs marketing-worker --tail=200
curl -sf http://localhost:8003/api/v1/facebook-ads/health/simple
```

Smoke test do Agent Query Tool:

```bash
curl -X POST http://localhost:8003/api/v1/facebook-ads/agent/query \
  -H "Content-Type: application/json" \
  -d '{"prompt":"listar top 5 campanhas por spend da config 1"}'
```

Resposta esperada: `{"success": true, ...}` com `sqlExecuted`, `operationType` e `rowsAffected`.

Teste de bloqueio destrutivo:

```bash
curl -X POST http://localhost:8003/api/v1/facebook-ads/agent/query \
  -H "Content-Type: application/json" \
  -d '{"prompt":"ajuste manual","sql":"UPDATE sistema_facebook_ads_ads SET status = ''PAUSED''"}'
```

Resposta esperada:

```json
{"detail": "UPDATE sem WHERE bloqueado"}
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

- Agent query bloqueando operacao esperada:
- validar se o SQL contem `WHERE` para `UPDATE/DELETE`;
- conferir se nao ha multiplas instrucoes/comentarios;
- revisar prompt para reduzir ambiguidade.
