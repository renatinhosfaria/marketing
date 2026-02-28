# APIs

## ML

Base path: `/api/v1`

Dominios principais:

- `GET /health` e checks de readiness/liveness.
- `POST /predictions/*` para previsao de CPL/leads.
- `GET /forecasts` para consultas de forecast.
- `GET/POST /classifications/*` para classificacao e feedback.
- `GET/POST /recommendations/*` para recomendacoes e aplicacao.
- `GET/POST /anomalies/*` para deteccao e acknowledge.
- `GET/POST /models/*` para treino e ciclo de vida de modelos.
- `POST /impact/analyze` para impacto causal.

## Facebook Ads

Base path: `/api/v1/facebook-ads`

Dominios principais:

- `GET /health` e `GET /health/simple`.
- `GET /oauth/url` e callback/completion de OAuth.
- `GET/POST/PUT/DELETE /config` para contas/configuracoes conectadas.
- `POST /sync/{config_id}` e endpoints de status/historico/cancelamento.
- endpoints de campanhas, adsets e ads.
- `GET /insights/*` para KPIs, series, rankings, comparativos e breakdowns.

### Agent Query Tool

Endpoint:

- `POST /agent/query`

Request JSON:

```json
{
  "prompt": "listar top 5 campanhas por spend da config 1",
  "sql": "SELECT 1",
  "context": {
    "dateFrom": "2026-02-01",
    "dateTo": "2026-02-28"
  }
}
```

Regras:

- `prompt` e obrigatorio.
- `sql` e opcional; quando ausente, o backend aplica traducao `NL -> SQL`.
- guardrails bloqueiam operacoes destrutivas: `DROP`, `TRUNCATE`, `DELETE` sem `WHERE`, `UPDATE` sem `WHERE`.

Response (sucesso):

```json
{
  "success": true,
  "data": {
    "operationType": "SELECT",
    "sqlExecuted": "SELECT campaign_id, SUM(spend) AS spend FROM ... LIMIT 5",
    "rowsAffected": 5,
    "rows": [],
    "durationMs": 42
  }
}
```

Response (bloqueio):

```json
{
  "detail": "UPDATE sem WHERE bloqueado"
}
```

## Autenticacao e seguranca

- APIs protegidas usam `X-API-Key` e, em alguns fluxos, JWT Bearer.
- Rate limit aplicado por middleware.
- Segredos devem ser fornecidos via variaveis de ambiente.
