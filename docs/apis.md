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

## Autenticacao e seguranca

- APIs protegidas usam `X-API-Key` e, em alguns fluxos, JWT Bearer.
- Rate limit aplicado por middleware.
- Segredos devem ser fornecidos via variaveis de ambiente.
