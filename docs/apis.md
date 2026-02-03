# APIs

Base: `/api/v1`

## ML

Arquivos: `projects/ml/api/*`

- `GET /health`
- `POST /predictions/cpl`
- `POST /predictions/leads`
- `GET /forecasts`
- `GET /classifications`
- `GET /recommendations`
- `GET /anomalies`
- `GET /models`

## Agente IA

Arquivo: `projects/agent/api/router.py`

- `POST /agent/chat`
- `POST /agent/chat/stream`
- `POST /agent/analyze`
- `GET /agent/status`
- `POST /agent/multi-agent/chat`
- `POST /agent/multi-agent/chat/stream`

## Facebook Ads

Arquivos: `projects/facebook_ads/api/*`

- `GET /facebook-ads/health/simple`
- `GET /facebook-ads/oauth/url`
- `GET /facebook-ads/oauth/callback`
- `POST /facebook-ads/sync/{config_id}`
- `GET /facebook-ads/insights/kpis`
- `GET /facebook-ads/campaigns`
