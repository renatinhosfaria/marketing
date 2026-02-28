# Documentacao Marketing

Este diretorio organiza a documentacao tecnica e operacional do projeto Marketing em formato docs-as-code.

## Comece aqui

Fluxo recomendado para onboarding tecnico:

1. Leia [overview.md](overview.md) para entender objetivo e escopo.
2. Leia [architecture.md](architecture.md) para visualizar componentes e fluxo de dados.
3. Leia [configuration.md](configuration.md) e [infra-deploy.md](infra-deploy.md) para subir o ambiente.
4. Consulte [apis.md](apis.md), [ml.md](ml.md) e [facebook-ads.md](facebook-ads.md) para contratos e comportamento funcional.
5. Use [runbooks.md](runbooks.md), [observability.md](observability.md) e [testing.md](testing.md) para operacao diaria.

## Mapa dos servicos

| Servico | Porta externa | Responsabilidade |
| --- | --- | --- |
| Frontend Next.js | 8000 | Interface de operacao e analise |
| ML API (FastAPI) | 8001 | Predicao, forecast, classificacao, recomendacao, anomalia |
| Facebook Ads API (FastAPI) | 8003 | OAuth, sync e analytics de Facebook Ads |
| Redis | 8007 | Broker Celery e cache |
| Flower | 5555 | Monitoramento de filas Celery |
| Grafana (opcional) | 3000 | Dashboards de metricas e traces |
| Prometheus (opcional) | 9090 | Coleta de metricas |

## Documentos

- [overview.md](overview.md): contexto de produto e fronteiras do sistema.
- [architecture.md](architecture.md): componentes, fluxo de dados e portas.
- [backend.md](backend.md): estrutura de backend, rotas e jobs.
- [frontend.md](frontend.md): app Next.js, scripts e configuracoes.
- [apis.md](apis.md): resumo dos contratos de API.
- [ml.md](ml.md): comportamento de features de machine learning.
- [facebook-ads.md](facebook-ads.md): fluxo funcional de Facebook Ads.
- [infra-deploy.md](infra-deploy.md): execucao local e deploy.
- [configuration.md](configuration.md): variaveis de ambiente e padroes.
- [observability.md](observability.md): metricas, traces e logs.
- [runbooks.md](runbooks.md): operacao e resposta a incidentes.
- [testing.md](testing.md): estrategia e comandos de teste.
- [contributing.md](contributing.md): padroes de colaboracao tecnica.
