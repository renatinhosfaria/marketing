# CLAUDE.md

Este arquivo orienta o Claude Code (claude.ai/code) ao trabalhar com o codigo deste repositorio.

## Regra de Idioma

**Sempre responda em portugues (Brasil).** Todas as mensagens, explicacoes, comentarios de codigo e commits devem ser em pt-BR.

## Visao Geral

FamaChat Marketing — sistema de microservicos para otimizacao de Facebook Ads usando ML. Construido com FastAPI (backend), Next.js 16 (frontend), Celery (fila de tarefas), PostgreSQL e Redis.

## Comandos

### Backend (Python)
```bash
pytest -q                                          # Rodar todos os testes
pytest tests/caminho/do/test_arquivo.py -v         # Rodar um arquivo de teste
pytest tests/caminho/do/test_arquivo.py::TestClasse::test_metodo -v  # Rodar um teste especifico
pytest -q --cov                                    # Testes com cobertura
```

### Frontend (Next.js)
```bash
cd frontend && npm run dev      # Servidor de desenvolvimento (porta 3001)
cd frontend && npm run build    # Build de producao
cd frontend && npm run lint     # ESLint
```

### Docker (producao)
```bash
docker compose up -d            # Subir todos os servicos
docker compose ps               # Verificar status
docker compose logs <servico>   # Ver logs
```

### Celery (desenvolvimento local)
```bash
celery -A app.celery worker -l info -Q ml,training     # Worker ML
celery -A app.celery worker -l info -Q default          # Worker FB Ads
celery -A app.celery beat -l info                       # Agendador
```

## Arquitetura

### Tres Microservicos FastAPI

| Servico | Entry Point | Porta (host->container) | Funcao |
|---------|------------|------------------------|--------|
| ML API | `app/main.py` | 8001->8000 | Previsoes, classificacoes, deteccao de anomalias, recomendacoes |
| Facebook Ads API | `app/fb_ads_main.py` | 8003->8002 | OAuth, sincronizacao de campanhas, insights |

Servicos de suporte: Frontend (8000->3001), Redis (8007->6379), Flower (5555), dois workers Celery, Celery Beat.

### Layout de Projetos (Domain-Driven)

```
projects/
├── ml/              # Algoritmos ML, treinamento, previsoes
│   ├── algorithms/  # Modelos: classificacao (XGBoost), series temporais (Prophet),
│   │                #   anomalia (IsolationForest/Z-score/IQR), recomendacoes,
│   │                #   impacto causal, transfer learning
│   ├── api/         # Endpoints REST
│   ├── jobs/        # Tasks Celery (training_tasks, scheduled_tasks)
│   ├── services/    # Logica de negocio
│   ├── schemas/     # Modelos Pydantic
│   └── db/          # Modelos de banco e repositorios
└── facebook_ads/    # Integracao com Facebook Ads
    ├── client/      # Wrapper do cliente da API do Facebook
    ├── jobs/        # Tasks Celery (sync, refresh de token, consolidacao)
    └── security/    # Criptografia de tokens
```

### Modulo Compartilhado

```
shared/
├── db/              # Gerenciamento de sessao async SQLAlchemy, modelos readonly
├── infrastructure/
│   ├── config/      # Pydantic BaseSettings (settings.py)
│   ├── logging/     # Configuracao structlog
│   ├── tracing/     # Middleware de rastreamento de requisicoes
│   ├── middleware/   # Rate limiting, CORS
│   └── persistence/ # Engine/session factory do banco
├── core/            # Excecoes, logging, decorators de tracing
└── domain/          # Interfaces e value objects compartilhados
```

### Padroes Principais

- **Async-first**: asyncio/asyncpg em todo o backend
- **Configuracao**: Pydantic BaseSettings carregado do `.env` via `shared/infrastructure/config/settings.py`; acessado com `from shared.config import settings`
- **Filas Celery**: `training` (treinamento de modelos), `ml` (previsoes/classificacoes), `default` (sync Facebook Ads)
- **Roteamento de tasks**: definido em `app/celery.py`; tasks de treinamento vao para fila `training`, tasks ML agendadas para `ml`, FB Ads para `default`
- **Hierarquia ML**: algoritmos operam em 3 niveis — campanha, adset, ad — cada um com seus proprios agendamentos de classificacao/recomendacao/deteccao de anomalias
- **Testes**: organizados em `tests/` como `unit/`, `integration/`, `ml/`, `core/` — espelha a estrutura dos projetos

### Convencoes

- Manter endpoints organizados por dominio em `projects/*/api/`
- Colocar configuracoes em `shared/infrastructure/config/` e `projects/*/config.py`
- Atualizar docs em `docs/` ao alterar comportamento
- Raiz do Python path e `.` (imports como `from projects.ml.services import ...`)
- Documentacao e comunicacao em portugues (Brasil)
- Timezone: `America/Sao_Paulo`
