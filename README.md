# FamaChat Marketing

Sistema de microserviços para **otimização de Facebook Ads com Machine Learning**, construído com FastAPI, Next.js 16, Celery, PostgreSQL e Redis. Inclui um ecossistema multi-agente inteligente baseado em LangGraph para análise conversacional e operações assistidas por IA.

---

## Stack Tecnológico

| Camada | Tecnologia |
|--------|------------|
| **Backend** | Python 3.11, FastAPI 0.109, SQLAlchemy 2.0 (async), Celery 5.3 |
| **Frontend** | Next.js 16, React 19, TypeScript, Tailwind CSS 4, Radix UI |
| **ML** | scikit-learn, XGBoost, Prophet, LightGBM, pandas, statsmodels |
| **Agente IA** | LangGraph, LangChain, OpenAI GPT-4o, pgvector |
| **Infra** | Docker Swarm, Traefik, PostgreSQL 15+, Redis 7 |
| **Observabilidade** | Prometheus, Grafana, OpenTelemetry, Tempo, Alertmanager |

---

## Serviços

| Serviço | Porta | Descrição |
|---------|-------|-----------|
| `marketing-frontend` | 8000 → 3001 | Dashboard SPA (Next.js) |
| `marketing-api` | 8001 → 8000 | API ML — previsões, classificações, anomalias, recomendações |
| `marketing-fb-ads` | 8003 → 8002 | API Facebook Ads — OAuth, sync, insights |
| `marketing-agent` | 8008 → 8001 | API Agent — chat multi-agente com SSE streaming |
| `marketing-worker` | — | Celery Worker ML (filas: `training`, `ml`) |
| `marketing-fb-ads-worker` | — | Celery Worker FB Ads (fila: `default`) |
| `marketing-beat` | — | Celery Beat — agendador de tarefas |
| `marketing-redis` | 8007 → 6379 | Redis — broker e cache |
| `marketing-flower` | 5555 | Flower — monitoramento Celery |

---

## Quickstart

### 1. Clone e configure o ambiente

```bash
git clone git@github.com:renatinhosfaria/marketing.git
cd marketing
cp .env.example .env  # Configure as variáveis de ambiente
```

### 2. Suba com Docker Compose (desenvolvimento)

```bash
docker compose up -d
```

### 3. Inicialize o banco de dados

```bash
alembic upgrade head
python scripts/init_db.py
```

### 4. Desenvolvimento local (sem Docker)

```bash
# Backend - ML API
uvicorn app.main:app --reload --port 8000

# Backend - FB Ads API
uvicorn app.fb_ads_main:app --reload --port 8002

# Backend - Agent API
uvicorn app.agent_main:app --reload --port 8001

# Celery Worker
celery -A app.celery worker -l info -Q ml,training

# Frontend
cd frontend && npm install && npm run dev
```

### 5. Verifique os serviços

```bash
curl http://localhost:8000/api/v1/health          # ML API
curl http://localhost:8002/api/v1/facebook-ads/health/simple  # FB Ads API
curl http://localhost:8001/api/v1/agent/health     # Agent API
```

---

## Estrutura do Projeto

```
marketing/
├── app/                    # Entry points dos microserviços
│   ├── main.py             # ML API (FastAPI)
│   ├── fb_ads_main.py      # Facebook Ads API (FastAPI)
│   ├── agent_main.py       # Agent API (FastAPI + LangGraph)
│   └── celery.py           # Configuração Celery + Beat schedule
│
├── projects/               # Bounded contexts (DDD)
│   ├── ml/                 # Módulo de Machine Learning
│   │   ├── algorithms/     # Modelos: XGBoost, Prophet, IsolationForest, etc.
│   │   ├── api/            # Endpoints REST
│   │   ├── jobs/           # Tasks Celery (treinamento + agendadas)
│   │   ├── services/       # Lógica de negócio
│   │   ├── schemas/        # Modelos Pydantic
│   │   └── db/             # Modelos e repositórios de banco
│   │
│   ├── facebook_ads/       # Módulo Facebook Ads
│   │   ├── client/         # Wrapper da Graph API v24.0
│   │   ├── api/            # Endpoints REST
│   │   ├── jobs/           # Tasks Celery (sync, consolidação, token)
│   │   ├── services/       # Serviços de sync e OAuth
│   │   └── security/       # Criptografia AES-256-GCM
│   │
│   └── agent/              # Módulo AI Agent
│       ├── graph/          # Grafo LangGraph (supervisor, agentes, synthesizer)
│       ├── tools/          # Ferramentas dos agentes
│       ├── memory/         # Checkpointer + Store pgvector
│       ├── llm/            # Provider de LLMs
│       ├── api/            # Endpoints SSE
│       └── jobs/           # Tasks de impacto e retenção
│
├── shared/                 # Infraestrutura compartilhada
│   ├── infrastructure/     # Config, persistence, middleware, logging
│   ├── db/                 # Session factory, modelos read-only
│   ├── observability/      # Tracing, métricas, instrumentação Celery
│   └── core/               # Exceções, decorators, utilitários
│
├── frontend/               # Next.js 16 + React 19 + Tailwind CSS 4
│   ├── app/                # App Router (páginas)
│   ├── components/         # Componentes React (layout, ML, FB Ads, AI Agent)
│   ├── hooks/              # Custom hooks (auth, API, chat SSE)
│   ├── lib/                # Utilitários e clientes API
│   └── types/              # Definições TypeScript
│
├── tests/                  # Testes (unit, integration, ML)
├── alembic/                # Migrações de banco de dados
├── scripts/                # Scripts operacionais
├── observability/          # Configurações Prometheus, Grafana, Tempo
├── docs/                   # Documentação completa do projeto
├── docker-compose.yml      # Ambiente de desenvolvimento
└── marketing-stack.yml     # Deploy Docker Swarm (produção)
```

---

## Comandos Úteis

### Backend (Python)

```bash
pytest -q                                          # Todos os testes
pytest tests/caminho/do/test_arquivo.py -v         # Arquivo específico
pytest tests/caminho/do/test_arquivo.py::TestClasse::test_metodo -v  # Teste específico
pytest -q --cov                                    # Testes com cobertura
```

### Frontend (Next.js)

```bash
cd frontend && npm run dev      # Servidor de desenvolvimento (porta 3001)
cd frontend && npm run build    # Build de produção
cd frontend && npm run lint     # ESLint
```

### Docker (produção)

```bash
docker compose up -d            # Subir todos os serviços
docker compose ps               # Verificar status
docker compose logs <servico>   # Ver logs
```

### Celery

```bash
celery -A app.celery worker -l info -Q ml,training     # Worker ML
celery -A app.celery worker -l info -Q default          # Worker FB Ads
celery -A app.celery beat -l info                       # Agendador
```

---

## Documentação

A documentação completa está em [`docs/`](docs/README.md), organizada em:

| Seção | Conteúdo |
|-------|----------|
| [Arquitetura](docs/architecture/) | Visão geral, pipeline ML, sistema multi-agente |
| [Referência de APIs](docs/api/) | ML API, Facebook Ads API, Agent API |
| [Guias](docs/guides/) | Getting started, desenvolvimento, testes, contribuição |
| [Operações](docs/operations/) | Deploy, monitoramento, runbook, tarefas Celery |
| [Dados](docs/data/) | Schema do banco, migrações |
| [Segurança](docs/security.md) | OAuth, criptografia, autenticação, rate limiting |

---

## Licença

Projeto proprietário — © FamaChat. Todos os direitos reservados.
