# FamaChat ML - Microserviço de Machine Learning

Microserviço Python para otimização de campanhas do Facebook Ads usando Machine Learning e **Agente IA Multi-Agent**.

> ⚠️ **NOTA IMPORTANTE:** O sistema legado single-agent está DEPRECADO desde 2026-01-21.
> Use o novo sistema multi-agente habilitado via `AGENT_MULTI_AGENT_ENABLED=true`.
> Veja [DEPRECATION.md](DEPRECATION.md) para detalhes.

## 📋 Visão Geral

O **FamaChat ML** é um microserviço independente que complementa o FamaChat principal, fornecendo:

- **Recomendações de Otimização** - Sugestões baseadas em regras e ML
- **Classificação de Campanhas** - Categorização por tiers de performance
- **Previsões de CPL/Leads** - Forecast usando Prophet time series
- **Detecção de Anomalias** - Identificação de comportamentos atípicos
- **🆕 Agente IA Multi-Agent** - Orquestrador com 6 subagentes especializados (2026-01-21)

## 🤖 Sistema Multi-Agente (Novo!)

O FamaChat ML agora possui um **sistema multi-agente hierárquico** que substitui o agente monolítico legado:

### Arquitetura Multi-Agente

```
                    ┌────────────────────────┐
                    │   ORCHESTRATOR AGENT   │
                    │  (Coordenador Central)  │
                    └───────────┬────────────┘
                                │
        ┌───────────┬───────────┼───────────┬───────────┐
        ▼           ▼           ▼           ▼           ▼
   ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
   │Classification│Anomaly││Forecast ││Recommendation│Campaign│
   │  Agent  │ │  Agent  │ │  Agent  │ │   Agent   │ │ Agent │
   └─────────┘ └─────────┘ └─────────┘ └───────────┘ └─────────┘
        │           │           │           │           │
        └───────────┴───────────┴───────────┴───────────┘
                                ▼
                        ┌──────────────┐
                        │Analysis Agent│
                        └──────────────┘
```

### 6 Subagentes Especializados

| Agente | Função | Tools |
|--------|--------|-------|
| **ClassificationAgent** | Analisa tiers de performance | 4 |
| **AnomalyAgent** | Identifica problemas críticos | 3 |
| **ForecastAgent** | Previsões de CPL/Leads | 3 |
| **RecommendationAgent** | Ações acionáveis | 3 |
| **CampaignAgent** | Detalhes de campanhas | 2 |
| **AnalysisAgent** | Análises avançadas e ROI | 5 |

### Vantagens

✅ **Análises paralelas** - Subagentes executam simultaneamente
✅ **Melhor performance** - Meta P95 ≤ 6s (vs 8s legado)
✅ **Síntese inteligente** - Priorização automática de insights
✅ **Escalável** - Fácil adicionar novos subagentes
✅ **Streaming SSE** - Eventos de progresso em tempo real

### Configuração

```env
# Habilitar sistema multi-agente (Staging: true | Prod: false)
AGENT_MULTI_AGENT_ENABLED=true
AGENT_ORCHESTRATOR_TIMEOUT=120
AGENT_MAX_PARALLEL_SUBAGENTS=4
```

Veja [app/agent/orchestrator/README.md](app/agent/orchestrator/README.md) para documentação completa.

## 🏗️ Arquitetura

```
┌─────────────────────────┐         REST API         ┌─────────────────────────┐
│                         │◄────────────────────────►│                         │
│   FamaChat (Node.js)    │      (API Key Auth)      │   FamaChat ML (Python)  │
│   - Express.js          │                          │   - FastAPI             │
│   - Port 5000           │                          │   - Port 8000           │
│                         │                          │   - Celery Workers      │
└───────────┬─────────────┘                          └───────────┬─────────────┘
            │                                                    │
            │              ┌──────────────────┐                  │
            └─────────────►│   PostgreSQL     │◄─────────────────┘
                           └──────────────────┘
```

## 🛠️ Stack Tecnológica

| Categoria | Tecnologia |
|-----------|------------|
| Framework | FastAPI + Uvicorn |
| ML | scikit-learn, XGBoost, LightGBM |
| Time Series | Prophet, statsmodels |
| Database | SQLAlchemy + asyncpg |
| Task Queue | Celery + Redis |
| Container | Docker + Docker Compose |

## 🚀 Quick Start

### Pré-requisitos

- Python 3.11+
- PostgreSQL (mesmo do FamaChat)
- Redis
- Docker (opcional, recomendado)

### Desenvolvimento Local

```bash
# Clonar e entrar no diretório
cd /var/www/famachat-ml

# Criar ambiente virtual
python -m venv venv
source venv/bin/activate

# Instalar dependências
pip install -r requirements.txt

# Configurar ambiente
cp .env.example .env
# Editar .env com suas configurações

# Criar tabelas no banco
python scripts/init_db.py

# Iniciar API
uvicorn app.main:app --reload --port 8000

# Em outro terminal: Iniciar Worker Celery
celery -A app.tasks.celery_app worker --loglevel=info

# Em outro terminal: Iniciar Beat (scheduler)
celery -A app.tasks.celery_app beat --loglevel=info
```

### Com Docker

```bash
# Build e iniciar todos os serviços
docker-compose up -d --build

# Ver logs
docker-compose logs -f famachat-ml-api

# Verificar status
curl http://localhost:8000/api/v1/health/detailed

# Parar
docker-compose down
```

## 📡 API Endpoints

### Health (Sem autenticação)
| Método | Endpoint | Descrição |
|--------|----------|-----------|
| GET | `/api/v1/health` | Health check simples |
| GET | `/api/v1/health/detailed` | Health check com dependências |

### Previsões
| Método | Endpoint | Descrição |
|--------|----------|-----------|
| POST | `/api/v1/predictions/cpl` | Prever CPL |
| POST | `/api/v1/predictions/leads` | Prever leads |
| GET | `/api/v1/predictions/series/{type}/{id}` | Série de previsões |

### Classificações
| Método | Endpoint | Descrição |
|--------|----------|-----------|
| GET | `/api/v1/classifications/campaigns` | Listar classificações |
| GET | `/api/v1/classifications/campaigns/{id}` | Obter classificação |
| POST | `/api/v1/classifications/campaigns/classify` | Classificar campanhas |

### Recomendações
| Método | Endpoint | Descrição |
|--------|----------|-----------|
| GET | `/api/v1/recommendations` | Listar recomendações |
| POST | `/api/v1/recommendations/generate` | Gerar recomendações |
| POST | `/api/v1/recommendations/{id}/dismiss` | Descartar |
| POST | `/api/v1/recommendations/{id}/apply` | Marcar como aplicada |

### Anomalias
| Método | Endpoint | Descrição |
|--------|----------|-----------|
| GET | `/api/v1/anomalies` | Listar anomalias |
| GET | `/api/v1/anomalies/summary` | Resumo de anomalias |
| POST | `/api/v1/anomalies/detect` | Detectar anomalias |

## 🔐 Autenticação

Todas as rotas (exceto health) requerem o header:

```
X-API-Key: sua-api-key
```

## 📊 Jobs Agendados

| Job | Horário | Função |
|-----|---------|--------|
| `daily_model_retraining` | 05:00 | Retreinar modelos |
| `daily_classification` | 06:00 | Classificar campanhas |
| `daily_recommendations` | 07:00 | Gerar recomendações |
| `hourly_anomaly_detection` | *:30 | Detectar anomalias |
| `batch_predictions` | */4h | Previsões em batch |

## 📁 Estrutura do Projeto

```
famachat-ml/
├── app/
│   ├── api/v1/endpoints/     # Endpoints da API
│   ├── core/                 # Segurança, logging, exceções
│   ├── db/                   # Modelos e repositórios
│   ├── ml/                   # Algoritmos ML
│   ├── services/             # Lógica de negócio
│   ├── tasks/                # Celery tasks
│   ├── schemas/              # Pydantic schemas
│   ├── config.py             # Configurações
│   └── main.py               # Entry point
├── scripts/                  # Scripts utilitários
├── tests/                    # Testes
├── models_storage/           # Modelos serializados
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## 🔧 Configuração

Variáveis de ambiente principais (`.env`):

```bash
# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/famachat

# Redis
REDIS_URL=redis://localhost:6380/0

# Segurança
ML_API_KEY=sua-chave-secreta

# Ambiente
ENVIRONMENT=production
LOG_LEVEL=INFO
DEBUG=false
```

## 🧪 Testes

```bash
# Executar todos os testes
pytest

# Com cobertura
pytest --cov=app --cov-report=html

# Testes específicos
pytest tests/unit/
pytest tests/integration/
```

## 📈 Monitoramento

- **Flower** (Celery): http://localhost:5555
- **API Docs** (dev): http://localhost:8000/docs
- **Health Check**: http://localhost:8000/api/v1/health/detailed

## 🚧 Status de Implementação

| Fase | Descrição | Status |
|------|-----------|--------|
| 1 | Setup do Microserviço | ✅ Completo |
| 2 | Data Pipeline | 🔄 Pendente |
| 3 | Recomendações | 🔄 Pendente |
| 4 | Classificação | 🔄 Pendente |
| 5 | Previsões | 🔄 Pendente |
| 6 | Anomalias | 🔄 Pendente |
| 7 | Integração Node.js | 🔄 Pendente |

## 📝 Licença

Proprietário - FamaChat

## 🤝 Contribuição

Consulte o documento `CONTRIBUTING.md` no repositório principal do FamaChat.
