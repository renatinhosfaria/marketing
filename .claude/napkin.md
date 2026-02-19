# Napkin

## Corrections
| Date | Source | What Went Wrong | What To Do Instead |
|------|--------|----------------|-------------------|

## User Preferences
- Sempre responder em portugues (Brasil)
- Testes rodam no Docker: `docker compose exec marketing-api python -m pytest ...`
- Rootdir no container: `/app` (nao `/var/www/marketing`)
- Monkeypatchar no namespace do consumidor, nao no modulo original

## Patterns That Work
- SQLite in-memory para testes de integracao com `create_async_engine("sqlite+aiosqlite:///:memory:")`
- Criar tabelas seletivamente com `table.create(checkfirst=True)`
- `pytest-asyncio` com `mode=Mode.STRICT` — usar `@pytest.mark.asyncio`
- Fixtures async: usar `pytest_asyncio.fixture` (nao `pytest.fixture`)

## Patterns That Don't Work
- `Base.metadata.create_all` puxa FKs de tabelas nao importadas — quebra em SQLite
- Monkeypatchar `shared.db.session.async_session_maker` nao afeta tools ja importadas
- Lazy imports nao podem ser mockados no modulo que importa — mockar no modulo original

## Patterns That Work (Teams)
- Despachar agentes em paralelo para tasks independentes e muito eficiente
- 3 agentes paralelos (bugfixer, tester, prompt-engineer) completam em minutos
- Verificar arquivos criados por agentes com `ls -la` para monitorar progresso

## Patterns That Work (Design Docs)
- Ao editar docs longos, numerar correcoes em apendices (A, B, C, D) por rodada de revisao
- Manter referencia cruzada entre secoes (ex: "ver secao 3.6") e atualizar ao renumerar
- Adicionar `ensure_ascii=False` em json.dumps para PT-BR em SSE streams
- Ao adicionar campo a TypedDict usado como input_schema, atualizar TODOS os subgraph states que dependem dele
- `APIRouter(prefix=X)` + `include_router(prefix=X)` duplica prefix — usar so em um lugar
- Em LangGraph, `store` e injetado via runtime: nodes usam `*, store: BaseStore`, tools usam `InjectedStore()`
- `interrupt()` deve ficar em nodes (nao em tools) — tools puras sao mais testaveis
- Idempotency key deve ser hash de campos estaveis (thread_id+campaign+op+value), nao depender de hora
- Tabelas em SQL: usar maps explicitos (TABLE_BY_TYPE) em vez de f-strings com pluralizacao

## Domain Notes
- FamaChat Marketing: microservicos FastAPI + Next.js + LangGraph + Celery
- 3 APIs: ML (8001), Agent (8002), Facebook Ads (8003)
- 6 subagentes: classificacao, anomalia, previsao, recomendacao, campanha, analise
- 20 tools async em `projects/agent/tools/`
- Design doc em `docs/plans/2026-02-10-ai-agent-ecosystem-design.md` — 5 rodadas de revisao (A: API, B: industrial-grade, C: producao-ready, D: consistencia, E: robustez)
- Prompts reescritos como especialistas seniors com benchmarks BR e frameworks ICE/PACED/Pareto
- API ML endpoints: `/api/v1/{classifications,forecasts,anomalies,recommendations,models,predictions}`
- API ML sem autenticacao — usa apenas `Depends(get_db)`
- docs_url=None em producao (settings.debug=False) — sem /docs
- 29 testes unitarios ML + 9 testes de integracao com dados reais — todos passam
- Prophet MAPE muito alto (42-92%) — previsoes subestimam massivamente valores reais
- Classificacoes usam `heuristic_v1` (sem XGBoost treinado — precisaria 10+ campanhas, tem 9)
