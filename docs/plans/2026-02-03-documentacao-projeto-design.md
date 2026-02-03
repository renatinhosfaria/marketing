# Documentacao completa do projeto (design)

Data: 2026-02-03

## Contexto e objetivo
Criar documentacao completa do projeto FamaChat ML cobrindo backend, frontend e infraestrutura, com indice central em `docs/README.md`. A documentacao deve estar em portugues e seguir praticas atuais de mercado (estrutura modular, navegacao clara, foco em onboarding, operacao e manutencao).

Publico-alvo principal: dev interno.
Escopo: backend (FastAPI, ML, agente, Facebook Ads), frontend (Next.js) e infraestrutura (Docker/Compose, Redis, Celery, observabilidade, operacao).
Segredos: manter exemplos com valores reais do `.env` conforme solicitado, com aviso de uso seguro.

## Abordagens avaliadas
1. Documentacao modular com indice central (recomendada)
2. Documentacao profunda por dominio (subdiretorios)
3. Documento unico grande com indice interno

Escolha: opcao 1, por equilibrio entre cobertura, navegacao e manutencao.

## Estrutura proposta
Indice central:
- `docs/README.md`

Docs tematicos:
- `docs/overview.md`
- `docs/architecture.md`
- `docs/backend.md`
- `docs/frontend.md`
- `docs/apis.md`
- `docs/ml.md`
- `docs/agent.md`
- `docs/facebook-ads.md`
- `docs/infra-deploy.md`
- `docs/configuration.md`
- `docs/observability.md`
- `docs/runbooks.md`
- `docs/testing.md`
- `docs/contributing.md`

## Conteudo por dominio (resumo)
- Overview: o que e o projeto, objetivos, escopo funcional.
- Arquitetura: topologia dos servicos, fluxos de dados, dependencias.
- Backend: FastAPI, routers, autenticacao, lifecycle, Celery.
- Frontend: Next.js, scripts, env vars, integracao com API.
- APIs: endpoints principais por servico, paths e autenticacao.
- ML: pipeline, previsao, classificacao, recomendacoes, anomalias, thresholds.
- Agent: modos de operacao, multi-agent, streaming, persistencia.
- Facebook Ads: OAuth, sync, insights, campanhas, rate limit.
- Infra/Deploy: docker-compose, portas, healthchecks, volumes.
- Configuration: variaveis de ambiente, exemplos e responsabilidades.
- Observabilidade: logs estruturados, trace middleware, health.
- Runbooks: operacao diaria, incidentes comuns, comandos.
- Testing: pytest e rotinas de teste.
- Contributing: padroes de pastas, boas praticas e fluxo de mudancas.

## Criterios de qualidade
- Navegacao simples e indice com links claros.
- Linguagem objetiva e orientada a fluxo (setup, uso, operacao).
- Informacoes rastreaveis ao codigo existente.
- Evitar duplicacao; preferir referencias cruzadas.

## Proximos passos
1. Gerar todos os arquivos em `docs/` conforme a estrutura.
2. Atualizar `docs/README.md` como indice central.
3. Revisar consistencia tecnica com o codigo atual.
