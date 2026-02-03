# Visao Geral

## Objetivo

O FamaChat ML e um conjunto de microservicos para otimizar campanhas de Facebook Ads usando Machine Learning, oferecendo previsoes, classificacoes, recomendacoes, deteccao de anomalias e um Agente IA para analises conversacionais.

## Escopo

- API de ML (FastAPI) para previsoes, classificacoes e recomendacoes.
- Agente IA (FastAPI separado) com chat, streaming e multi-agent.
- Modulo Facebook Ads com OAuth, sync e insights.
- Frontend Next.js para operacao e visualizacao.
- Infra com Docker Compose, Celery e Redis.

## Nao objetivos

- Substituir o Gerenciador de Anuncios do Facebook.
- Ser uma plataforma de BI generica.
- Executar ETL externo fora do dominio de campanhas.
