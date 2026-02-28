# Overview

## Objetivo

O projeto Marketing centraliza operacao e inteligencia para campanhas de Facebook Ads, combinando:

- ingestao e sincronizacao de dados de midia;
- analise e recomendacao orientadas por ML;
- orquestracao de tarefas assincronas para workloads pesados;
- visualizacao integrada via frontend web.

## Escopo

Escopo atual do repositorio:

- duas APIs FastAPI separadas por responsabilidade (ML e Facebook Ads);
- frontend Next.js para operacao, monitoramento e workflows;
- workers Celery e scheduler para jobs periodicos e processamento em fila;
- persistencia em PostgreSQL e mensageria/cache em Redis;
- stack de observabilidade com OpenTelemetry, Prometheus, Tempo e Grafana.

## Nao objetivos

Fora do escopo deste projeto:

- gestao financeira ou billing de plataforma de anuncios;
- criacao de campanhas diretamente em plataformas de terceiros fora dos fluxos definidos;
- data warehouse corporativo generalista;
- substituicao completa de plataformas BI externas.

## Publico alvo

- engenharia backend/frontend;
- engenharia de dados e ML;
- SRE/DevOps;
- operadores de marketing que dependem dos endpoints e dashboards.
