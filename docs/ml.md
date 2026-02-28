# Machine Learning

## Previsoes

O modulo de previsao atende casos de uso de CPL e leads com endpoints dedicados e processamento em lote.

Capacidades:

- predicao pontual por entidade;
- series historicas para analise;
- validacao periodica de previsoes passadas.

## Classificacoes

Classificacao de entidades (campanha/adset/ad) em tiers de performance.

Fluxos principais:

- inferencia de classe em tempo de requisicao;
- treino de classificadores em jobs;
- coleta de feedback para ajuste de modelo.

## Recomendacoes

Recomendacoes orientadas por regras e sinais de modelos.

Fluxos principais:

- geracao de recomendacoes por entidade;
- resumo por nivel de severidade/prioridade;
- acoes de dismiss/apply para controle operacional.

## Anomalias e treino

- deteccao de outliers por janelas historicas;
- treino agendado de detectores;
- calibracao e retreinamento continuo no Celery Beat.
