# Contributing

## Fluxo de contribuicao

1. alinhar escopo da mudanca;
2. implementar alteracoes pequenas e com contexto claro;
3. adicionar/ajustar testes e documentacao;
4. abrir PR com resumo tecnico e evidencias de verificacao.

## Padroes

Padroes esperados no repositorio:

- manter separacao por dominio (`projects/ml`, `projects/facebook_ads`, `shared`);
- evitar acoplamento entre camadas de apresentacao, aplicacao e infraestrutura;
- padronizar naming, logs e tratamento de erro;
- preservar comandos reprodutiveis na documentacao.

## Checklist antes de merge

- testes relevantes executados;
- migracoes validadas quando ha mudanca de schema;
- impacto operacional descrito;
- documentacao atualizada quando comportamento muda.
