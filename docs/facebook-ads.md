# Facebook Ads

## OAuth

A autenticacao com Facebook e tratada por endpoints publicos de OAuth:

- geracao da URL de autorizacao;
- callback e finalizacao de setup;
- refresh e validacao de token.

## Sync

A sincronizacao de dados ocorre por API e por agendamento:

- sync incremental recorrente;
- sync full diario;
- endpoints de status, historico e cancelamento.

## Insights

Camada de consulta analitica para operacao:

- KPIs agregados;
- series diarias;
- rankings;
- comparativos;
- breakdowns e diagnosticos de qualidade.

## Configuracoes

Gestao de contas conectadas e metadados operacionais:

- CRUD de configuracoes por cliente;
- validacao de conectividade;
- descoberta de ad accounts associadas.

## Agent Query Tool

Ferramenta para o agente `fbads` executar consultas e operacoes SQL a partir de linguagem natural com trilha de auditoria.

Fluxo:

1. recebe `prompt` (e opcionalmente `sql`);
2. traduz `prompt` para SQL quando necessario;
3. aplica guardrails de seguranca;
4. executa no banco;
5. grava auditoria em `fbads_agent_query_audit`.

Guardrails obrigatorios:

- bloqueia `DROP`;
- bloqueia `TRUNCATE`;
- bloqueia `DELETE` sem `WHERE`;
- bloqueia `UPDATE` sem `WHERE`;
- bloqueia comentarios SQL e multiplas instrucoes.

Tabela de auditoria:

- `requested_at`
- `requested_by`
- `prompt`
- `generated_sql`
- `operation_type`
- `execution_status`
- `rows_affected`
- `duration_ms`
- `error_message`
- `metadata`

Endpoint operacional:

- `POST /api/v1/facebook-ads/agent/query`

Exemplo:

```json
{
  "prompt": "listar top 5 campanhas por spend da config 1"
}
```

Resposta de sucesso:

```json
{
  "success": true,
  "data": {
    "operationType": "SELECT",
    "sqlExecuted": "SELECT campaign_id, SUM(spend) AS spend FROM sistema_facebook_ads_insights_history WHERE config_id = 1 GROUP BY campaign_id ORDER BY spend DESC LIMIT 5",
    "rowsAffected": 5,
    "rows": [],
    "durationMs": 37
  }
}
```
