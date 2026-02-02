# Facebook Ads Hard Delete Design

## Goal
Adicionar exclusao definitiva de conexoes do Facebook Ads ao endpoint `DELETE /config/{config_id}` quando `hardDelete=true`, removendo a configuracao e todos os dados relacionados no banco de dados (campanhas, adsets, ads, insights, historico de sync, logs do modulo, dados de ML e conversas do agente).

## Architecture
A rota existente permanece com soft delete por padrao. Quando `hardDelete=true`, o backend executa uma sequencia de deletes explicitos dentro de uma transacao usando `AsyncSession`. A exclusao e ordenada dos dados mais dependentes para a configuracao. A resposta retorna sucesso imediato; se houver erro, a transacao e revertida e a requisicao falha.

## Components
- API: `projects/facebook_ads/api/config_endpoints.py` (adicionar parametro `hardDelete`, caminho de hard delete, logs).
- FB Ads data: `shared/db/models/famachat_readonly.py` (config, campaigns, adsets, ads, insights today/history).
- FB Ads module: `projects/facebook_ads/models/sync.py` (sync history), `projects/facebook_ads/models/management.py` (management/rate limit logs).
- ML data: `projects/ml/db/models.py` (trained models, predictions, classifications, recommendations, anomalies, features, forecasts, training jobs).
- Agent data: `projects/agent/db/models.py` (conversations) + cleanup por thread_id para checkpoints/writes.
- Tests: novo teste unitario cobrindo a lista de deletes e o fluxo hard delete.

## Data Flow
1) Recebe `DELETE /config/{id}?hardDelete=true`.
2) Valida existencia da configuracao.
3) Executa deletes por `config_id` em todas as tabelas relacionadas (incluindo subqueries para apagar checkpoints/writes por thread_id).
4) Remove a configuracao.
5) Commit e resposta de sucesso.

## Error Handling
- `404` se a configuracao nao existir.
- Em erro durante delete: rollback automatico e retorno `500` (via excecao).
- Soft delete continua inalterado quando `hardDelete` nao e enviado.

## Testing
- Teste unitario para verificar que a lista de deletes inclui todas as tabelas esperadas e respeita a ordem.
- Teste unitario para validar que o endpoint executa o fluxo hard delete e retorna sucesso.
