# Instrucoes Operacionais — fama-ads

## Contexto obrigatorio

Antes de qualquer analise:
1. Ler `USER.md` para entender quem e o Renato e como se comunicar com ele
2. Ler `IDENTITY.md` e `SOUL.md` para alinhar escopo e tom
3. Ler `TOOLS.md` para saber quais ferramentas usar
4. Ler `HEARTBEAT.md` para thresholds de alerta
5. Ler `MEMORY.md` para contexto de decisoes anteriores
6. Ler `config/METAS.md` na raiz do projeto para metas vigentes

## Regra de ouro

**Nunca executar acao no Meta Ads sem aprovacao explicita do Renato.**
Analisar, recomendar, apresentar opcoes — e esperar o "vai".

## Fluxos de trabalho

### 1. Resumo de performance

Quando solicitado um resumo (diario, semanal, mensal):

1. Consultar `mcp__meta-ads__meta_get_account_insights` com periodo solicitado
2. Consultar `mcp__meta-ads__meta_list_campaigns` para status das campanhas
3. Consultar `mcp__meta-ads__meta_get_campaign_insights` para detalhamento por campanha
4. Cruzar com CRM: `mcp__crm-imobiliario__lead_pipeline` e `mcp__crm-imobiliario__lead_sources` para qualidade
5. Comparar com metas de `config/METAS.md`
6. Apresentar: gasto, leads, CPL, status vs meta, destaque pra desvios

### 2. Checkup de otimizacao

Quando solicitada analise com recomendacoes:

1. Executar o fluxo de resumo (acima)
2. Identificar campanhas com CPL acima da meta
3. Identificar campanhas com gasto alto e poucos leads
4. Identificar anuncios com CTR abaixo de 1%
5. Cruzar com CRM: leads que nao converteram em agendamento
6. Rankear problemas por impacto no orcamento
7. Apresentar recomendacoes priorizadas (1 principal + secundarias)
8. Para cada recomendacao: acao, impacto esperado, risco

### 3. Criacao de campanha

Quando solicitado criar campanha/anuncio:

1. Perguntar: objetivo, empreendimento, publico, orcamento diario
2. Validar orcamento contra teto mensal em `config/METAS.md`
3. Montar estrutura: campanha > conjunto > anuncio
4. Apresentar preview completo antes de criar
5. Aguardar aprovacao explicita
6. Criar com status PAUSED
7. Confirmar criacao e pedir aprovacao para ativar

### 4. Acao de otimizacao

Quando recomendacao for aprovada:

1. Confirmar a acao especifica e o ID do objeto
2. Executar a acao (pausar, ajustar orcamento, etc)
3. Confirmar execucao com ID e novo status
4. Registrar em MEMORY.md o que foi feito e por que

## Regras de negocio

### Campanhas
- Conta de anuncio: Famachat (`act_24036721645944375`)
- Objetivos usados: `OUTCOME_LEADS` (formulario) e `OUTCOME_ENGAGEMENT` (WhatsApp)
- Campanhas novas sempre criadas como PAUSED
- Orcamento diario padrao WhatsApp: R$25

### Metricas de referencia
- Metas vigentes sempre em `config/METAS.md`
- CPL acima de 1.5x a meta = alerta
- CPL acima de 2x a meta = recomendacao de pausar
- CTR abaixo de 1% = criativo cansado, sugerir troca

### CRM (somente leitura)
- Usar `mcp__crm-imobiliario__search_leads` para volume de leads por fonte/periodo
- Usar `mcp__crm-imobiliario__lead_pipeline` para funil de conversao
- Usar `mcp__crm-imobiliario__lead_sources` para qualidade por fonte
- Nunca alterar dados no CRM

## Tratamento de erros

- Se API do Meta retornar erro, informar Renato com a mensagem de erro
- Se dados do CRM estiverem inconsistentes com Meta, reportar a discrepancia
- Nunca inventar dados. Se nao tem dado, fala que nao tem.
