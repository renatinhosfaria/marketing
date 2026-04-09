# fama-ads — Documentacao Tecnica

**Versao:** 1.0  
**Data:** 2026-04-09  
**Autor:** Renato Faria  
**Status:** Producao

---

## Visao Geral

O fama-ads e um agent de gestao de trafego pago desenvolvido para a Fama Negocios Imobiliarios. Opera como consultor especializado em Meta Ads, analisando campanhas, cruzando dados com o CRM e recomendando otimizacoes — sempre com aprovacao explicita antes de executar qualquer acao.

### Problema que resolve

A Fama opera trafego pago 100% online para captacao de leads MCMV. Antes do fama-ads, toda gestao era manual via Gerenciador de Anuncios do Meta: sem visao consolidada, otimizacao reativa (so quando sobrava tempo), e sem cruzamento entre investimento em ads e qualidade real dos leads no CRM.

### Proposta de valor

- Visao consolidada de performance (gasto, leads, CPL) em segundos
- Cruzamento automatico entre Meta Ads e CRM — CPL barato com lead ruim nao passa
- Recomendacoes priorizadas com acao, impacto e risco
- Criacao assistida de campanhas com validacao de orcamento
- Monitoramento com thresholds de alerta configurados

---

## Arquitetura

### Estrutura de arquivos

```
.claude/
  agents/
    fama-ads.md                  <- entrypoint (claude --agent fama-ads)
    fama-ads/                    <- workspace (source of truth)
      USER.md                    <- perfil do usuario e regras de comunicacao
      IDENTITY.md                <- identidade, missao, escopo e limites
      SOUL.md                    <- personalidade, tom de voz, principios
      AGENTS.md                  <- instrucoes operacionais, fluxos, regras de negocio
      TOOLS.md                   <- ferramentas MCP curadas com referencias
      HEARTBEAT.md               <- rotina de monitoramento e thresholds
      MEMORY.md                  <- indice de memorias persistentes
  skills/
    fama-ads/
      SKILL.md                   <- skill interativa (/fama-ads)
```

### Principio de design: source of truth unica

Os 7 documentos do workspace sao a unica fonte de verdade. Tanto a skill quanto o entrypoint do agent sao bootstrappers minimalistas que carregam esses documentos. Editar qualquer doc do workspace reflete automaticamente nos dois caminhos de invocacao.

```
/fama-ads (skill)          ---+
                               +---> .claude/agents/fama-ads/ (7 docs)
claude --agent (agent)     ---+          + config/METAS.md
```

---

## Invocacao

### Skill interativa

```
/fama-ads
```

Uso no dia a dia dentro do Claude Code. Carrega o contexto completo e aguarda instrucao. Exemplos:

- `/fama-ads` + "checkup" — resumo do dia com alertas
- `/fama-ads` + "como ta o CPL?" — consulta rapida de metricas
- `/fama-ads` + "cria campanha pro Residencial X" — criacao assistida

### Agent programatico

```bash
claude --agent fama-ads
```

Invocavel via terminal ou Agent tool. Mesmo comportamento da skill. Preparado para automacoes futuras via cron e hooks.

---

## Nivel de Autonomia

**Modo: Consultor**

O fama-ads nunca executa acoes no Meta Ads sem aprovacao explicita. O fluxo e:

1. Agent analisa dados e identifica oportunidades/problemas
2. Agent apresenta recomendacao com acao, impacto esperado e risco
3. Renato aprova ou ajusta
4. Agent executa a acao aprovada
5. Agent confirma execucao e registra em MEMORY.md

---

## Fluxos de Trabalho

### 1. Resumo de Performance

Visao consolidada para um periodo (diario, semanal, mensal).

| Passo | Acao | Ferramenta |
|-------|------|------------|
| 1 | Consultar metricas gerais da conta | `meta_get_account_insights` |
| 2 | Listar campanhas e status | `meta_list_campaigns` |
| 3 | Detalhar performance por campanha | `meta_get_campaign_insights` |
| 4 | Cruzar com funil do CRM | `lead_pipeline` + `lead_sources` |
| 5 | Comparar com metas vigentes | `config/METAS.md` |
| 6 | Apresentar desvios e status vs meta | — |

### 2. Checkup de Otimizacao

Analise profunda com recomendacoes priorizadas.

| Passo | Acao |
|-------|------|
| 1 | Executar fluxo de resumo |
| 2 | Identificar campanhas com CPL acima da meta |
| 3 | Identificar campanhas com gasto alto e poucos leads |
| 4 | Identificar anuncios com CTR abaixo de 1% |
| 5 | Cruzar com CRM: leads que nao converteram |
| 6 | Rankear problemas por impacto no orcamento |
| 7 | Apresentar 1 recomendacao principal + secundarias |
| 8 | Para cada: acao, impacto esperado, risco |

### 3. Criacao de Campanha

Criacao assistida com validacao de orcamento.

| Passo | Acao |
|-------|------|
| 1 | Coletar: objetivo, empreendimento, publico, orcamento |
| 2 | Validar orcamento contra teto mensal |
| 3 | Montar estrutura: campanha > conjunto > anuncio |
| 4 | Apresentar preview completo |
| 5 | Aguardar aprovacao explicita |
| 6 | Criar com status PAUSED |
| 7 | Pedir aprovacao para ativar |

### 4. Acao de Otimizacao

Execucao de recomendacao aprovada.

| Passo | Acao |
|-------|------|
| 1 | Confirmar acao e ID do objeto |
| 2 | Executar (pausar, ajustar orcamento, etc) |
| 3 | Confirmar execucao com novo status |
| 4 | Registrar em MEMORY.md |

---

## Monitoramento

### Checkup Diario

Formato padrao de saida quando solicitado:

```
Checkup [data]

Gasto hoje: R$ XX,XX | Acumulado mes: R$ XX,XX / R$ [teto]
Leads hoje: XX | Acumulado mes: XX / [meta]
CPL hoje: R$ XX,XX | CPL mes: R$ XX,XX (meta: R$ [CPL max])

Campanhas ativas: X
[tabela por campanha: nome, gasto, leads, CPL]

Alertas:
- [desvios identificados]

Proxima acao recomendada:
- [acao prioritaria]
```

### Checkup Semanal

Inclui tudo do diario mais:
- Tendencia de CPL (semana atual vs anterior)
- Funil completo do CRM (lead > agendamento > visita > venda)
- Analise de fontes de lead por qualidade
- Ranking de campanhas por eficiencia
- 1 recomendacao principal + ate 2 secundarias

### Thresholds de Alerta

| Situacao | Nivel | Acao |
|----------|-------|------|
| CPL > 1.5x meta | Atencao | Informar e monitorar |
| CPL > 2x meta | Critico | Recomendar pausar |
| CTR < 1% | Atencao | Sugerir troca de criativo |
| Gasto mensal > 80% do teto | Atencao | Alertar ritmo de gasto |
| Gasto mensal > 95% do teto | Critico | Recomendar pausar campanhas |
| Campanha sem leads em 3 dias | Critico | Recomendar pausar |

---

## Fontes de Dados

### Meta Ads (leitura + escrita com aprovacao)

Referencia completa: `MCP´s/MCP-META-ADS.md`

Subconjunto curado de 53 ferramentas disponiveis:

| Categoria | Ferramentas | Permissao |
|-----------|-------------|----------|
| Consulta | account_insights, list_campaigns, campaign_insights, get_insights, list_adsets, list_ads, get_ad | Livre |
| Otimizacao | update_campaign, update_adset, update_ad | Com aprovacao |
| Criacao | create_campaign, create_adset, create_ad, create_ad_creative | Com aprovacao |
| Pesquisa | search_ad_library | Livre |

**Ferramentas proibidas:** delete_campaign, delete_adset, delete_ad (destrutivas e irreversiveis).

### CRM Postgres (somente leitura)

Referencia completa: `MCP´s/MCP-CRM-POSTGRES.md`

| Ferramenta | Uso |
|------------|-----|
| search_leads | Volume de leads por fonte e periodo |
| lead_pipeline | Funil de conversao por etapa |
| lead_sources | Qualidade por fonte (score medio) |
| client_timeline | Historico completo de um lead |
| broker_performance | Taxa de conversao dos corretores |
| daily_report | Resumo diario: leads, agendamentos, vendas |

**Nenhuma escrita no CRM e permitida.**

---

## Regras de Negocio

| Regra | Valor |
|-------|-------|
| Conta de anuncio | Famachat (`act_24036721645944375`) |
| Objetivos de campanha | `OUTCOME_LEADS` (formulario) e `OUTCOME_ENGAGEMENT` (WhatsApp) |
| Status de campanha nova | Sempre `PAUSED` |
| Orcamento diario padrao (WhatsApp) | R$ 25,00 |
| Metas vigentes | `config/METAS.md` (atualizado mensalmente) |

---

## Personalidade e Comunicacao

O fama-ads se comporta como um analista de trafego senior: direto, pragmatico, sem enrolacao. Cinco principios guiam suas respostas:

1. **Dado antes de opiniao** — toda recomendacao ancorada em metrica
2. **Orcamento e sagrado** — nunca sugerir gasto sem justificativa com projecao
3. **CPL nao e vaidade** — sempre cruzar com qualidade real do lead (CRM)
4. **Simplicidade** — preferir a opcao mais simples de executar
5. **Priorizar implacavelmente** — 1 recomendacao principal, o resto e secundario

Formato: tabelas para comparativos, bullets para recomendacoes, sempre fecha com "proxima acao recomendada".

---

## Memorias Persistentes

O arquivo `MEMORY.md` funciona como indice de memorias entre sessoes. Sempre que o agent executa uma acao aprovada (pausa campanha, ajusta orcamento, cria anuncio), registra:

- O que foi feito
- Por que foi feito
- Resultado observado

Isso permite que sessoes futuras tenham contexto das decisoes passadas sem depender do historico de conversa.

---

## Dependencias Externas

| Recurso | Caminho | Funcao |
|---------|---------|--------|
| Metas de trafego | `config/METAS.md` | Metas vigentes (orcamento, CPL, leads, conversao) |
| Doc Meta Ads | `MCP´s/MCP-META-ADS.md` | Referencia das 53 ferramentas |
| Doc CRM | `MCP´s/MCP-CRM-POSTGRES.md` | Referencia das 35 ferramentas |

---

## Riscos Conhecidos

| Risco | Mitigacao |
|-------|-----------|
| Rate limit da API do Meta Ads | Fluxos otimizados para minimizar chamadas redundantes |
| CRM desatualizado (Reno/formulario falha em registrar lead) | Agent reporta discrepancias entre Meta e CRM |
| Metas genericas (macro, nao por empreendimento) | Suficiente para operacao atual. Reavaliar se operacao crescer |
| Memorias acumulando sem curadoria | Revisao periodica do MEMORY.md para remover entradas obsoletas |
| METAS.md desatualizado para novo mes | Agent deve alertar se nao encontrar metas pro mes vigente |

---

## Evolucao Futura

Capacidades planejadas para versoes futuras (nao implementadas):

- **Automacao via cron/hooks** — checkup diario automatico com alerta
- **Metas por empreendimento** — quando operacao crescer
- **Integracao MinIO** — gestao de criativos (imagens, videos) direto no agent
- **Relatorios semanais automaticos** — geracao e armazenamento em `reports/`
