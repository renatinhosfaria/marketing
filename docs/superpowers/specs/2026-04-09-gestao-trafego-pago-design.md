# Gestao de Trafego Pago — Design

**Data:** 2026-04-09
**Autor:** Renato Faria + Claude
**Status:** Aprovado

---

## Contexto

A Fama Negocios Imobiliarios opera trafego pago no Meta Ads para captacao de leads (formulario e WhatsApp). Hoje Renato gerencia tudo manualmente pelo Gerenciador de Anuncios. O objetivo e trazer essa gestao para dentro do Claude Code, com skills especializadas que consultam dados via MCP e apresentam informacao acionavel.

### Situacao atual

- 1 conta de anuncio: Famachat (`act_24036721645944375`)
- 21 campanhas (8 ativas, 13 pausadas)
- 2 tipos de objetivo: `OUTCOME_LEADS` (formulario) e `OUTCOME_ENGAGEMENT` (mensagem WhatsApp)
- Orcamento diario das campanhas ativas de WhatsApp: R$25 cada
- Carteira de empreendimentos relativamente estavel
- Leads de formulario caem automaticamente no FamaChat
- Leads de WhatsApp sao atendidos pelo agente Reno (OpenClaw), que registra no FamaChat automaticamente

### Problema

Falta de visao consolidada, otimizacao reativa (so quando sobra tempo), e criacao manual de campanhas. Tudo exige entrar no Gerenciador de Anuncios.

---

## Decisoes de design

### Abordagem escolhida: Skills + Comandos

Criar skills do Claude Code — cada uma e um comando especializado que consulta Meta Ads e CRM via MCP, processa os dados e apresenta resultado acionavel.

**Alternativas descartadas:**
- Agente autonomo 24/7 — overengineering pro momento, custo alto de tokens e manutencao
- Dashboard web — foge do escopo "dentro do Claude Code", tempo de desenvolvimento alto

### Nivel de autonomia

O sistema sugere acoes e oferece executar. Renato confirma antes de qualquer mudanca. Nada roda sem aprovacao explicita.

### Metas e configuracao

Metas macro da operacao (orcamento, CPL, volume) ficam em arquivo no projeto (`config/trafego-metas.md`). Skills leem esse arquivo antes de qualquer analise. Metas sao gerais, nao por empreendimento.

### Automacoes (fase futura)

Hooks e cron para execucao automatica das skills ficam para uma fase posterior. Por agora, tudo e manual via comando.

---

## Arquitetura

### Estrutura de arquivos

```
Marketing/
  skills/
    trafego-resumo.md        -> /trafego-resumo
    trafego-checkup.md       -> /trafego-checkup
    trafego-criar.md         -> /trafego-criar (fase futura)
  config/
    trafego-metas.md         -> metas e orcamento da operacao
```

### Skills

| Skill | Funcao | Consulta |
|-------|--------|----------|
| `/trafego-resumo` | Foto atual: gasto, leads, CPL, performance por campanha | Meta Ads + CRM + config |
| `/trafego-checkup` | Analise + recomendacoes de acao (pausar, ajustar, trocar criativo) | Meta Ads + CRM + config |
| `/trafego-criar` | Criacao rapida de campanhas (fase futura) | Meta Ads + config |

### Fluxo de dados

```
Usuario digita o comando
        |
        v
Skill le config/trafego-metas.md (metas e regras)
        |
        v
Skill consulta MCP Meta Ads (campanhas, gastos, metricas)
        |
        v
Skill consulta MCP CRM Postgres (leads, status, conversao)
        |
        v
Claude processa, compara com metas, apresenta resultado
        |
        v
Usuario decide e age (conversando)
```

### Ferramentas MCP utilizadas

**Meta Ads:**
- `meta_list_campaigns` — listar campanhas
- `meta_get_insights` — metricas de performance
- `meta_list_ads` — listar anuncios
- `meta_get_ad` — detalhes de anuncio
- `meta_list_adsets` — listar conjuntos
- `meta_create_campaign`, `meta_create_adset`, `meta_create_ad` — criacao (fase futura)
- `meta_update_ad`, `meta_update_adset`, `meta_update_campaign` — acoes de otimizacao

**CRM Postgres:**
- `search_leads` — leads recebidos
- `client_timeline` — historico do lead
- `lead_pipeline` — funil de conversao

---

## Prioridade de implementacao

1. `/trafego-resumo` — base de tudo, visao consolidada
2. `/trafego-checkup` — otimizacao e alertas, depende do resumo
3. `config/trafego-metas.md` — arquivo de metas (conteudo definido junto com as skills)
4. `/trafego-criar` — fase futura, menor prioridade

---

## Riscos

- **Limite de API do Meta Ads:** Muitas chamadas seguidas podem bater rate limit. Skills devem ser eficientes nas consultas.
- **Dados do CRM desatualizados:** Se o Reno ou o formulario falhar em registrar um lead, o resumo vai mostrar menos leads do que realmente entraram.
- **Metas genericas demais:** Metas macro funcionam agora, mas se a operacao crescer, pode precisar de metas por empreendimento.
- **Sem automacao:** Enquanto nao tiver hooks/cron, Renato precisa lembrar de rodar os comandos. Risco de nao usar.
