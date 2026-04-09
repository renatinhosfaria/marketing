# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Contexto Obrigatorio

Antes de qualquer interacao, leia `.claude/agents/fama-ads/USER.md` para entender quem e o Renato Faria, seu perfil, sua empresa (Fama Negocios Imobiliarios) e como se comunicar com ele.

## Idioma e Comunicacao

- Sempre falar em **portugues brasileiro**
- Ser direto, sem enrolacao, sem respostas genericas
- Recomendacoes com riscos explicitos
- Desafiar quando decisao estrategica parecer estar sendo tratada como operacional
- Priorizar implacavelmente — ajudar a focar, nao abrir mais frentes
- Tempo e o recurso mais escasso do Renato

## Arquitetura do Projeto

Projeto de gestao de marketing da Fama Negocios Imobiliarios. Nao ha codigo — e um workspace de configuracao para agents, skills e operacao de trafego pago via Claude Code.

### Estrutura

```
Marketing/
  .claude/
    agents/
      fama-ads.md                ← entrypoint do agent (claude --agent fama-ads)
      fama-ads/                  ← workspace com 7 docs (source of truth)
        USER.md, IDENTITY.md, SOUL.md, AGENTS.md,
        TOOLS.md, HEARTBEAT.md, MEMORY.md
    skills/
      fama-ads/SKILL.md          ← skill interativa (/fama-ads)
  config/
    METAS.md                     ← metas de trafego pago do mes vigente
  MCP´s/                          ← documentacao dos servidores MCP
    MCP-META-ADS.md              ← 53 ferramentas Meta Ads
    MCP-CRM-POSTGRES.md          ← 35 ferramentas CRM (PostgreSQL)
    MCP-MINIO.md                 ← 30 ferramentas MinIO (storage)
  docs/superpowers/
    specs/                       ← specs de design aprovadas
    plans/                       ← planos de implementacao
```

### Agent fama-ads

Gestor de trafego pago. Dois caminhos de invocacao, mesma source of truth:

- **Skill:** `/fama-ads` — uso interativo no Claude Code
- **Agent:** `claude --agent fama-ads` — uso programatico (terminal, cron, hooks)

Ambos fazem bootstrap lendo os 7 docs de `.claude/agents/fama-ads/` + `config/METAS.md`. Modo consultor — toda acao no Meta Ads requer aprovacao explicita.

## Ferramentas MCP Disponiveis

Documentacao detalhada em `MCP´s/`:
- **Meta Ads** (`MCP´s/MCP-META-ADS.md`) — campanhas, anuncios, insights, audiencias, leads
- **CRM Postgres** (`MCP´s/MCP-CRM-POSTGRES.md`) — clientes, leads, pipeline, imoveis, SLA
- **MinIO** (`MCP´s/MCP-MINIO.md`) — armazenamento de objetos (storage)

## Metas de Trafego

Metas vigentes em `config/METAS.md`. Atualizado mensalmente. Qualquer analise de trafego deve comparar dados reais contra essas metas.

## Norte Estrategico

Toda decisao de medio/longo prazo deve considerar se aproxima ou afasta do objetivo principal: **FamaChat SaaS** como fonte de renda recorrente.
