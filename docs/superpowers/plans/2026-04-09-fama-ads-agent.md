# Agent fama-ads — Plano de Implementacao

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preencher os 6 documentos do workspace do agent fama-ads, criar o entrypoint `.claude/agents/fama-ads.md` e a skill `/fama-ads` para invocacao interativa.

**Architecture:** Documentos markdown que configuram um agent de gestao de trafego pago (Meta Ads) com modo consultor. O workspace do agent vive em `.claude/agents/fama-ads/` (6 docs), o entrypoint em `.claude/agents/fama-ads.md`, e a skill em `.claude/skills/fama-ads/SKILL.md`. Ambos leem do mesmo workspace — source of truth unica.

**Tech Stack:** Claude Code agents, Claude Code skills, MCP Meta Ads, MCP CRM Postgres

**Spec:** `docs/superpowers/specs/2026-04-09-fama-ads-agent-design.md`

---

### Task 1: IDENTITY.md

**Files:**
- Modify: `.claude/agents/fama-ads/IDENTITY.md` (arquivo vazio)

- [ ] **Step 1: Escrever IDENTITY.md**

Conteudo aprovado na spec, secao "Documento 1: IDENTITY.md". Inclui: nome, papel, missao, escopo de atuacao, fora de escopo, quem comanda.

- [ ] **Step 2: Commit**

```bash
git add .claude/agents/fama-ads/IDENTITY.md
git commit -m "feat(fama-ads): definir identidade do agent"
```

---

### Task 2: SOUL.md

**Files:**
- Modify: `.claude/agents/fama-ads/SOUL.md` (arquivo vazio)

- [ ] **Step 1: Escrever SOUL.md**

Conteudo aprovado na spec, secao "Documento 2: SOUL.md". Inclui: personalidade, tom de voz, 5 principios, o que nunca fazer, formato de apresentacao.

- [ ] **Step 2: Commit**

```bash
git add .claude/agents/fama-ads/SOUL.md
git commit -m "feat(fama-ads): definir personalidade e tom de voz"
```

---

### Task 3: AGENTS.md

**Files:**
- Modify: `.claude/agents/fama-ads/AGENTS.md` (arquivo vazio)

- [ ] **Step 1: Escrever AGENTS.md**

Conteudo aprovado na spec, secao "Documento 3: AGENTS.md". Inclui:
- Contexto obrigatorio (ordem de leitura dos arquivos)
- Regra de ouro (nunca executar sem aprovacao)
- 4 fluxos: resumo, checkup, criacao de campanha, acao de otimizacao
- Regras de negocio: conta `act_24036721645944375`, objetivos, thresholds
- CRM somente leitura
- Tratamento de erros

- [ ] **Step 2: Commit**

```bash
git add .claude/agents/fama-ads/AGENTS.md
git commit -m "feat(fama-ads): definir instrucoes operacionais e fluxos"
```

---

### Task 4: TOOLS.md

**Files:**
- Modify: `.claude/agents/fama-ads/TOOLS.md` (arquivo vazio)

- [ ] **Step 1: Escrever TOOLS.md**

Conteudo aprovado na spec, secao "Documento 4: TOOLS.md". Inclui:
- Meta Ads: referencia externa + tabelas de uso frequente, otimizacao, criacao, pesquisa
- CRM Postgres: referencia externa + tabela de ferramentas permitidas
- Lista explicita de ferramentas proibidas

- [ ] **Step 2: Commit**

```bash
git add .claude/agents/fama-ads/TOOLS.md
git commit -m "feat(fama-ads): definir ferramentas MCP curadas"
```

---

### Task 5: HEARTBEAT.md

**Files:**
- Modify: `.claude/agents/fama-ads/HEARTBEAT.md` (arquivo vazio)

- [ ] **Step 1: Escrever HEARTBEAT.md**

Conteudo aprovado na spec, secao "Documento 5: HEARTBEAT.md". Inclui:
- Checkup diario: 6 passos + formato de saida
- Checkup semanal: diario + tendencias + recomendacoes
- Tabela de thresholds de alerta (6 situacoes com nivel e acao)

- [ ] **Step 2: Commit**

```bash
git add .claude/agents/fama-ads/HEARTBEAT.md
git commit -m "feat(fama-ads): definir rotina de monitoramento e thresholds"
```

---

### Task 6: MEMORY.md

**Files:**
- Modify: `.claude/agents/fama-ads/MEMORY.md` (arquivo vazio)

- [ ] **Step 1: Escrever MEMORY.md**

Indice vazio com descricao do proposito. Memorias serao adicionadas conforme o agent opera.

- [ ] **Step 2: Commit**

```bash
git add .claude/agents/fama-ads/MEMORY.md
git commit -m "feat(fama-ads): inicializar indice de memorias"
```

---

### Task 7: Skill /fama-ads

**Files:**
- Create: `.claude/skills/fama-ads/SKILL.md`

- [ ] **Step 1: Criar diretorio da skill**

```bash
mkdir -p .claude/skills/fama-ads
```

- [ ] **Step 2: Escrever SKILL.md**

Arquivo com frontmatter (name, description) e instrucoes para:
1. Carregar os 6 documentos do agent na ordem
2. Carregar `config/METAS.md`
3. Apresentar-se brevemente
4. Aguardar instrucao do usuario

Conteudo aprovado na spec, secao "Invocacao > Skill interativa".

- [ ] **Step 3: Commit**

```bash
git add .claude/skills/fama-ads/SKILL.md
git commit -m "feat(fama-ads): criar skill /fama-ads para invocacao interativa"
```

---

### Task 8: Entrypoint do agent

**Files:**
- Create: `.claude/agents/fama-ads.md`

- [ ] **Step 1: Escrever fama-ads.md (entrypoint)**

Arquivo `.md` unico na raiz de `.claude/agents/` que o Claude Code reconhece como agent. Contem instrucoes para fazer bootstrap lendo os 6 docs do workspace:

```markdown
---
name: fama-ads
description: Gestor de trafego pago da Fama Negocios Imobiliarios. Analisa campanhas Meta Ads, cruza com CRM, recomenda otimizacoes. Modo consultor.
---

# fama-ads

Voce e o fama-ads, gestor de trafego pago da Fama Negocios Imobiliarios.

## Bootstrap

Antes de qualquer interacao, leia os seguintes arquivos na ordem:

1. `.claude/agents/fama-ads/IDENTITY.md`
2. `.claude/agents/fama-ads/SOUL.md`
3. `.claude/agents/fama-ads/AGENTS.md`
4. `.claude/agents/fama-ads/TOOLS.md`
5. `.claude/agents/fama-ads/HEARTBEAT.md`
6. `.claude/agents/fama-ads/MEMORY.md`
7. `config/METAS.md`

Siga rigorosamente as instrucoes de AGENTS.md e o tom de SOUL.md.
Apresente-se brevemente e aguarde instrucao do Renato.
```

- [ ] **Step 2: Commit**

```bash
git add .claude/agents/fama-ads.md
git commit -m "feat(fama-ads): criar entrypoint do agent"
```

---

### Task 9: Limpeza

**Files:**
- Delete: `.claude/agents/fama-ads/CONFIG.md` (se existir)
- Delete: `.claude/agents/fama-ads/scripts/` (diretorio vazio)
- Delete: `.claude/agents/fama-ads/skills/` (diretorio vazio)

- [ ] **Step 1: Remover arquivos e diretorios desnecessarios**

```bash
rm -f .claude/agents/fama-ads/CONFIG.md
rmdir .claude/agents/fama-ads/scripts 2>/dev/null || true
rmdir .claude/agents/fama-ads/skills 2>/dev/null || true
```

- [ ] **Step 2: Verificar estrutura final**

```bash
find .claude/agents/fama-ads/ -type f
find .claude/skills/fama-ads/ -type f
```

Esperado:
```
.claude/agents/fama-ads/IDENTITY.md
.claude/agents/fama-ads/SOUL.md
.claude/agents/fama-ads/AGENTS.md
.claude/agents/fama-ads/TOOLS.md
.claude/agents/fama-ads/HEARTBEAT.md
.claude/agents/fama-ads/MEMORY.md
.claude/skills/fama-ads/SKILL.md
```

- [ ] **Step 3: Commit de limpeza (se houve mudancas)**

```bash
git add -A .claude/agents/fama-ads/
git commit -m "chore(fama-ads): remover CONFIG.md e diretorios vazios"
```

---

### Task 10: Validacao

- [ ] **Step 1: Testar invocacao da skill**

Invocar `/fama-ads` no Claude Code e verificar que:
- Carrega os 6 documentos + METAS.md
- Se apresenta brevemente
- Aguarda instrucao

- [ ] **Step 2: Testar agent via terminal**

```bash
claude --agent fama-ads
```

Verificar que:
- O entrypoint `.claude/agents/fama-ads.md` e reconhecido
- Faz bootstrap lendo os 6 docs do workspace
- Mesmo comportamento da skill
