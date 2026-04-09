# Gestao de Trafego Pago — Plano de Implementacao

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Criar a estrutura de skills e configuracao para gestao de trafego pago do Meta Ads dentro do Claude Code.

**Architecture:** Skills locais do Claude Code em `.claude/skills/`, cada uma com seu SKILL.md. Arquivo de metas em `config/trafego-metas.md`. Skills criadas em branco (estrutura apenas), conteudo preenchido em fase posterior.

**Tech Stack:** Claude Code skills (Markdown), MCP Meta Ads, MCP CRM Postgres

**Spec:** `docs/superpowers/specs/2026-04-09-gestao-trafego-pago-design.md`

---

## Estrutura de arquivos

```
Marketing/
  .claude/
    skills/
      trafego-resumo/
        SKILL.md             -> /trafego-resumo (skill em branco)
      trafego-checkup/
        SKILL.md             -> /trafego-checkup (skill em branco)
      trafego-criar/
        SKILL.md             -> /trafego-criar (skill em branco, fase futura)
  config/
    trafego-metas.md         -> metas e orcamento da operacao (em branco)
```

---

### Task 1: Criar skill /trafego-resumo

**Files:**
- Create: `.claude/skills/trafego-resumo/SKILL.md`

- [ ] **Step 1: Criar diretorio da skill**

```bash
mkdir -p ".claude/skills/trafego-resumo"
```

- [ ] **Step 2: Criar SKILL.md com frontmatter**

Criar `.claude/skills/trafego-resumo/SKILL.md` com o seguinte conteudo:

```markdown
---
name: trafego-resumo
description: Visao consolidada das campanhas de trafego pago do Meta Ads — gasto, leads, CPL, performance por campanha
---

<!-- Conteudo da skill sera definido em fase posterior -->
<!-- Esta skill deve: -->
<!-- 1. Ler config/trafego-metas.md para obter metas e regras -->
<!-- 2. Consultar MCP Meta Ads: campanhas ativas, gastos, impressoes, cliques, leads, CPL -->
<!-- 3. Consultar MCP CRM Postgres: leads recebidos, status do lead -->
<!-- 4. Comparar metricas com metas definidas -->
<!-- 5. Apresentar resumo organizado com destaques de melhor/pior performance -->
<!-- Aceita parametros: hoje (padrao), semana, mes -->
```

- [ ] **Step 3: Verificar que a skill aparece**

```bash
ls -la ".claude/skills/trafego-resumo/SKILL.md"
```

Expected: arquivo existe com conteudo correto.

- [ ] **Step 4: Commit**

```bash
git add .claude/skills/trafego-resumo/SKILL.md
git commit -m "feat: criar skill /trafego-resumo (estrutura)"
```

---

### Task 2: Criar skill /trafego-checkup

**Files:**
- Create: `.claude/skills/trafego-checkup/SKILL.md`

- [ ] **Step 1: Criar diretorio da skill**

```bash
mkdir -p ".claude/skills/trafego-checkup"
```

- [ ] **Step 2: Criar SKILL.md com frontmatter**

Criar `.claude/skills/trafego-checkup/SKILL.md` com o seguinte conteudo:

```markdown
---
name: trafego-checkup
description: Analise de otimizacao das campanhas de trafego pago — identifica problemas e sugere acoes com opcao de executar
---

<!-- Conteudo da skill sera definido em fase posterior -->
<!-- Esta skill deve: -->
<!-- 1. Ler config/trafego-metas.md para obter metas e regras -->
<!-- 2. Consultar MCP Meta Ads: metricas detalhadas por campanha, conjunto e anuncio -->
<!-- 3. Consultar MCP CRM Postgres: qualidade dos leads (visitas, vendas) -->
<!-- 4. Identificar problemas: CPL alto, CTR baixo, criativo cansado, orcamento mal distribuido -->
<!-- 5. Sugerir acoes concretas e oferecer executar via MCP (com confirmacao do usuario) -->
```

- [ ] **Step 3: Verificar que a skill aparece**

```bash
ls -la ".claude/skills/trafego-checkup/SKILL.md"
```

Expected: arquivo existe com conteudo correto.

- [ ] **Step 4: Commit**

```bash
git add .claude/skills/trafego-checkup/SKILL.md
git commit -m "feat: criar skill /trafego-checkup (estrutura)"
```

---

### Task 3: Criar skill /trafego-criar

**Files:**
- Create: `.claude/skills/trafego-criar/SKILL.md`

- [ ] **Step 1: Criar diretorio da skill**

```bash
mkdir -p ".claude/skills/trafego-criar"
```

- [ ] **Step 2: Criar SKILL.md com frontmatter**

Criar `.claude/skills/trafego-criar/SKILL.md` com o seguinte conteudo:

```markdown
---
name: trafego-criar
description: Criacao rapida de campanhas, conjuntos e anuncios no Meta Ads com padroes da Fama
---

<!-- Conteudo da skill sera definido em fase posterior (menor prioridade) -->
<!-- Esta skill deve: -->
<!-- 1. Ler config/trafego-metas.md para obter padroes e limites -->
<!-- 2. Guiar o usuario na criacao de campanha via conversa -->
<!-- 3. Usar MCP Meta Ads: meta_create_campaign, meta_create_adset, meta_create_ad -->
<!-- 4. Aplicar nomenclatura padrao da Fama: [ANO-MES] [TIPO] Empreendimento | Construtora -->
<!-- 5. Confirmar cada etapa antes de executar -->
```

- [ ] **Step 3: Verificar que a skill aparece**

```bash
ls -la ".claude/skills/trafego-criar/SKILL.md"
```

Expected: arquivo existe com conteudo correto.

- [ ] **Step 4: Commit**

```bash
git add .claude/skills/trafego-criar/SKILL.md
git commit -m "feat: criar skill /trafego-criar (estrutura)"
```

---

### Task 4: Criar arquivo de metas

**Files:**
- Create: `config/trafego-metas.md`

- [ ] **Step 1: Criar diretorio config**

```bash
mkdir -p "config"
```

- [ ] **Step 2: Criar trafego-metas.md**

Criar `config/trafego-metas.md` com o seguinte conteudo:

```markdown
# Metas de Trafego Pago — Fama Negocios Imobiliarios

<!-- Preencher com as metas macro da operacao -->
<!-- Exemplos de metas a definir: -->
<!-- - Orcamento mensal total de trafego -->
<!-- - CPL maximo aceitavel (geral) -->
<!-- - Meta de leads por mes -->
<!-- - Meta de visitas agendadas por mes -->
<!-- - Meta de vendas por mes -->
```

- [ ] **Step 3: Verificar que o arquivo existe**

```bash
ls -la "config/trafego-metas.md"
```

Expected: arquivo existe com conteudo correto.

- [ ] **Step 4: Commit**

```bash
git add config/trafego-metas.md
git commit -m "feat: criar arquivo de metas de trafego (estrutura)"
```

---

### Task 5: Inicializar repositorio git (se necessario)

**Pre-condicao:** O projeto Marketing nao e um repositorio git ainda.

**Files:**
- Create: `.gitignore`

- [ ] **Step 1: Verificar se ja e um repositorio git**

```bash
git status
```

Se ja for repositorio, pular para o Step 4.

- [ ] **Step 2: Inicializar repositorio**

```bash
git init
```

- [ ] **Step 3: Criar .gitignore**

Criar `.gitignore` com o seguinte conteudo:

```
# Claude Code
.claude/settings.local.json

# OS
.DS_Store
Thumbs.db
```

- [ ] **Step 4: Commit inicial (se repositorio novo)**

```bash
git add .
git commit -m "feat: inicializar projeto Marketing com estrutura de gestao de trafego"
```

Nota: Se o repositorio ja existia, os commits individuais das Tasks 1-4 ja foram feitos.
