# Metas de Trafego Pago — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preencher as metas de trafego pago de Abril 2026 e criar diretorio para relatorios semanais futuros.

**Architecture:** Arquivo unico `config/trafego-metas.md` como fonte de verdade das metas. Diretorio `reports/semanal/` reservado para uso futuro.

**Tech Stack:** Markdown, Git

**Spec:** `superpowers/specs/2026-04-09-metas-trafego-design.md`

---

### Task 1: Preencher metas de Abril 2026

**Files:**
- Modify: `config/trafego-metas.md` (reescrever conteudo completo)

- [ ] **Step 1: Substituir conteudo de `config/trafego-metas.md`**

Substituir todo o conteudo do arquivo por:

```markdown
# Metas de Trafego Pago — Fama Negocios Imobiliarios

## Abril 2026

| Meta | Valor |
|---|---|
| Orcamento mensal (teto) | R$ 3.000,00 |
| CPL maximo | R$ 12,00 |
| Leads | 250 |
| Agendamentos | 30 |
| Visitas realizadas | 20 |
| Vendas | 5 |

### Taxas de conversao implicitas

- Lead -> Agendamento: 12%
- Agendamento -> Visita: 67%
- Visita -> Venda: 25%
- Lead -> Venda: 2%

### Referencia historica (Q1 2026)

| Periodo | Gasto | Leads | CPL |
|---|---|---|---|
| Q1 (Jan-Mar) | R$ 9.718,86 | 819 | R$ 11,87 |
| Jan+Fev | R$ 6.455,00 | 534 | R$ 12,09 |
| Marco | R$ 3.263,86 | 285 | R$ 11,45 |
```

- [ ] **Step 2: Verificar conteudo**

Run: `cat config/trafego-metas.md`
Expected: Conteudo com metas de Abril 2026 e referencia historica Q1.

- [ ] **Step 3: Commit**

```bash
git add config/trafego-metas.md
git commit -m "feat: definir metas de trafego pago Abril 2026"
```

---

### Task 2: Criar diretorio de relatorios semanais

**Files:**
- Create: `reports/semanal/.gitkeep`

- [ ] **Step 1: Criar diretorio com .gitkeep**

```bash
mkdir -p reports/semanal
touch reports/semanal/.gitkeep
```

- [ ] **Step 2: Verificar criacao**

Run: `ls -la reports/semanal/`
Expected: Diretorio existe com arquivo `.gitkeep`.

- [ ] **Step 3: Commit**

```bash
git add reports/semanal/.gitkeep
git commit -m "feat: criar diretorio para relatorios semanais"
```
