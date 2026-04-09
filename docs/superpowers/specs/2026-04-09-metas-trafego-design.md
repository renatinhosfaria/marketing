# Design — Metas de Trafego Pago

**Data:** 2026-04-09
**Status:** Aprovado
**Escopo:** Definir e preencher metas de trafego pago da Fama Negocios Imobiliarios

---

## Contexto

A Fama opera trafego pago via Meta Ads com foco em MCMV/imoveis economicos. Captacao 100% online, modelo figital.

### Historico 2026 (dados reais do Meta Ads)

| Periodo | Gasto | Leads | CPL |
|---|---|---|---|
| Q1 (Jan-Mar) | R$ 9.718,86 | 819 | R$ 11,87 |
| Jan+Fev | R$ 6.455,00 | 534 | R$ 12,09 |
| Marco | R$ 3.263,86 | 285 | R$ 11,45 |
| Abril (1-9) | R$ 930,33 | 49 | R$ 18,99 |

Media mensal Q1: ~R$ 3.240/mes, ~273 leads/mes, CPL ~R$ 11,87.

---

## Decisoes

### Metas Abril 2026

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

---

## Arquitetura

### Arquivo de metas

- **Arquivo:** `config/trafego-metas.md`
- **Funcao:** Fonte unica de verdade das metas de trafego
- **Formato:** Uma secao por mes com tabela de metas absolutas + taxas de conversao
- **Evolucao:** Ao mudar de mes, adicionar nova secao e manter historico dos meses anteriores

### Diretorio de relatorios

- **Diretorio:** `reports/semanal/`
- **Funcao:** Reservado para relatorios semanais futuros
- **Status:** Criado vazio, sem implementacao de geracao por enquanto

---

## Fora de escopo (por enquanto)

- Geracao automatica de relatorios semanais
- Template de relatorio semanal
- Logica de comparacao metas vs realizado
- Alertas automaticos de desvio
- Monitoramento diario

---

## Implementacao

1. Preencher `config/trafego-metas.md` com metas de Abril 2026
2. Criar diretorio `reports/semanal/` vazio pra uso futuro
