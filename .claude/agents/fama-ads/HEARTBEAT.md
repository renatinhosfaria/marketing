# Rotina de Monitoramento — fama-ads

## Checkup diario

Quando Renato pedir um checkup ou resumo do dia:

1. **Gasto do dia** — `meta_get_account_insights` (date_preset: today)
2. **Gasto acumulado no mes** — `meta_get_account_insights` (since: primeiro dia do mes)
3. **Campanhas ativas** — `meta_list_campaigns` (status_filter: ACTIVE)
4. **CPL do dia vs meta** — comparar com `config/METAS.md`
5. **Leads no CRM** — `daily_report` (date: hoje)
6. **Alertas** — qualquer campanha com CPL > 1.5x meta

### Formato de saida

```
Checkup [data]

Gasto hoje: R$ XX,XX | Acumulado mes: R$ XX,XX / R$ [teto de METAS.md]
Leads hoje: XX | Acumulado mes: XX / [meta de leads de METAS.md]
CPL hoje: R$ XX,XX | CPL mes: R$ XX,XX (meta: R$ [CPL maximo de METAS.md])

Campanhas ativas: X
[tabela com nome, gasto, leads, CPL de cada campanha ativa]

Alertas:
- [campanha X com CPL acima da meta]

Proxima acao recomendada:
- [acao prioritaria]
```

## Checkup semanal

Analise mais profunda, incluindo tendencias:

1. Tudo do checkup diario, mas com periodo de 7 dias
2. **Tendencia de CPL** — comparar semana atual vs anterior
3. **Funil CRM** — `lead_pipeline` para ver conversao real
4. **Fontes de lead** — `lead_sources` para qualidade por canal
5. **Performance por campanha** — rankear por eficiencia (CPL + conversao)
6. **Recomendacoes** — 1 principal, ate 2 secundarias

## Thresholds de alerta

| Situacao | Nivel | Acao |
|----------|-------|------|
| CPL > 1.5x meta | Atencao | Informar, monitorar |
| CPL > 2x meta | Critico | Recomendar pausar |
| CTR < 1% | Atencao | Sugerir troca de criativo |
| Gasto mensal > 80% do teto | Atencao | Alertar ritmo de gasto |
| Gasto mensal > 95% do teto | Critico | Recomendar pausar campanhas |
| Campanha ativa sem leads em 3 dias | Critico | Recomendar pausar |
