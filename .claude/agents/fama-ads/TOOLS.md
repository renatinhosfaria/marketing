# Ferramentas — fama-ads

## Meta Ads (leitura + escrita com aprovacao)

Referencia completa: `MCP's/MCP-META-ADS.md`

### Uso frequente
| Ferramenta | Quando usar |
|------------|-------------|
| `mcp__meta-ads__meta_get_account_insights` | Resumo geral: gasto, impressoes, cliques, CPL |
| `mcp__meta-ads__meta_list_campaigns` | Listar campanhas e status (ativas/pausadas) |
| `mcp__meta-ads__meta_get_campaign_insights` | Performance detalhada por campanha |
| `mcp__meta-ads__meta_get_insights` | Metricas de qualquer objeto (conta, campanha, adset, ad) |
| `mcp__meta-ads__meta_list_adsets` | Conjuntos de uma campanha |
| `mcp__meta-ads__meta_list_ads` | Anuncios de um conjunto ou campanha |
| `mcp__meta-ads__meta_get_ad` | Detalhes de anuncio (inclui diagnostico de reprovacao) |

### Otimizacao (requer aprovacao)
| Ferramenta | Quando usar |
|------------|-------------|
| `mcp__meta-ads__meta_update_campaign` | Pausar/ativar campanha, ajustar orcamento |
| `mcp__meta-ads__meta_update_adset` | Ajustar segmentacao, orcamento do conjunto |
| `mcp__meta-ads__meta_update_ad` | Pausar/ativar anuncio, trocar criativo |

### Criacao (requer aprovacao)
| Ferramenta | Quando usar |
|------------|-------------|
| `mcp__meta-ads__meta_create_campaign` | Criar campanha (sempre PAUSED) |
| `mcp__meta-ads__meta_create_adset` | Criar conjunto de anuncios |
| `mcp__meta-ads__meta_create_ad` | Criar anuncio |
| `mcp__meta-ads__meta_create_ad_creative` | Criar criativo |

### Pesquisa
| Ferramenta | Quando usar |
|------------|-------------|
| `mcp__meta-ads__meta_search_ad_library` | Espionar concorrentes na biblioteca publica |

## CRM Postgres (somente leitura)

Referencia completa: `MCP's/MCP-CRM-POSTGRES.md`

| Ferramenta | Quando usar |
|------------|-------------|
| `mcp__crm-imobiliario__search_leads` | Buscar leads por fonte, status, periodo |
| `mcp__crm-imobiliario__lead_pipeline` | Funil: quantos leads em cada etapa |
| `mcp__crm-imobiliario__lead_sources` | Qualidade por fonte (score medio, volume) |
| `mcp__crm-imobiliario__client_timeline` | Historico de um lead especifico |
| `mcp__crm-imobiliario__broker_performance` | Performance dos corretores (conversao) |
| `mcp__crm-imobiliario__daily_report` | Resumo diario: leads, agendamentos, vendas |

## Ferramentas que NAO usar
- Qualquer ferramenta de escrita no CRM (create_task, add_client_note, update_task)
- MinIO (fora do escopo)
- Ferramentas destrutivas: `mcp__meta-ads__meta_delete_campaign`, `mcp__meta-ads__meta_delete_adset`, `mcp__meta-ads__meta_delete_ad`
