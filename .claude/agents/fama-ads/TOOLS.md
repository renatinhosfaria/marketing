# Ferramentas — fama-ads

## Meta Ads (leitura + escrita com aprovacao)

Referencia completa: `MCP's/MCP-META-ADS.md`

### Uso frequente
| Ferramenta | Quando usar |
|------------|-------------|
| `meta_get_account_insights` | Resumo geral: gasto, impressoes, cliques, CPL |
| `meta_list_campaigns` | Listar campanhas e status (ativas/pausadas) |
| `meta_get_campaign_insights` | Performance detalhada por campanha |
| `meta_get_insights` | Metricas de qualquer objeto (conta, campanha, adset, ad) |
| `meta_list_adsets` | Conjuntos de uma campanha |
| `meta_list_ads` | Anuncios de um conjunto ou campanha |
| `meta_get_ad` | Detalhes de anuncio (inclui diagnostico de reprovacao) |

### Otimizacao (requer aprovacao)
| Ferramenta | Quando usar |
|------------|-------------|
| `meta_update_campaign` | Pausar/ativar campanha, ajustar orcamento |
| `meta_update_adset` | Ajustar segmentacao, orcamento do conjunto |
| `meta_update_ad` | Pausar/ativar anuncio, trocar criativo |

### Criacao (requer aprovacao)
| Ferramenta | Quando usar |
|------------|-------------|
| `meta_create_campaign` | Criar campanha (sempre PAUSED) |
| `meta_create_adset` | Criar conjunto de anuncios |
| `meta_create_ad` | Criar anuncio |
| `meta_create_ad_creative` | Criar criativo |

### Pesquisa
| Ferramenta | Quando usar |
|------------|-------------|
| `meta_search_ad_library` | Espionar concorrentes na biblioteca publica |

## CRM Postgres (somente leitura)

Referencia completa: `MCP's/MCP-CRM-POSTGRES.md`

| Ferramenta | Quando usar |
|------------|-------------|
| `search_leads` | Buscar leads por fonte, status, periodo |
| `lead_pipeline` | Funil: quantos leads em cada etapa |
| `lead_sources` | Qualidade por fonte (score medio, volume) |
| `client_timeline` | Historico de um lead especifico |
| `broker_performance` | Performance dos corretores (conversao) |
| `daily_report` | Resumo diario: leads, agendamentos, vendas |

## Ferramentas que NAO usar
- Qualquer ferramenta de escrita no CRM (create_task, add_client_note, update_task)
- MinIO (fora do escopo)
- Ferramentas destrutivas: `meta_delete_campaign`, `meta_delete_adset`, `meta_delete_ad`
