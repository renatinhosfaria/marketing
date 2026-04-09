# MCP Meta Ads - Documentacao Completa de Ferramentas

> Referencia detalhada de todas as ferramentas do servidor MCP meta-ads para Meta Ads.
>
> **Total: 53 ferramentas** organizadas em 11 categorias.

---

## Sumario

1. Conta de Anuncios
2. Campanhas
3. Ad Sets
4. Anuncios
5. Criativos
6. Imagens e Videos
7. Audiencias
8. Leads
9. Insights
10. Conversions API
11. Biblioteca de Anuncios

---

## 1. Conta de Anuncios

### 1.1 meta_list_ad_accounts
Lista contas acessiveis pelo token. Retorna ID, nome, status, moeda, saldo, gasto.

| Parametro | Tipo | Obrigatorio | Descricao |
|-----------|------|:-----------:|-----------|
| business_id | string | Nao | ID do Business Manager |
| limit | integer | Nao | Max (padrao: 50, max: 500) |

### 1.2 meta_get_ad_account
Detalhes: saldo, gasto, limite, moeda, fuso, status.
- ad_account_id (Opcional)

### 1.3 meta_get_account_insights
Resumo performance: gasto, impressoes, alcance, cliques, CTR, CPM, CPC.

| Parametro | Tipo | Obrigatorio | Descricao |
|-----------|------|:-----------:|-----------|
| ad_account_id | string | Nao | ID da conta |
| date_preset | enum | Nao | today, yesterday, last_3d, last_7d, last_14d, last_28d, last_30d, last_90d, this_month, last_month, etc |
| since / until | string | Nao | Datas YYYY-MM-DD |
| level | enum | Nao | account, campaign, adset, ad |
| breakdowns | array | Nao | age, gender, country, region, publisher_platform, etc |
| fields | string | Nao | Metricas (padrao: spend, impressions, reach, clicks, ctr, cpm, cpc) |
| limit | integer | Nao | Max linhas (padrao: 100) |

---

## 2. Campanhas

### 2.1 meta_create_campaign
Cria campanha. Objetivos: OUTCOME_AWARENESS, OUTCOME_ENGAGEMENT, OUTCOME_LEADS, OUTCOME_SALES, OUTCOME_TRAFFIC, OUTCOME_APP_PROMOTION

| Parametro | Tipo | Obrigatorio | Descricao |
|-----------|------|:-----------:|-----------|
| name | string | **Sim** | Nome (max: 400) |
| objective | enum | **Sim** | Objetivo |
| status | enum | Nao | ACTIVE ou PAUSED (padrao) |
| daily_budget | integer | Nao | Centavos (5000=R$50). Exclusivo com lifetime_budget |
| lifetime_budget | integer | Nao | Centavos |
| bid_strategy | enum | Nao | LOWEST_COST_WITHOUT_CAP, LOWEST_COST_WITH_BID_CAP, COST_CAP, LOWEST_COST_WITH_MIN_ROAS |
| special_ad_categories | array | Nao | NONE, EMPLOYMENT, HOUSING, CREDIT, etc |
| spend_cap | integer | Nao | Limite (centavos) |
| start_time / stop_time | string | Nao | ISO 8601 |
| validate_only | boolean | Nao | Validar sem criar |

### 2.2 meta_get_campaign - campaign_id (**Obrigatorio**)
### 2.3 meta_list_campaigns - ad_account_id, status_filter, limit
### 2.4 meta_update_campaign - campaign_id (**Obrigatorio**) + name, status, daily_budget, lifetime_budget, bid_strategy, spend_cap, start_time, stop_time
### 2.5 meta_delete_campaign - **IRREVERSIVEL** - campaign_id (**Obrigatorio**)
### 2.6 meta_get_campaign_insights - campaign_ids, ad_account_id, date_preset, since/until, level, breakdowns, fields, limit

---

## 3. Conjuntos de Anuncios - Ad Sets

### 3.1 meta_create_adset

| Parametro | Tipo | Obrigatorio | Descricao |
|-----------|------|:-----------:|-----------|
| campaign_id | string | **Sim** | Campanha pai |
| name | string | **Sim** | Nome (max: 400) |
| optimization_goal | string | **Sim** | LINK_CLICKS, REACH, LEAD_GENERATION, OFFSITE_CONVERSIONS, etc |
| billing_event | enum | **Sim** | IMPRESSIONS, LINK_CLICKS, APP_INSTALLS, VIDEO_VIEWS, THRUPLAY |
| targeting | JSON | **Sim** | Segmentacao: age_min, age_max, geo_locations, interests |
| daily_budget | integer | Nao | Centavos |
| lifetime_budget | integer | Nao | Centavos |
| bid_strategy / bid_amount | varies | Nao | Estrategia e lance |
| start_time / end_time | string | Nao | ISO 8601 |
| status | enum | Nao | ACTIVE ou PAUSED |

### 3.2 meta_get_adset - adset_id (**Obrigatorio**)
### 3.3 meta_list_adsets - ad_account_id, campaign_id, status_filter, limit
### 3.4 meta_update_adset - adset_id (**Obrigatorio**) + campos a alterar
### 3.5 meta_delete_adset - **IRREVERSIVEL** - adset_id (**Obrigatorio**)

---

## 4. Anuncios - Ads

### 4.1 meta_create_ad

| Parametro | Tipo | Obrigatorio | Descricao |
|-----------|------|:-----------:|-----------|
| adset_id | string | **Sim** | Ad set pai |
| name | string | **Sim** | Nome (max: 400) |
| creative | JSON | **Sim** | {creative_id: ID} ou object_story_spec inline |
| status | enum | Nao | ACTIVE ou PAUSED |
| bid_amount | integer | Nao | Centavos |
| conversion_domain | string | Nao | Dominio |
| tracking_specs | JSON | Nao | Rastreamento |

### 4.2 meta_get_ad - ad_id (**Obrigatorio**) - Diagnostico reprovacoes
### 4.3 meta_list_ads - ad_account_id, campaign_id, adset_id, status_filter, limit
### 4.4 meta_update_ad - ad_id (**Obrigatorio**) + name, status, creative, bid_amount, conversion_domain
### 4.5 meta_delete_ad - **IRREVERSIVEL** - ad_id (**Obrigatorio**)
### 4.6 meta_get_ad_preview - ad_id (**Obrigatorio**), ad_format (padrao: DESKTOP_FEED_STANDARD)

---

## 5. Criativos - Ad Creatives

### 5.1 meta_create_ad_creative

| Parametro | Tipo | Obrigatorio | Descricao |
|-----------|------|:-----------:|-----------|
| name | string | **Sim** | Nome (max: 100) |
| body | string | Nao | Texto principal |
| title | string | Nao | Titulo |
| link_url | URI | Nao | URL destino |
| image_hash | string | Nao | Hash imagem |
| image_url | URI | Nao | URL imagem |
| video_id | string | Nao | ID video |
| page_id | string | Nao | Page ID |
| instagram_user_id | string | Nao | Instagram ID |
| call_to_action_type | string | Nao | LEARN_MORE, SIGN_UP, SHOP_NOW |
| url_tags | string | Nao | UTMs |
| object_story_spec | JSON | Nao | Spec completo |
| asset_feed_spec | JSON | Nao | DCO/variantes |
| creative_json | JSON | Nao | Avancado |

### 5.2 meta_get_ad_creative - creative_id (**Obrigatorio**)
### 5.3 meta_list_ad_creatives - ad_account_id, limit
### 5.4 meta_update_ad_creative - creative_id + name (ambos **Obrigatorios**)
### 5.5 meta_get_creative_preview - creative_id (**Obrigatorio**), ad_format

---

## 6. Imagens e Videos

### 6.1 meta_upload_ad_image - Upload via file_path, file_url ou base64_data
### 6.2 meta_list_ad_images - ad_account_id, limit
### 6.3 meta_get_ad_image - Busca por ID ou hash
### 6.4 meta_upload_ad_video - Via file_path, file_url ou base64_data + title, description
### 6.5 meta_list_ad_videos - ad_account_id, limit
### 6.6 meta_get_ad_video - video_id (**Obrigatorio**)
### 6.7 meta_get_ad_video_status - Status processamento - video_id (**Obrigatorio**)

---

## 7. Audiencias - Audiences

### 7.1 meta_create_custom_audience

| Parametro | Tipo | Obrigatorio | Descricao |
|-----------|------|:-----------:|-----------|
| name | string | **Sim** | Nome (max: 100) |
| subtype | string | Nao | CUSTOM, WEBSITE, ENGAGEMENT |
| rule | JSON | Nao | Regra remarketing |
| retention_days | integer | Nao | 1-180 dias |
| customer_file_source | string | Nao | USER_PROVIDED_ONLY, PARTNER_PROVIDED_ONLY, BOTH |

### 7.2 meta_create_lookalike_audience - name + origin_audience_id + country (**Obrigatorios**), ratio (0.01-0.20)
### 7.3 meta_get_audience - audience_id (**Obrigatorio**)
### 7.4 meta_list_audiences - ad_account_id, limit, subtype_filter
### 7.5 meta_update_audience - audience_id (**Obrigatorio**) + name, description, retention_days, rule
### 7.6 meta_delete_audience - audience_id (**Obrigatorio**)
### 7.7 meta_add_audience_users - audience_id + schema + users (**Obrigatorios**)
### 7.8 meta_remove_audience_users - Mesmos parametros
### 7.9 meta_replace_audience_users - Substitui TODOS os membros

---

## 8. Leads e Formularios

### 8.1 meta_list_lead_forms - page_id (**Obrigatorio**), limit
### 8.2 meta_get_lead_form - form_id (**Obrigatorio**)
### 8.3 meta_list_form_leads - form_id (**Obrigatorio**), limit
### 8.4 meta_get_lead - lead_id (**Obrigatorio**)
### 8.5 meta_download_form_leads - Exporta CSV - form_id (**Obrigatorio**), limit (padrao: 200, max: 500)

---

## 9. Insights e Metricas

### 9.1 meta_get_insights - Ferramenta UNIVERSAL

| Parametro | Tipo | Obrigatorio | Descricao |
|-----------|------|:-----------:|-----------|
| object_id | string | **Sim** | ID: conta (act_XXX), campanha, ad set ou anuncio |
| date_preset | enum | Nao | Periodo |
| since / until | string | Nao | Datas |
| level | enum | Nao | account, campaign, adset, ad |
| breakdowns | array | Nao | age, gender, country, publisher_platform, etc |
| fields | string | Nao | Metricas |
| limit | integer | Nao | Max (padrao: 100) |

### 9.2 meta_get_account_insights -> Ver secao 1.3
### 9.3 meta_get_campaign_insights -> Ver secao 2.6

---

## 10. Conversions API

### 10.1 meta_send_conversion_event

| Parametro | Tipo | Obrigatorio | Descricao |
|-----------|------|:-----------:|-----------|
| pixel_id | string | **Sim** | ID do Pixel |
| event_name | string | **Sim** | Purchase, Lead, ViewContent, AddToCart, CompleteRegistration |
| event_time | integer | **Sim** | Timestamp Unix |
| user_data | JSON | Nao | Email, telefone (hashed) |
| custom_data | JSON | Nao | Valor, moeda |
| event_source_url | URI | Nao | URL origem |
| event_id | string | Nao | Deduplicacao |
| action_source | string | Nao | website, system_generated, app |
| test_event_code | string | Nao | Codigo teste |

### 10.2 meta_send_conversion_events_batch - pixel_id + events (JSON array)
### 10.3 meta_validate_conversion_payload - Valida SEM enviar - events (JSON array)

---

## 11. Biblioteca de Anuncios

### 11.1 meta_search_ad_library
Pesquisa na biblioteca publica da Meta (Ads Archive). Espionar concorrentes.

---

## Referencia Rapida

### Hierarquia


### Valores em Centavos
| Valor | Centavos |
|-------|----------|
| R$10 | 1000 |
| R$50 | 5000 |
| R$100 | 10000 |
| R$500 | 50000 |

### Objetivos de Campanha
| Objetivo | Uso |
|----------|-----|
| OUTCOME_AWARENESS | Alcance |
| OUTCOME_ENGAGEMENT | Engajamento |
| OUTCOME_TRAFFIC | Trafego |
| OUTCOME_LEADS | Leads |
| OUTCOME_SALES | Vendas |
| OUTCOME_APP_PROMOTION | Apps |

---
> Documento gerado em 09/04/2026 com base nos schemas do MCP meta-ads.
