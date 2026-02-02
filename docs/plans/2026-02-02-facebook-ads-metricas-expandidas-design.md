# Design: Expansão de Métricas do Facebook Ads

**Data:** 2026-02-02
**Status:** Proposta

## Contexto

O sistema atual busca ~25 campos de insights da Meta API, 5 breakdowns padrão e 2 action breakdowns. A análise da documentação da Meta Marketing API revelou dezenas de métricas, breakdowns e indicadores adicionais que podem melhorar significativamente a capacidade de otimização e diagnóstico do sistema.

## Estado Atual

### Métricas buscadas hoje
- **Alcance:** impressions, reach, frequency
- **Cliques:** clicks, unique_clicks, inline_link_clicks, outbound_clicks
- **Custo:** spend, cpc, cpm
- **Eficiência:** ctr
- **Conversão:** conversions, conversion_values, leads, cost_per_lead
- **Vídeo:** video_30_sec_watched_actions, video_p100_watched_actions
- **Engajamento:** post_engagement, post_reactions, post_comments, post_shares
- **Raw:** actions, action_values, cost_per_action_type

### Breakdowns atuais
- **Padrão:** age, gender, country, publisher_platform, device_platform
- **Action:** action_type, action_device

### Métricas calculadas (metrics_calculator.py)
- CTR, CPC, CPM, CPL, Frequency

### Problema identificado
- O campo `cpp` é pedido na API mas não é salvo no banco

---

## Tier 1 — Alta Prioridade (Alto Impacto, Baixo Esforço)

### 1.1 Diagnósticos de Qualidade do Anúncio

| Campo | Tipo | O que mede |
|-------|------|-----------|
| `quality_ranking` | String (enum) | Qualidade percebida vs concorrentes |
| `engagement_rate_ranking` | String (enum) | Taxa de engajamento esperada vs concorrentes |
| `conversion_rate_ranking` | String (enum) | Taxa de conversão esperada vs concorrentes |

**Valores possíveis:** `BELOW_AVERAGE_10`, `BELOW_AVERAGE_20`, `BELOW_AVERAGE_35`, `AVERAGE`, `ABOVE_AVERAGE`

**Restrições:** Disponível apenas no nível do anúncio com 500+ impressões.

**Impacto:** Permite diagnóstico automático — "criativo fraco" vs "landing page não converte" vs "público não engaja". Alimenta diretamente o agente de recomendações.

### 1.2 ROAS

| Campo | Tipo | O que mede |
|-------|------|-----------|
| `purchase_roas` | List[AdsActionStats] | Retorno sobre investimento em ads |
| `website_purchase_roas` | List[AdsActionStats] | ROAS de compras via pixel do site |

**Impacto:** Métrica mais importante para e-commerce. O sistema já armazena `conversion_values` mas não calcula o ratio.

### 1.3 Métricas de Custo Granulares

| Campo | Tipo | O que mede |
|-------|------|-----------|
| `cost_per_unique_click` | Decimal | Custo por clique único — elimina inflação de repetidos |
| `cost_per_inline_link_click` | Decimal | Custo por clique no link — ignora cliques em curtir/comentar |
| `cost_per_outbound_click` | List[AdsActionStats] | Custo por clique que sai do Facebook |
| `cost_per_thruplay` | List[AdsActionStats] | Custo por ThruPlay (vídeo completo ou 15s) |

### 1.4 CTRs Específicos

| Campo | Fórmula | O que mede |
|-------|---------|-----------|
| `unique_ctr` | unique_clicks / reach | CTR sem inflação de frequência |
| `inline_link_click_ctr` | inline_link_clicks / impressions | CTR só de cliques no link |

### 1.5 Funil Completo de Vídeo

| Campo | Ponto do funil |
|-------|---------------|
| `video_play_actions` | Início do vídeo (play) |
| `video_15_sec_watched_actions` | 15s (definição de ThruPlay) |
| `video_p25_watched_actions` | 25% assistido |
| `video_p50_watched_actions` | 50% assistido |
| `video_p75_watched_actions` | 75% assistido |
| `video_p95_watched_actions` | 95% assistido |
| `video_avg_time_watched_actions` | Tempo médio assistido |
| `video_thruplay_watched_actions` | ThruPlays (completo ou 15s) |

**Impacto:** Identifica exatamente onde as pessoas abandonam o vídeo. Poucos plays = thumbnail ruim. Queda em 25% = hook fraco.

### 1.6 Salvar CPP

O campo `cpp` já é buscado na API mas não existe coluna no banco. Adicionar coluna `cpp Numeric(15,4)` nas tabelas de insights.

- **CPM** = spend / impressions × 1000 (pode contar a mesma pessoa várias vezes)
- **CPP** = spend / reach × 1000 (pessoas únicas)

---

## Tier 2 — Média Prioridade

### 2.1 Breakdowns Padrão

| Breakdown | O que faz | Caso de uso |
|-----------|----------|------------|
| `platform_position` | Feed, Stories, Reels, Right Column, Marketplace | Performance por posicionamento — o mais importante que falta |
| `frequency_value` | Performance por frequência (1x, 2x, 3x...) | Identifica fadiga de frequência |
| `hourly_stats_aggregated_by_advertiser_time_zone` | Performance por hora do dia | Day-parting |
| `region` | Região/estado dentro do país | Otimização geográfica sub-nacional |
| `impression_device` | Dispositivo real (iPhone, Android, desktop) | Diferente de device_platform |

### 2.2 Action Breakdowns

| Action Breakdown | O que faz |
|-----------------|----------|
| `action_destination` | Para onde a pessoa foi (site, app, messenger) |
| `action_video_sound` | Vídeo assistido com ou sem som |
| `action_carousel_card_id` / `action_carousel_card_name` | Qual card do carrossel gerou a ação |
| `action_reaction` | Tipo de reação (Like, Love, Haha, Sad, Angry) |

### 2.3 Brand Awareness e Landing Page

| Campo | O que mede |
|-------|-----------|
| `estimated_ad_recallers` | Pessoas que lembrariam do anúncio em 2 dias |
| `estimated_ad_recall_rate` | % do alcance que lembra do anúncio |
| `landing_page_view` (via actions) | Visualizações da landing page após clique |

### 2.4 Métricas Únicas/Deduplicadas

| Campo | O que mede |
|-------|-----------|
| `unique_inline_link_clicks` | Cliques únicos no link |
| `unique_outbound_clicks` | Cliques únicos de saída |
| `unique_link_clicks_ctr` | CTR único de link |
| `unique_conversions` | Conversões únicas |
| `cost_per_unique_conversion` | Custo por conversão única (verdadeiro CAC) |
| `cost_per_unique_outbound_click` | Custo por clique único de saída |
| `cost_per_inline_post_engagement` | Custo por engajamento |

### 2.5 CTRs Adicionais

| Campo | O que mede |
|-------|-----------|
| `outbound_clicks_ctr` | % de impressões que resultaram em saída do Facebook |
| `unique_inline_link_click_ctr` | CTR de link, único, por alcance |
| `unique_outbound_clicks_ctr` | CTR de saída, único |

---

## Tier 3 — Condicional (Adicionar Quando Necessário)

### 3.1 Dynamic Creative / Advantage+ (se usar DCO)

| Breakdown | O que testa |
|-----------|------------|
| `body_asset` | Variante de texto do corpo |
| `title_asset` | Variante de título |
| `image_asset` | Variante de imagem |
| `video_asset` | Variante de vídeo |
| `call_to_action_asset` | Variante de CTA |
| `description_asset` | Variante de descrição |
| `link_url_asset` | Variante de URL |
| `ad_format_asset` | Variante de formato |

### 3.2 Catálogo / E-commerce (se usar Dynamic Ads)

| Campo | O que mede |
|-------|-----------|
| `catalog_segment_actions` | Ações por segmento do catálogo |
| `catalog_segment_value` | Valor por segmento |
| `converted_product_quantity` | Quantidade de produtos nas conversões |
| `converted_product_value` | Valor dos produtos nas conversões |

### 3.3 Auction & Competitiveness

| Campo | O que mede |
|-------|-----------|
| `auction_bid` | Seu bid no leilão |
| `auction_competitiveness` | Competitividade do anúncio |
| `auction_max_competitor_bid` | Bid máximo dos concorrentes |

---

## Mudanças da API a Considerar (2025-2026)

| Mudança | Impacto | Status atual |
|---------|---------|-------------|
| Remoção de `7d_view` e `28d_view` (Jan 2026) | Sistema já correto | OK |
| Reach + breakdowns limitado a 13 meses | Queries históricas com age/gender/country podem falhar | Verificar |
| Unified Attribution (Jun 2025) | Conversões on-Facebook contam na impressão | Verificar consistência |
| API v23.0 (Mai 2025) | Última versão estável | Verificar versão usada |

---

## Impacto na Arquitetura

### Alterações necessárias por componente

1. **Client (`projects/facebook_ads/client/`):**
   - Expandir `INSIGHT_FIELDS` com novos campos
   - Adicionar novos breakdowns e action breakdowns nas queries

2. **Models (`shared/db/models/famachat_readonly.py`):**
   - Adicionar colunas nas tabelas `InsightsHistory` e `InsightsToday`
   - Criar tabelas de breakdown se necessário (para platform_position, frequency_value, etc.)

3. **Sync Services (`projects/facebook_ads/services/`):**
   - Atualizar `sync_insights.py` para processar novos campos
   - Adicionar lógica de extração para métricas de vídeo e rankings

4. **Metrics Calculator (`projects/facebook_ads/utils/metrics_calculator.py`):**
   - Expandir extração de actions para novos tipos
   - Adicionar cálculos derivados (ROAS, CPP se não vier da API)

5. **Schemas (`projects/facebook_ads/schemas/`):**
   - Adicionar novos campos nos schemas de response

6. **API (`projects/facebook_ads/api/insights.py`):**
   - Expor novas métricas nos endpoints existentes
   - Considerar novo endpoint para breakdowns avançados

7. **Frontend (`frontend/`):**
   - Atualizar tipos TypeScript
   - Adicionar visualizações para funil de vídeo, rankings, breakdowns

8. **Alembic Migration:**
   - Nova migration para colunas adicionais

9. **Agente de IA (`projects/agent/`):**
   - Atualizar tools e prompts para considerar novas métricas nas análises e recomendações
