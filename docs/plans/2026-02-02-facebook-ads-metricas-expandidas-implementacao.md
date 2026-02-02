# Expansão de Métricas do Facebook Ads - Plano de Implementação

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Expandir as métricas, breakdowns e indicadores buscados da Meta API para cobrir diagnósticos de qualidade, ROAS, custos granulares, funil de vídeo, métricas deduplicadas, breakdowns avançados e métricas condicionais (DCO, catálogo, auction).

**Architecture:** Mudanças bottom-up: migration DB -> models -> client fields -> sync service extraction -> metrics calculator -> schemas -> API endpoints -> frontend types. Breakdowns avançados vão para uma tabela separada `sistema_facebook_ads_insights_breakdowns` para não poluir a tabela principal.

**Tech Stack:** Python 3.11, FastAPI, SQLAlchemy (async), Alembic, Pydantic, TypeScript/Next.js

---

### Task 1: Alembic Migration — Novas Colunas nas Tabelas de Insights

**Files:**
- Create: `alembic/versions/009_expand_insights_metrics.py`

**Step 1: Criar a migration**

```python
"""Expandir métricas de insights do Facebook Ads.

Revision ID: 009
Revises: 008
Create Date: 2026-02-02

Adiciona colunas para:
- Diagnósticos de qualidade (quality_ranking, engagement_rate_ranking, conversion_rate_ranking)
- ROAS (purchase_roas, website_purchase_roas)
- Custos granulares (cpp, cost_per_unique_click, cost_per_inline_link_click, cost_per_outbound_click, cost_per_thruplay)
- CTRs específicos (unique_ctr, inline_link_click_ctr, outbound_clicks_ctr)
- Funil de vídeo (video_plays, video_15s, video_p25, video_p50, video_p75, video_p95, video_thruplay, video_avg_time)
- Métricas únicas (unique_inline_link_clicks, unique_outbound_clicks, unique_conversions)
- Custos únicos (cost_per_unique_conversion, cost_per_unique_outbound_click, cost_per_inline_post_engagement)
- CTRs únicos (unique_link_clicks_ctr, unique_inline_link_click_ctr, unique_outbound_clicks_ctr)
- Brand awareness (estimated_ad_recallers, estimated_ad_recall_rate, cost_per_estimated_ad_recallers)
- Landing page views
- Catálogo (converted_product_quantity, converted_product_value)
- Action values raw (action_values JSON)
"""

from alembic import op
import sqlalchemy as sa

revision = "009"
down_revision = "008"
branch_labels = None
depends_on = None

# Colunas a adicionar em AMBAS as tabelas (insights_history e insights_today)
NEW_COLUMNS = [
    # Diagnósticos de qualidade
    ("quality_ranking", sa.String(50)),
    ("engagement_rate_ranking", sa.String(50)),
    ("conversion_rate_ranking", sa.String(50)),

    # ROAS
    ("purchase_roas", sa.Numeric(10, 4)),
    ("website_purchase_roas", sa.Numeric(10, 4)),

    # Custos granulares
    ("cpp", sa.Numeric(15, 4)),
    ("cost_per_unique_click", sa.Numeric(15, 4)),
    ("cost_per_inline_link_click", sa.Numeric(15, 4)),
    ("cost_per_outbound_click", sa.Numeric(15, 4)),
    ("cost_per_thruplay", sa.Numeric(15, 4)),

    # CTRs específicos
    ("unique_ctr", sa.Numeric(10, 4)),
    ("inline_link_click_ctr", sa.Numeric(10, 4)),
    ("outbound_clicks_ctr", sa.Numeric(10, 4)),

    # Funil de vídeo
    ("video_plays", sa.Integer()),
    ("video_15s_watched", sa.Integer()),
    ("video_p25_watched", sa.Integer()),
    ("video_p50_watched", sa.Integer()),
    ("video_p75_watched", sa.Integer()),
    ("video_p95_watched", sa.Integer()),
    ("video_thruplay", sa.Integer()),
    ("video_avg_time", sa.Numeric(10, 2)),

    # Métricas únicas
    ("unique_inline_link_clicks", sa.Integer()),
    ("unique_outbound_clicks", sa.Integer()),
    ("unique_conversions", sa.Integer()),

    # Custos únicos
    ("cost_per_unique_conversion", sa.Numeric(15, 4)),
    ("cost_per_unique_outbound_click", sa.Numeric(15, 4)),
    ("cost_per_inline_post_engagement", sa.Numeric(15, 4)),

    # CTRs únicos
    ("unique_link_clicks_ctr", sa.Numeric(10, 4)),
    ("unique_inline_link_click_ctr", sa.Numeric(10, 4)),
    ("unique_outbound_clicks_ctr", sa.Numeric(10, 4)),

    # Brand awareness
    ("estimated_ad_recallers", sa.Integer()),
    ("estimated_ad_recall_rate", sa.Numeric(10, 4)),
    ("cost_per_estimated_ad_recallers", sa.Numeric(15, 4)),

    # Landing page
    ("landing_page_views", sa.Integer()),

    # Catálogo
    ("converted_product_quantity", sa.Integer()),
    ("converted_product_value", sa.Numeric(15, 2)),

    # Action values raw
    ("action_values", sa.JSON()),
]

TABLES = [
    "sistema_facebook_ads_insights_history",
    "sistema_facebook_ads_insights_today",
]


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    for table_name in TABLES:
        if not inspector.has_table(table_name):
            continue

        existing_columns = {col["name"] for col in inspector.get_columns(table_name)}

        for col_name, col_type in NEW_COLUMNS:
            if col_name not in existing_columns:
                op.add_column(table_name, sa.Column(col_name, col_type))

    # Tabela de breakdowns avançados
    if not inspector.has_table("sistema_facebook_ads_insights_breakdowns"):
        op.create_table(
            "sistema_facebook_ads_insights_breakdowns",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column("config_id", sa.Integer(), nullable=False, index=True),
            sa.Column("ad_id", sa.String(), nullable=False),
            sa.Column("adset_id", sa.String(), nullable=False),
            sa.Column("campaign_id", sa.String(), nullable=False),
            sa.Column("date", sa.DateTime(), nullable=False),
            sa.Column("breakdown_type", sa.String(100), nullable=False),
            sa.Column("breakdown_value", sa.String(255), nullable=False),
            sa.Column("impressions", sa.Integer(), default=0),
            sa.Column("reach", sa.Integer(), default=0),
            sa.Column("clicks", sa.Integer(), default=0),
            sa.Column("spend", sa.Numeric(15, 2), default=0),
            sa.Column("leads", sa.Integer(), default=0),
            sa.Column("conversions", sa.Integer(), default=0),
            sa.Column("conversion_values", sa.Numeric(15, 2)),
            sa.Column("ctr", sa.Numeric(10, 4)),
            sa.Column("cpc", sa.Numeric(15, 4)),
            sa.Column("cpl", sa.Numeric(15, 4)),
            sa.Column("actions", sa.JSON()),
            sa.Column("synced_at", sa.DateTime(), server_default=sa.func.now()),
        )
        op.execute(
            "CREATE INDEX IF NOT EXISTS ix_fb_insights_bd_config_date "
            "ON sistema_facebook_ads_insights_breakdowns (config_id, date, breakdown_type)"
        )


def downgrade() -> None:
    op.drop_table("sistema_facebook_ads_insights_breakdowns")
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    for table_name in TABLES:
        if not inspector.has_table(table_name):
            continue
        existing_columns = {col["name"] for col in inspector.get_columns(table_name)}
        for col_name, _ in NEW_COLUMNS:
            if col_name in existing_columns:
                op.drop_column(table_name, col_name)
```

**Step 2: Rodar a migration**

Run: `cd /var/www/famachat-ml && alembic upgrade head`
Expected: Migration 009 aplicada sem erros.

**Step 3: Commit**

```bash
git add alembic/versions/009_expand_insights_metrics.py
git commit -m "feat(db): migration 009 - expandir métricas de insights do Facebook Ads"
```

---

### Task 2: Atualizar Modelos SQLAlchemy — Novas Colunas

**Files:**
- Modify: `shared/db/models/famachat_readonly.py` (linhas 190-307)

**Step 1: Adicionar novas colunas em `SistemaFacebookAdsInsightsHistory`**

Após a linha 246 (`actions: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON)`), adicionar:

```python
    # Action values raw
    action_values: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON)

    # Diagnósticos de qualidade (disponível com 500+ impressões)
    quality_ranking: Mapped[Optional[str]] = mapped_column(String(50))
    engagement_rate_ranking: Mapped[Optional[str]] = mapped_column(String(50))
    conversion_rate_ranking: Mapped[Optional[str]] = mapped_column(String(50))

    # ROAS
    purchase_roas: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 4))
    website_purchase_roas: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 4))

    # Custos granulares
    cpp: Mapped[Optional[Decimal]] = mapped_column(Numeric(15, 4))
    cost_per_unique_click: Mapped[Optional[Decimal]] = mapped_column(Numeric(15, 4))
    cost_per_inline_link_click: Mapped[Optional[Decimal]] = mapped_column(Numeric(15, 4))
    cost_per_outbound_click: Mapped[Optional[Decimal]] = mapped_column(Numeric(15, 4))
    cost_per_thruplay: Mapped[Optional[Decimal]] = mapped_column(Numeric(15, 4))

    # CTRs específicos
    unique_ctr: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 4))
    inline_link_click_ctr: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 4))
    outbound_clicks_ctr: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 4))

    # Funil de vídeo expandido
    video_plays: Mapped[Optional[int]] = mapped_column(Integer)
    video_15s_watched: Mapped[Optional[int]] = mapped_column(Integer)
    video_p25_watched: Mapped[Optional[int]] = mapped_column(Integer)
    video_p50_watched: Mapped[Optional[int]] = mapped_column(Integer)
    video_p75_watched: Mapped[Optional[int]] = mapped_column(Integer)
    video_p95_watched: Mapped[Optional[int]] = mapped_column(Integer)
    video_thruplay: Mapped[Optional[int]] = mapped_column(Integer)
    video_avg_time: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 2))

    # Métricas únicas
    unique_inline_link_clicks: Mapped[Optional[int]] = mapped_column(Integer)
    unique_outbound_clicks: Mapped[Optional[int]] = mapped_column(Integer)
    unique_conversions: Mapped[Optional[int]] = mapped_column(Integer)

    # Custos únicos
    cost_per_unique_conversion: Mapped[Optional[Decimal]] = mapped_column(Numeric(15, 4))
    cost_per_unique_outbound_click: Mapped[Optional[Decimal]] = mapped_column(Numeric(15, 4))
    cost_per_inline_post_engagement: Mapped[Optional[Decimal]] = mapped_column(Numeric(15, 4))

    # CTRs únicos
    unique_link_clicks_ctr: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 4))
    unique_inline_link_click_ctr: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 4))
    unique_outbound_clicks_ctr: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 4))

    # Brand awareness
    estimated_ad_recallers: Mapped[Optional[int]] = mapped_column(Integer)
    estimated_ad_recall_rate: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 4))
    cost_per_estimated_ad_recallers: Mapped[Optional[Decimal]] = mapped_column(Numeric(15, 4))

    # Landing page
    landing_page_views: Mapped[Optional[int]] = mapped_column(Integer)

    # Catálogo
    converted_product_quantity: Mapped[Optional[int]] = mapped_column(Integer)
    converted_product_value: Mapped[Optional[Decimal]] = mapped_column(Numeric(15, 2))
```

**Step 2: Mesmas colunas em `SistemaFacebookAdsInsightsToday`**

Após a linha 296 (`actions: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON)`), adicionar as mesmas colunas.

**Step 3: Adicionar modelo de Breakdowns**

Após `SistemaFacebookAdsInsightsToday`, adicionar:

```python
class SistemaFacebookAdsInsightsBreakdowns(Base):
    """Insights com breakdowns avançados (platform_position, region, etc.)."""
    __tablename__ = "sistema_facebook_ads_insights_breakdowns"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    config_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("sistema_facebook_ads_config.id", ondelete="CASCADE")
    )
    ad_id: Mapped[str] = mapped_column(String, nullable=False)
    adset_id: Mapped[str] = mapped_column(String, nullable=False)
    campaign_id: Mapped[str] = mapped_column(String, nullable=False)
    date: Mapped[datetime] = mapped_column(DateTime, nullable=False)

    breakdown_type: Mapped[str] = mapped_column(String(100), nullable=False)
    breakdown_value: Mapped[str] = mapped_column(String(255), nullable=False)

    impressions: Mapped[int] = mapped_column(Integer, default=0)
    reach: Mapped[int] = mapped_column(Integer, default=0)
    clicks: Mapped[int] = mapped_column(Integer, default=0)
    spend: Mapped[Decimal] = mapped_column(Numeric(15, 2), default=0)
    leads: Mapped[int] = mapped_column(Integer, default=0)
    conversions: Mapped[int] = mapped_column(Integer, default=0)
    conversion_values: Mapped[Optional[Decimal]] = mapped_column(Numeric(15, 2))
    ctr: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 4))
    cpc: Mapped[Optional[Decimal]] = mapped_column(Numeric(15, 4))
    cpl: Mapped[Optional[Decimal]] = mapped_column(Numeric(15, 4))
    actions: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON)
    synced_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
```

**Step 4: Adicionar alias**

No final do arquivo, adicionar:
```python
FacebookAdsInsightBreakdown = SistemaFacebookAdsInsightsBreakdowns
```

**Step 5: Commit**

```bash
git add shared/db/models/famachat_readonly.py
git commit -m "feat(models): adicionar colunas expandidas de insights e modelo de breakdowns"
```

---

### Task 3: Expandir INSIGHT_FIELDS no Client

**Files:**
- Modify: `projects/facebook_ads/client/insights.py` (linhas 18-63)

**Step 1: Expandir INSIGHT_FIELDS**

Substituir o `INSIGHT_FIELDS` atual (linhas 18-50) por:

```python
INSIGHT_FIELDS = [
    # Identificação
    "ad_id",
    "ad_name",
    "adset_id",
    "adset_name",
    "campaign_id",
    "campaign_name",
    "account_id",
    "account_name",
    "objective",
    "date_start",
    "date_stop",
    # Alcance e impressões
    "impressions",
    "reach",
    "frequency",
    # Cliques
    "clicks",
    "unique_clicks",
    "inline_link_clicks",
    "outbound_clicks",
    "unique_inline_link_clicks",
    "unique_outbound_clicks",
    # Custo
    "spend",
    "cpc",
    "cpm",
    "cpp",
    "cost_per_unique_click",
    "cost_per_inline_link_click",
    "cost_per_outbound_click",
    "cost_per_thruplay",
    "cost_per_unique_conversion",
    "cost_per_unique_outbound_click",
    "cost_per_inline_post_engagement",
    "cost_per_estimated_ad_recallers",
    # CTRs
    "ctr",
    "unique_ctr",
    "inline_link_click_ctr",
    "outbound_clicks_ctr",
    "unique_link_clicks_ctr",
    "unique_inline_link_click_ctr",
    "unique_outbound_clicks_ctr",
    # Conversões
    "actions",
    "action_values",
    "cost_per_action_type",
    "conversions",
    "conversion_values",
    "cost_per_conversion",
    "unique_conversions",
    # ROAS
    "purchase_roas",
    "website_purchase_roas",
    # Diagnósticos de qualidade
    "quality_ranking",
    "engagement_rate_ranking",
    "conversion_rate_ranking",
    # Vídeo — funil completo
    "video_play_actions",
    "video_15_sec_watched_actions",
    "video_p25_watched_actions",
    "video_p50_watched_actions",
    "video_p75_watched_actions",
    "video_p95_watched_actions",
    "video_30_sec_watched_actions",
    "video_p100_watched_actions",
    "video_thruplay_watched_actions",
    "video_avg_time_watched_actions",
    # Brand awareness
    "estimated_ad_recallers",
    "estimated_ad_recall_rate",
    # Catálogo
    "catalog_segment_actions",
    "catalog_segment_value",
    "converted_product_quantity",
    "converted_product_value",
]
```

**Step 2: Expandir breakdowns e action breakdowns**

Substituir as listas de breakdowns (linhas 52-60) por:

```python
# Breakdowns padrão
STANDARD_BREAKDOWNS = [
    "age",
    "gender",
    "country",
    "publisher_platform",
    "device_platform",
]

# Breakdowns avançados (queries separadas — não combinar entre si)
ADVANCED_BREAKDOWNS = {
    "platform_position": ["publisher_platform", "platform_position"],
    "region": ["region"],
    "impression_device": ["impression_device"],
    "frequency_value": ["frequency_value"],
    "hourly": ["hourly_stats_aggregated_by_advertiser_time_zone"],
}

# Action breakdowns
ACTION_BREAKDOWNS = ["action_type", "action_device"]

ADVANCED_ACTION_BREAKDOWNS = [
    "action_destination",
    "action_video_sound",
    "action_carousel_card_id",
    "action_carousel_card_name",
    "action_reaction",
]

# Dynamic Creative Asset breakdowns (para campanhas Advantage+/DCO)
DCO_BREAKDOWNS = [
    "body_asset",
    "title_asset",
    "image_asset",
    "video_asset",
    "call_to_action_asset",
    "description_asset",
    "link_url_asset",
    "ad_format_asset",
]
```

**Step 3: Commit**

```bash
git add projects/facebook_ads/client/insights.py
git commit -m "feat(client): expandir INSIGHT_FIELDS e breakdowns da Meta API"
```

---

### Task 4: Expandir Metrics Calculator

**Files:**
- Modify: `projects/facebook_ads/utils/metrics_calculator.py`

**Step 1: Adicionar novas funções de extração**

Após `calculate_frequency` (linha 55), adicionar:

```python
def calculate_cpp(spend: Decimal | float, reach: int) -> Optional[Decimal]:
    """CPP = (spend / reach) * 1000 — custo por 1000 pessoas únicas."""
    result = safe_divide(spend, reach, 6)
    return (
        (result * 1000).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)
        if result
        else None
    )


def extract_roas_from_list(roas_list: list | None) -> Optional[Decimal]:
    """Extrai valor de ROAS de uma lista de AdsActionStats."""
    if not roas_list or not isinstance(roas_list, list):
        return None
    for item in roas_list:
        if isinstance(item, dict):
            try:
                val = Decimal(str(item.get("value", "0")))
                if val > 0:
                    return val.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)
            except (InvalidOperation, ValueError):
                pass
    return None


def extract_video_metric(insight: dict, field_name: str) -> int:
    """Extrai métrica de vídeo genérica de uma lista de AdsActionStats."""
    data = insight.get(field_name, [])
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                try:
                    return int(item.get("value", 0))
                except (ValueError, TypeError):
                    pass
    return 0


def extract_video_avg_time(insight: dict) -> Optional[Decimal]:
    """Extrai tempo médio de vídeo assistido."""
    data = insight.get("video_avg_time_watched_actions", [])
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                try:
                    return Decimal(str(item.get("value", "0"))).quantize(
                        Decimal("0.01"), rounding=ROUND_HALF_UP
                    )
                except (InvalidOperation, ValueError):
                    pass
    return None


def extract_action_stat_value(
    data: list | None, decimal_result: bool = False
) -> int | Optional[Decimal]:
    """Extrai valor de um campo que é List[AdsActionStats] (ex: cost_per_outbound_click)."""
    if not data or not isinstance(data, list):
        return Decimal("0") if decimal_result else 0
    for item in data:
        if isinstance(item, dict):
            try:
                val = item.get("value", "0")
                if decimal_result:
                    return Decimal(str(val)).quantize(
                        Decimal("0.0001"), rounding=ROUND_HALF_UP
                    )
                return int(val)
            except (InvalidOperation, ValueError, TypeError):
                pass
    return Decimal("0") if decimal_result else 0


def extract_landing_page_views(actions: list | None) -> int:
    """Extrai landing_page_view do array de actions."""
    if not actions or not isinstance(actions, list):
        return 0
    for action in actions:
        if isinstance(action, dict) and action.get("action_type") == "landing_page_view":
            try:
                return int(action.get("value", 0))
            except (ValueError, TypeError):
                pass
    return 0
```

**Step 2: Commit**

```bash
git add projects/facebook_ads/utils/metrics_calculator.py
git commit -m "feat(utils): adicionar funções de extração para métricas expandidas"
```

---

### Task 5: Atualizar Sync Service — Processar Novos Campos

**Files:**
- Modify: `projects/facebook_ads/services/sync_insights.py`

**Step 1: Atualizar imports**

Adicionar novos imports do metrics_calculator (linha 19-22):

```python
from projects.facebook_ads.utils.metrics_calculator import (
    calculate_ctr, calculate_cpc, calculate_cpm, calculate_cpl,
    calculate_frequency, calculate_cpp, extract_leads_from_actions,
    extract_roas_from_list, extract_video_metric, extract_video_avg_time,
    extract_action_stat_value, extract_landing_page_views,
)
```

**Step 2: Atualizar `_parse_insight_to_today`**

Após a linha 273 (`actions=actions if actions else None,`), adicionar os novos campos antes de `synced_at`:

```python
            # Action values raw
            action_values=insight.get("action_values") if insight.get("action_values") else None,
            # Diagnósticos de qualidade
            quality_ranking=insight.get("quality_ranking"),
            engagement_rate_ranking=insight.get("engagement_rate_ranking"),
            conversion_rate_ranking=insight.get("conversion_rate_ranking"),
            # ROAS
            purchase_roas=extract_roas_from_list(insight.get("purchase_roas")),
            website_purchase_roas=extract_roas_from_list(insight.get("website_purchase_roas")),
            # Custos granulares
            cpp=Decimal(str(insight["cpp"])) if insight.get("cpp") else calculate_cpp(spend, reach),
            cost_per_unique_click=Decimal(str(insight["cost_per_unique_click"])) if insight.get("cost_per_unique_click") else None,
            cost_per_inline_link_click=Decimal(str(insight["cost_per_inline_link_click"])) if insight.get("cost_per_inline_link_click") else None,
            cost_per_outbound_click=extract_action_stat_value(insight.get("cost_per_outbound_click"), decimal_result=True) or None,
            cost_per_thruplay=extract_action_stat_value(insight.get("cost_per_thruplay"), decimal_result=True) or None,
            # CTRs
            unique_ctr=Decimal(str(insight["unique_ctr"])) if insight.get("unique_ctr") else None,
            inline_link_click_ctr=Decimal(str(insight["inline_link_click_ctr"])) if insight.get("inline_link_click_ctr") else None,
            outbound_clicks_ctr=Decimal(str(insight["outbound_clicks_ctr"])) if insight.get("outbound_clicks_ctr") else None,
            # Funil de vídeo
            video_plays=extract_video_metric(insight, "video_play_actions"),
            video_15s_watched=extract_video_metric(insight, "video_15_sec_watched_actions"),
            video_p25_watched=extract_video_metric(insight, "video_p25_watched_actions"),
            video_p50_watched=extract_video_metric(insight, "video_p50_watched_actions"),
            video_p75_watched=extract_video_metric(insight, "video_p75_watched_actions"),
            video_p95_watched=extract_video_metric(insight, "video_p95_watched_actions"),
            video_thruplay=extract_video_metric(insight, "video_thruplay_watched_actions"),
            video_avg_time=extract_video_avg_time(insight),
            # Métricas únicas
            unique_inline_link_clicks=int(insight.get("unique_inline_link_clicks", 0)),
            unique_outbound_clicks=extract_action_stat_value(insight.get("unique_outbound_clicks")),
            unique_conversions=int(insight.get("unique_conversions", 0)) if insight.get("unique_conversions") else None,
            # Custos únicos
            cost_per_unique_conversion=Decimal(str(insight["cost_per_unique_conversion"])) if insight.get("cost_per_unique_conversion") else None,
            cost_per_unique_outbound_click=extract_action_stat_value(insight.get("cost_per_unique_outbound_click"), decimal_result=True) or None,
            cost_per_inline_post_engagement=Decimal(str(insight["cost_per_inline_post_engagement"])) if insight.get("cost_per_inline_post_engagement") else None,
            # CTRs únicos
            unique_link_clicks_ctr=Decimal(str(insight["unique_link_clicks_ctr"])) if insight.get("unique_link_clicks_ctr") else None,
            unique_inline_link_click_ctr=Decimal(str(insight["unique_inline_link_click_ctr"])) if insight.get("unique_inline_link_click_ctr") else None,
            unique_outbound_clicks_ctr=Decimal(str(insight["unique_outbound_clicks_ctr"])) if insight.get("unique_outbound_clicks_ctr") else None,
            # Brand awareness
            estimated_ad_recallers=int(insight["estimated_ad_recallers"]) if insight.get("estimated_ad_recallers") else None,
            estimated_ad_recall_rate=Decimal(str(insight["estimated_ad_recall_rate"])) if insight.get("estimated_ad_recall_rate") else None,
            cost_per_estimated_ad_recallers=Decimal(str(insight["cost_per_estimated_ad_recallers"])) if insight.get("cost_per_estimated_ad_recallers") else None,
            # Landing page
            landing_page_views=extract_landing_page_views(actions),
            # Catálogo
            converted_product_quantity=int(insight["converted_product_quantity"]) if insight.get("converted_product_quantity") else None,
            converted_product_value=Decimal(str(insight["converted_product_value"])) if insight.get("converted_product_value") else None,
```

**Step 3: Mesmos campos em `_parse_insight_to_history`**

Replicar os mesmos campos no método `_parse_insight_to_history`.

**Step 4: Mesmos campos em `_update_insight_history`**

Adicionar as mesmas atribuições no método `_update_insight_history`.

**Step 5: Commit**

```bash
git add projects/facebook_ads/services/sync_insights.py
git commit -m "feat(sync): processar métricas expandidas na sincronização de insights"
```

---

### Task 6: Atualizar Schemas Pydantic

**Files:**
- Modify: `projects/facebook_ads/schemas/insights.py`

**Step 1: Expandir KPIResponse**

Adicionar novos campos após `frequency`:

```python
    # Custos granulares
    cpp: Optional[Decimal] = None
    cost_per_unique_click: Optional[Decimal] = None
    cost_per_thruplay: Optional[Decimal] = None
    # CTRs
    unique_ctr: Optional[Decimal] = None
    inline_link_click_ctr: Optional[Decimal] = None
    # ROAS
    purchase_roas: Optional[Decimal] = None
```

**Step 2: Expandir DailyInsightResponse**

Adicionar após `cpl`:
```python
    unique_clicks: int = 0
    unique_ctr: Optional[Decimal] = None
    video_plays: Optional[int] = None
    video_thruplay: Optional[int] = None
```

**Step 3: Expandir CampaignInsightResponse**

Adicionar após `cpl`:
```python
    cpp: Optional[Decimal] = None
    unique_ctr: Optional[Decimal] = None
    cost_per_unique_click: Optional[Decimal] = None
```

**Step 4: Adicionar BreakdownDetailResponse**

```python
class BreakdownDetailResponse(CamelCaseModel):
    """Insight detalhado com breakdown avançado."""
    breakdown_type: str
    breakdown_value: str
    campaign_id: Optional[str] = None
    campaign_name: Optional[str] = None
    spend: Decimal = Decimal("0")
    impressions: int = 0
    reach: int = 0
    clicks: int = 0
    leads: int = 0
    conversions: int = 0
    ctr: Optional[Decimal] = None
    cpc: Optional[Decimal] = None
    cpl: Optional[Decimal] = None


class VideoFunnelResponse(CamelCaseModel):
    """Funil completo de vídeo."""
    video_plays: int = 0
    video_15s_watched: int = 0
    video_p25_watched: int = 0
    video_p50_watched: int = 0
    video_p75_watched: int = 0
    video_p95_watched: int = 0
    video_30s_watched: int = 0
    video_p100_watched: int = 0
    video_thruplay: int = 0
    video_avg_time: Optional[Decimal] = None


class QualityDiagnosticsResponse(CamelCaseModel):
    """Diagnósticos de qualidade de anúncios."""
    ad_id: str
    ad_name: Optional[str] = None
    quality_ranking: Optional[str] = None
    engagement_rate_ranking: Optional[str] = None
    conversion_rate_ranking: Optional[str] = None
    impressions: int = 0
    spend: Decimal = Decimal("0")
```

**Step 5: Commit**

```bash
git add projects/facebook_ads/schemas/insights.py
git commit -m "feat(schemas): adicionar schemas para métricas expandidas, breakdowns e vídeo"
```

---

### Task 7: Adicionar Modelo de Breakdowns ao Models __init__

**Files:**
- Modify: `projects/facebook_ads/models/__init__.py`

**Step 1: Adicionar import do novo modelo**

Adicionar na lista de imports de `shared.db.models.famachat_readonly`:
```python
from shared.db.models.famachat_readonly import SistemaFacebookAdsInsightsBreakdowns
```

E no `__all__` (se existir), adicionar `"SistemaFacebookAdsInsightsBreakdowns"`.

**Step 2: Commit**

```bash
git add projects/facebook_ads/models/__init__.py
git commit -m "feat(models): exportar modelo de breakdowns de insights"
```

---

### Task 8: Atualizar API Endpoints — Expor Novas Métricas

**Files:**
- Modify: `projects/facebook_ads/api/insights.py`

**Step 1: Adicionar novos imports**

```python
from shared.db.models.famachat_readonly import (
    SistemaFacebookAdsInsightsHistory,
    SistemaFacebookAdsInsightsToday,
    SistemaFacebookAdsCampaigns,
    SistemaFacebookAdsInsightsBreakdowns,
)
```

**Step 2: Expandir métricas retornadas em `/kpis`**

No `current_metrics` dict (linha 96-109), adicionar:

```python
        "unique_clicks": unique_clicks,
        "cpp": float(calculate_cpp(spend, reach) or 0),
        "unique_ctr": float(safe_divide(unique_clicks, reach) or 0) * 100 if reach else 0,
```

**Step 3: Adicionar endpoint `/quality-diagnostics`**

```python
@router.get("/quality-diagnostics")
async def get_quality_diagnostics(
    config_id: int = Query(..., alias="configId"),
    date_from: Optional[str] = Query(None, alias="dateFrom"),
    date_to: Optional[str] = Query(None, alias="dateTo"),
    date_preset: Optional[str] = Query("last_30d", alias="datePreset"),
    db: AsyncSession = Depends(get_db),
):
    """Diagnósticos de qualidade dos anúncios (rankings)."""
    since, until = _parse_date_params(date_from, date_to, date_preset)
    I = SistemaFacebookAdsInsightsHistory

    result = await db.execute(
        select(
            I.ad_id,
            I.quality_ranking,
            I.engagement_rate_ranking,
            I.conversion_rate_ranking,
            func.sum(I.impressions).label("impressions"),
            func.sum(I.spend).label("spend"),
        ).where(
            and_(
                I.config_id == config_id,
                I.date >= since,
                I.date <= until,
                I.quality_ranking.isnot(None),
            )
        ).group_by(
            I.ad_id, I.quality_ranking, I.engagement_rate_ranking, I.conversion_rate_ranking,
        ).order_by(desc(func.sum(I.spend)))
    )
    rows = result.all()

    data = [
        camel_keys({
            "ad_id": r.ad_id,
            "quality_ranking": r.quality_ranking,
            "engagement_rate_ranking": r.engagement_rate_ranking,
            "conversion_rate_ranking": r.conversion_rate_ranking,
            "impressions": int(r.impressions or 0),
            "spend": float(r.spend or 0),
        })
        for r in rows
    ]
    return {"success": True, "data": data}
```

**Step 4: Adicionar endpoint `/video-funnel`**

```python
@router.get("/video-funnel")
async def get_video_funnel(
    config_id: int = Query(..., alias="configId"),
    date_from: Optional[str] = Query(None, alias="dateFrom"),
    date_to: Optional[str] = Query(None, alias="dateTo"),
    date_preset: Optional[str] = Query("last_30d", alias="datePreset"),
    campaign_id: Optional[str] = Query(None, alias="campaignId"),
    db: AsyncSession = Depends(get_db),
):
    """Funil completo de métricas de vídeo."""
    since, until = _parse_date_params(date_from, date_to, date_preset)
    I = SistemaFacebookAdsInsightsHistory

    filters = [
        I.config_id == config_id,
        I.date >= since,
        I.date <= until,
    ]
    if campaign_id:
        filters.append(I.campaign_id == campaign_id)

    result = await db.execute(
        select(
            func.coalesce(func.sum(I.video_plays), 0).label("video_plays"),
            func.coalesce(func.sum(I.video_15s_watched), 0).label("video_15s_watched"),
            func.coalesce(func.sum(I.video_p25_watched), 0).label("video_p25_watched"),
            func.coalesce(func.sum(I.video_p50_watched), 0).label("video_p50_watched"),
            func.coalesce(func.sum(I.video_p75_watched), 0).label("video_p75_watched"),
            func.coalesce(func.sum(I.video_p95_watched), 0).label("video_p95_watched"),
            func.coalesce(func.sum(I.video_views), 0).label("video_30s_watched"),
            func.coalesce(func.sum(I.video_p100_watched), 0).label("video_p100_watched"),
            func.coalesce(func.sum(I.video_thruplay), 0).label("video_thruplay"),
            func.avg(I.video_avg_time).label("video_avg_time"),
        ).where(and_(*filters))
    )
    row = result.one()

    data = camel_keys({
        "video_plays": int(row.video_plays),
        "video_15s_watched": int(row.video_15s_watched),
        "video_p25_watched": int(row.video_p25_watched),
        "video_p50_watched": int(row.video_p50_watched),
        "video_p75_watched": int(row.video_p75_watched),
        "video_p95_watched": int(row.video_p95_watched),
        "video_30s_watched": int(row.video_30s_watched),
        "video_p100_watched": int(row.video_p100_watched),
        "video_thruplay": int(row.video_thruplay),
        "video_avg_time": float(row.video_avg_time or 0),
    })
    return {"success": True, "data": data}
```

**Step 5: Adicionar endpoint `/breakdowns`**

```python
@router.get("/breakdowns")
async def get_breakdown_insights(
    config_id: int = Query(..., alias="configId"),
    breakdown_type: str = Query(..., alias="breakdownType"),
    date_from: Optional[str] = Query(None, alias="dateFrom"),
    date_to: Optional[str] = Query(None, alias="dateTo"),
    date_preset: Optional[str] = Query("last_30d", alias="datePreset"),
    campaign_id: Optional[str] = Query(None, alias="campaignId"),
    limit: int = Query(50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
):
    """Insights agregados por breakdown avançado."""
    since, until = _parse_date_params(date_from, date_to, date_preset)
    B = SistemaFacebookAdsInsightsBreakdowns

    filters = [
        B.config_id == config_id,
        B.date >= since,
        B.date <= until,
        B.breakdown_type == breakdown_type,
    ]
    if campaign_id:
        filters.append(B.campaign_id == campaign_id)

    result = await db.execute(
        select(
            B.breakdown_value,
            func.sum(B.spend).label("spend"),
            func.sum(B.impressions).label("impressions"),
            func.sum(B.reach).label("reach"),
            func.sum(B.clicks).label("clicks"),
            func.sum(B.leads).label("leads"),
            func.sum(B.conversions).label("conversions"),
        ).where(
            and_(*filters)
        ).group_by(B.breakdown_value)
        .order_by(desc(func.sum(B.spend)))
        .limit(limit)
    )
    rows = result.all()

    data = []
    for row in rows:
        spend = Decimal(str(row.spend or 0))
        clicks = int(row.clicks or 0)
        impressions = int(row.impressions or 0)
        leads = int(row.leads or 0)

        data.append(camel_keys({
            "breakdown_type": breakdown_type,
            "breakdown_value": row.breakdown_value,
            "spend": float(spend),
            "impressions": impressions,
            "reach": int(row.reach or 0),
            "clicks": clicks,
            "leads": leads,
            "conversions": int(row.conversions or 0),
            "ctr": float(calculate_ctr(clicks, impressions) or 0),
            "cpc": float(calculate_cpc(spend, clicks) or 0),
            "cpl": float(calculate_cpl(spend, leads) or 0),
        }))

    return {
        "success": True,
        "data": data,
        "meta": camel_keys({"breakdown_type": breakdown_type, "period": date_preset, "total": len(data)}),
    }
```

**Step 6: Commit**

```bash
git add projects/facebook_ads/api/insights.py
git commit -m "feat(api): adicionar endpoints de quality diagnostics, video funnel e breakdowns"
```

---

### Task 9: Adicionar Serviço de Sync de Breakdowns

**Files:**
- Create: `projects/facebook_ads/services/sync_breakdowns.py`

**Step 1: Criar serviço de sync de breakdowns**

```python
"""Serviço de sincronização de breakdowns avançados do Facebook Ads."""

from datetime import datetime
from decimal import Decimal

from sqlalchemy import delete, and_
from sqlalchemy.ext.asyncio import AsyncSession

from shared.core.logging import get_logger
from projects.facebook_ads.client.base import FacebookGraphClient
from projects.facebook_ads.client.insights import (
    InsightsClient, INSIGHT_FIELDS, ADVANCED_BREAKDOWNS,
)
from projects.facebook_ads.security.token_encryption import decrypt_token
from projects.facebook_ads.utils.date_helpers import get_date_range
from projects.facebook_ads.utils.metrics_calculator import (
    calculate_ctr, calculate_cpc, calculate_cpl, extract_leads_from_actions,
)
from shared.db.models.famachat_readonly import (
    SistemaFacebookAdsConfig,
    SistemaFacebookAdsInsightsBreakdowns,
)

logger = get_logger(__name__)

# Campos reduzidos para queries com breakdown (evita campos incompatíveis)
BREAKDOWN_FIELDS = [
    "ad_id", "adset_id", "campaign_id",
    "impressions", "reach", "clicks", "spend",
    "actions", "action_values", "conversions", "conversion_values",
    "ctr", "cpc",
    "date_start", "date_stop",
]


class SyncBreakdownsService:
    """Sincroniza insights com breakdowns avançados."""

    BATCH_SIZE = 500

    def __init__(self, db: AsyncSession):
        self.db = db

    async def sync_breakdowns(
        self,
        config: SistemaFacebookAdsConfig,
        breakdown_types: list[str] | None = None,
        days_back: int = 30,
    ) -> dict[str, int]:
        """Sincroniza breakdowns para os tipos especificados."""
        types_to_sync = breakdown_types or list(ADVANCED_BREAKDOWNS.keys())
        since, until = get_date_range(days_back)
        time_range = {"since": since, "until": until}

        access_token = decrypt_token(config.access_token)
        graph_client = FacebookGraphClient(access_token, config.account_id)
        insights_client = InsightsClient(graph_client)

        total_inserted = 0
        total_errors = 0

        try:
            for bd_type in types_to_sync:
                bd_fields = ADVANCED_BREAKDOWNS.get(bd_type)
                if not bd_fields:
                    continue

                logger.info(
                    "Sincronizando breakdown",
                    config_id=config.id,
                    breakdown_type=bd_type,
                )

                try:
                    fb_insights = await insights_client.get_insights(
                        f"act_{config.account_id}",
                        time_range=time_range,
                        level="ad",
                        fields=BREAKDOWN_FIELDS,
                        breakdowns=bd_fields,
                    )

                    # Limpar dados anteriores deste breakdown/período
                    await self.db.execute(
                        delete(SistemaFacebookAdsInsightsBreakdowns).where(
                            and_(
                                SistemaFacebookAdsInsightsBreakdowns.config_id == config.id,
                                SistemaFacebookAdsInsightsBreakdowns.breakdown_type == bd_type,
                                SistemaFacebookAdsInsightsBreakdowns.date >= datetime.strptime(since, "%Y-%m-%d"),
                                SistemaFacebookAdsInsightsBreakdowns.date <= datetime.strptime(until, "%Y-%m-%d"),
                            )
                        )
                    )

                    inserted = 0
                    for insight in fb_insights:
                        try:
                            # O valor do breakdown está no campo com o mesmo nome
                            bd_value = None
                            for field in bd_fields:
                                bd_value = insight.get(field)
                                if bd_value:
                                    break

                            if not bd_value:
                                continue

                            actions = insight.get("actions", [])
                            impressions = int(insight.get("impressions", 0))
                            clicks = int(insight.get("clicks", 0))
                            spend = Decimal(str(insight.get("spend", "0")))
                            leads = extract_leads_from_actions(actions)

                            obj = SistemaFacebookAdsInsightsBreakdowns(
                                config_id=config.id,
                                ad_id=insight.get("ad_id", ""),
                                adset_id=insight.get("adset_id", ""),
                                campaign_id=insight.get("campaign_id", ""),
                                date=datetime.strptime(insight.get("date_start", ""), "%Y-%m-%d"),
                                breakdown_type=bd_type,
                                breakdown_value=str(bd_value),
                                impressions=impressions,
                                reach=int(insight.get("reach", 0)),
                                clicks=clicks,
                                spend=spend,
                                leads=leads,
                                conversions=int(insight.get("conversions", 0)) if insight.get("conversions") else 0,
                                conversion_values=Decimal(str(insight.get("conversion_values", "0"))) if insight.get("conversion_values") else None,
                                ctr=calculate_ctr(clicks, impressions),
                                cpc=calculate_cpc(spend, clicks),
                                cpl=calculate_cpl(spend, leads),
                                actions=actions if actions else None,
                            )
                            self.db.add(obj)
                            inserted += 1

                            if inserted % self.BATCH_SIZE == 0:
                                await self.db.flush()

                        except Exception as e:
                            logger.error("Erro ao processar breakdown", error=str(e))
                            total_errors += 1

                    await self.db.flush()
                    total_inserted += inserted
                    logger.info(
                        "Breakdown sincronizado",
                        breakdown_type=bd_type,
                        inserted=inserted,
                    )

                except Exception as e:
                    logger.error(
                        "Erro ao sincronizar breakdown",
                        breakdown_type=bd_type,
                        error=str(e),
                    )
                    total_errors += 1

            return {"inserted": total_inserted, "errors": total_errors}

        finally:
            await graph_client.close()
```

**Step 2: Commit**

```bash
git add projects/facebook_ads/services/sync_breakdowns.py
git commit -m "feat(sync): adicionar serviço de sincronização de breakdowns avançados"
```

---

### Task 10: Atualizar Tipos TypeScript do Frontend

**Files:**
- Modify: `frontend/types/facebook-ads.ts`

**Step 1: Expandir `CampaignWithMetrics`**

Após `cpl: number | null;` (linha 84), adicionar:
```typescript
  cpp: number | null;
  uniqueCtr: number | null;
  costPerUniqueClick: number | null;
  purchaseRoas: number | null;
```

**Step 2: Expandir `InsightsSummary`**

Adicionar novos campos:
```typescript
  totalUniqueClicks: number;
  avgCpp: number;
  avgUniqueCtr: number;
  avgPurchaseRoas: number | null;
```

**Step 3: Expandir `TimeSeriesDataPoint`**

Adicionar:
```typescript
  uniqueClicks: number;
  uniqueCtr: number | null;
  videoPlays: number | null;
  videoThruplay: number | null;
```

**Step 4: Adicionar novas interfaces**

```typescript
// ==============================
// INTERFACES DE MÉTRICAS EXPANDIDAS
// ==============================

export interface VideoFunnel {
  videoPlays: number;
  video15sWatched: number;
  videoP25Watched: number;
  videoP50Watched: number;
  videoP75Watched: number;
  videoP95Watched: number;
  video30sWatched: number;
  videoP100Watched: number;
  videoThruplay: number;
  videoAvgTime: number;
}

export interface QualityDiagnostics {
  adId: string;
  adName: string | null;
  qualityRanking: string | null;
  engagementRateRanking: string | null;
  conversionRateRanking: string | null;
  impressions: number;
  spend: number;
}

export type QualityRanking =
  | "BELOW_AVERAGE_10"
  | "BELOW_AVERAGE_20"
  | "BELOW_AVERAGE_35"
  | "AVERAGE"
  | "ABOVE_AVERAGE";

export interface BreakdownInsight {
  breakdownType: string;
  breakdownValue: string;
  spend: number;
  impressions: number;
  reach: number;
  clicks: number;
  leads: number;
  conversions: number;
  ctr: number;
  cpc: number;
  cpl: number | null;
}

export type BreakdownType =
  | "platform_position"
  | "region"
  | "impression_device"
  | "frequency_value"
  | "hourly"
  | "age"
  | "gender"
  | "country";
```

**Step 5: Commit**

```bash
git add frontend/types/facebook-ads.ts
git commit -m "feat(frontend): adicionar tipos TypeScript para métricas expandidas"
```

---

### Task 11: Registrar Novo Modelo no Alembic env.py

**Files:**
- Modify: `alembic/env.py`

**Step 1: Adicionar import do novo modelo**

Após os imports existentes de `projects.facebook_ads.models`, adicionar:
```python
from shared.db.models.famachat_readonly import SistemaFacebookAdsInsightsBreakdowns  # noqa: F401
```

**Step 2: Commit**

```bash
git add alembic/env.py
git commit -m "feat(alembic): registrar modelo de breakdowns no env.py"
```

---

### Task 12: Commit Final e Verificação

**Step 1: Verificar que todos os arquivos foram commitados**

Run: `git status`
Expected: Working tree clean ou apenas arquivos não relacionados.

**Step 2: Verificar imports**

Run: `cd /var/www/famachat-ml && python -c "from shared.db.models.famachat_readonly import SistemaFacebookAdsInsightsBreakdowns; print('OK')"`
Expected: `OK`

Run: `cd /var/www/famachat-ml && python -c "from projects.facebook_ads.utils.metrics_calculator import calculate_cpp, extract_roas_from_list, extract_video_metric; print('OK')"`
Expected: `OK`

Run: `cd /var/www/famachat-ml && python -c "from projects.facebook_ads.client.insights import INSIGHT_FIELDS, ADVANCED_BREAKDOWNS; print(f'Fields: {len(INSIGHT_FIELDS)}, Breakdowns: {list(ADVANCED_BREAKDOWNS.keys())}')"`
Expected: `Fields: ~75, Breakdowns: ['platform_position', 'region', 'impression_device', 'frequency_value', 'hourly']`
