"""Expand insights metrics and create breakdowns table.

Revision ID: 009
Revises: 008
Create Date: 2026-02-02

Changes:
- Add quality diagnostics, ROAS, granular costs, CTRs, video funnel,
  unique metrics, brand awareness, landing page, catalog and action_values
  columns to both insights_history and insights_today tables.
- Create sistema_facebook_ads_insights_breakdowns table.
"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "009"
down_revision = "008"
branch_labels = None
depends_on = None

INSIGHTS_TABLES = [
    "sistema_facebook_ads_insights_history",
    "sistema_facebook_ads_insights_today",
]

NEW_COLUMNS = [
    # Quality diagnostics
    ("quality_ranking", sa.String(50)),
    ("engagement_rate_ranking", sa.String(50)),
    ("conversion_rate_ranking", sa.String(50)),
    # ROAS
    ("purchase_roas", sa.Numeric(10, 4)),
    ("website_purchase_roas", sa.Numeric(10, 4)),
    # Granular costs
    ("cpp", sa.Numeric(15, 4)),
    ("cost_per_unique_click", sa.Numeric(15, 4)),
    ("cost_per_inline_link_click", sa.Numeric(15, 4)),
    ("cost_per_outbound_click", sa.Numeric(15, 4)),
    ("cost_per_thruplay", sa.Numeric(15, 4)),
    # Specific CTRs
    ("unique_ctr", sa.Numeric(10, 4)),
    ("inline_link_click_ctr", sa.Numeric(10, 4)),
    ("outbound_clicks_ctr", sa.Numeric(10, 4)),
    # Video funnel
    ("video_plays", sa.Integer()),
    ("video_15s_watched", sa.Integer()),
    ("video_p25_watched", sa.Integer()),
    ("video_p50_watched", sa.Integer()),
    ("video_p75_watched", sa.Integer()),
    ("video_p95_watched", sa.Integer()),
    ("video_thruplay", sa.Integer()),
    ("video_avg_time", sa.Numeric(10, 2)),
    # Unique metrics
    ("unique_inline_link_clicks", sa.Integer()),
    ("unique_outbound_clicks", sa.Integer()),
    ("unique_conversions", sa.Integer()),
    # Unique costs
    ("cost_per_unique_conversion", sa.Numeric(15, 4)),
    ("cost_per_unique_outbound_click", sa.Numeric(15, 4)),
    ("cost_per_inline_post_engagement", sa.Numeric(15, 4)),
    # Unique CTRs
    ("unique_link_clicks_ctr", sa.Numeric(10, 4)),
    ("unique_inline_link_click_ctr", sa.Numeric(10, 4)),
    ("unique_outbound_clicks_ctr", sa.Numeric(10, 4)),
    # Brand awareness
    ("estimated_ad_recallers", sa.Integer()),
    ("estimated_ad_recall_rate", sa.Numeric(10, 4)),
    ("cost_per_estimated_ad_recallers", sa.Numeric(15, 4)),
    # Landing page
    ("landing_page_views", sa.Integer()),
    # Catalog
    ("converted_product_quantity", sa.Integer()),
    ("converted_product_value", sa.Numeric(15, 2)),
    # Action values raw
    ("action_values", sa.JSON()),
]


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    # --- Add new columns to both insights tables ---
    for table_name in INSIGHTS_TABLES:
        if not inspector.has_table(table_name):
            continue

        existing_columns = {col["name"] for col in inspector.get_columns(table_name)}

        for col_name, col_type in NEW_COLUMNS:
            if col_name not in existing_columns:
                op.add_column(table_name, sa.Column(col_name, col_type))

    # --- Create breakdowns table ---
    if not inspector.has_table("sistema_facebook_ads_insights_breakdowns"):
        op.create_table(
            "sistema_facebook_ads_insights_breakdowns",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column("config_id", sa.Integer(), nullable=False, index=True),
            sa.Column("ad_id", sa.String(100)),
            sa.Column("adset_id", sa.String(100)),
            sa.Column("campaign_id", sa.String(100)),
            sa.Column("date", sa.Date(), nullable=False),
            sa.Column("breakdown_type", sa.String(100), nullable=False),
            sa.Column("breakdown_value", sa.String(255), nullable=False),
            sa.Column("impressions", sa.Integer()),
            sa.Column("reach", sa.Integer()),
            sa.Column("clicks", sa.Integer()),
            sa.Column("spend", sa.Numeric(15, 4)),
            sa.Column("leads", sa.Integer()),
            sa.Column("conversions", sa.Integer()),
            sa.Column("conversion_values", sa.Numeric(15, 4)),
            sa.Column("ctr", sa.Numeric(10, 4)),
            sa.Column("cpc", sa.Numeric(15, 4)),
            sa.Column("cpl", sa.Numeric(15, 4)),
            sa.Column("actions", sa.JSON()),
            sa.Column("synced_at", sa.DateTime(), server_default=sa.func.now()),
        )

        op.create_index(
            "ix_fb_insights_breakdowns_config_date_type",
            "sistema_facebook_ads_insights_breakdowns",
            ["config_id", "date", "breakdown_type"],
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    # --- Drop breakdowns table ---
    if inspector.has_table("sistema_facebook_ads_insights_breakdowns"):
        op.drop_table("sistema_facebook_ads_insights_breakdowns")

    # --- Drop new columns from both insights tables ---
    for table_name in INSIGHTS_TABLES:
        if not inspector.has_table(table_name):
            continue

        existing_columns = {col["name"] for col in inspector.get_columns(table_name)}

        for col_name, _ in NEW_COLUMNS:
            if col_name in existing_columns:
                op.drop_column(table_name, col_name)
