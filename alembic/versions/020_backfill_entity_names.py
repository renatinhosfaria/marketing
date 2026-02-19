"""Backfill entity names (campaign_name, adset_name, ad_name) em tabelas ML.

Revision ID: 020
Revises: 019
Create Date: 2026-02-18

Changes:
- Popula campaign_name em registros ML antigos que foram criados
  antes da feature de nomes ser implementada.
- Faz JOIN com tabelas de Facebook Ads para buscar nomes.
- Processa em batch para nao sobrecarregar o banco.
"""

from alembic import op


revision = "020"
down_revision = "019"
branch_labels = None
depends_on = None


# Tabelas ML que possuem campaign_name, adset_name, ad_name
ML_TABLES = [
    "ml_classifications",
    "ml_recommendations",
    "ml_anomalies",
    "ml_features",
    "ml_forecasts",
    "ml_predictions",
]


def upgrade() -> None:
    # Backfill campaign_name via JOIN com tabela de campanhas
    for table in ML_TABLES:
        op.execute(f"""
            UPDATE {table} t
            SET campaign_name = c.name
            FROM sistema_facebook_ads_campaigns c
            WHERE t.campaign_name IS NULL
              AND t.entity_type = 'campaign'
              AND t.entity_id = c.campaign_id
              AND t.config_id = c.config_id
        """)

    # Backfill adset_name + campaign_name para registros de adset
    for table in ML_TABLES:
        op.execute(f"""
            UPDATE {table} t
            SET adset_name = a.name,
                campaign_name = COALESCE(t.campaign_name, c.name)
            FROM sistema_facebook_ads_adsets a
            LEFT JOIN sistema_facebook_ads_campaigns c
              ON a.campaign_id = c.campaign_id AND a.config_id = c.config_id
            WHERE (t.adset_name IS NULL OR t.campaign_name IS NULL)
              AND t.entity_type = 'adset'
              AND t.entity_id = a.adset_id
              AND t.config_id = a.config_id
        """)

    # Backfill ad_name + adset_name + campaign_name para registros de ad
    for table in ML_TABLES:
        op.execute(f"""
            UPDATE {table} t
            SET ad_name = ad.name,
                adset_name = COALESCE(t.adset_name, a.name),
                campaign_name = COALESCE(t.campaign_name, c.name)
            FROM sistema_facebook_ads_ads ad
            LEFT JOIN sistema_facebook_ads_adsets a
              ON ad.adset_id = a.adset_id AND ad.config_id = a.config_id
            LEFT JOIN sistema_facebook_ads_campaigns c
              ON a.campaign_id = c.campaign_id AND a.config_id = c.config_id
            WHERE (t.ad_name IS NULL OR t.adset_name IS NULL OR t.campaign_name IS NULL)
              AND t.entity_type = 'ad'
              AND t.entity_id = ad.ad_id
              AND t.config_id = ad.config_id
        """)


def downgrade() -> None:
    # Nao e possivel reverter com precisao (nao sabemos quais eram NULL antes).
    # Deixar os nomes populados — sao dados complementares, nao estruturais.
    pass
