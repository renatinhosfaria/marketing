"""Add ml_features, ml_forecasts and model_version

Revision ID: 005_add_ml_features_forecasts
Revises: 004_add_ml_indexes
Create Date: 2025-12-11
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "005_add_ml_features_forecasts"
down_revision: Union[str, None] = "004_add_ml_indexes"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "ml_campaign_classifications",
        sa.Column("model_version", sa.String(length=50), nullable=True),
    )

    op.create_table(
        "ml_features",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("config_id", sa.Integer(), nullable=False),
        sa.Column("campaign_id", sa.String(length=100), nullable=False),
        sa.Column("window_days", sa.Integer(), nullable=False, server_default="30"),
        sa.Column("feature_date", sa.Date(), nullable=False),
        sa.Column("features", sa.JSON(), nullable=True),
        sa.Column("insufficient_data", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("generated_at", sa.DateTime(), server_default=sa.text("now()")),
    )
    op.create_index(
        "ux_ml_features_unique",
        "ml_features",
        ["config_id", "campaign_id", "window_days", "feature_date"],
        unique=True,
    )
    op.create_index(
        "ix_ml_features_config_date",
        "ml_features",
        ["config_id", "feature_date"],
        unique=False,
    )

    op.create_table(
        "ml_forecasts",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("config_id", sa.Integer(), nullable=False),
        sa.Column("entity_type", sa.String(length=50), nullable=False),
        sa.Column("entity_id", sa.String(length=100), nullable=False),
        sa.Column("target_metric", sa.String(length=50), nullable=False),
        sa.Column("horizon_days", sa.Integer(), nullable=False),
        sa.Column("method", sa.String(length=50), nullable=False),
        sa.Column("model_version", sa.String(length=50), nullable=True),
        sa.Column("window_days", sa.Integer(), nullable=True),
        sa.Column("forecast_date", sa.Date(), nullable=False),
        sa.Column("predictions", sa.JSON(), nullable=True),
        sa.Column("insufficient_data", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("now()")),
    )
    op.create_index(
        "ux_ml_forecasts_unique",
        "ml_forecasts",
        ["config_id", "entity_type", "entity_id", "target_metric", "horizon_days", "forecast_date"],
        unique=True,
    )
    op.create_index(
        "ix_ml_forecasts_config_date",
        "ml_forecasts",
        ["config_id", "forecast_date"],
        unique=False,
    )
    op.create_index(
        "ix_ml_forecasts_metric",
        "ml_forecasts",
        ["config_id", "target_metric"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_ml_forecasts_metric", table_name="ml_forecasts")
    op.drop_index("ix_ml_forecasts_config_date", table_name="ml_forecasts")
    op.drop_index("ux_ml_forecasts_unique", table_name="ml_forecasts")
    op.drop_table("ml_forecasts")

    op.drop_index("ix_ml_features_config_date", table_name="ml_features")
    op.drop_index("ux_ml_features_unique", table_name="ml_features")
    op.drop_table("ml_features")

    op.drop_column("ml_campaign_classifications", "model_version")
