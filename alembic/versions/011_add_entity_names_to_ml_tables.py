"""Add entity names to ML tables for visual identification.

Revision ID: 011
Revises: 010
Create Date: 2026-02-05

Changes:
- Create ml_classification_feedback table
- Add campaign_name, adset_name, ad_name columns to:
  - ml_predictions
  - ml_classifications
  - ml_recommendations
  - ml_anomalies
  - ml_features
  - ml_forecasts
  - ml_classification_feedback
"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "011"
down_revision = "010"
branch_labels = None
depends_on = None

# Tables that need entity name columns (existing tables only)
EXISTING_TABLES = [
    "ml_predictions",
    "ml_classifications",
    "ml_recommendations",
    "ml_anomalies",
    "ml_features",
    "ml_forecasts",
]


def upgrade() -> None:
    """Create ml_classification_feedback table and add entity name columns to all ML tables."""

    # Get the existing enum type from PostgreSQL
    from sqlalchemy.dialects import postgresql

    campaigntier_enum = postgresql.ENUM(
        "HIGH_PERFORMER", "MODERATE", "LOW", "UNDERPERFORMER",
        name="campaigntier",
        create_type=False,
    )

    # 1. Create ml_classification_feedback table
    op.create_table(
        "ml_classification_feedback",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("config_id", sa.Integer(), nullable=False),
        sa.Column("entity_id", sa.String(100), nullable=False),
        sa.Column("entity_type", sa.String(20), nullable=False, server_default="campaign"),
        sa.Column("campaign_name", sa.String(255), nullable=True),
        sa.Column("adset_name", sa.String(255), nullable=True),
        sa.Column("ad_name", sa.String(255), nullable=True),
        sa.Column("original_tier", campaigntier_enum, nullable=False),
        sa.Column(
            "original_classification_id",
            sa.Integer(),
            sa.ForeignKey("ml_classifications.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("correct_tier", campaigntier_enum, nullable=False),
        sa.Column("feedback_reason", sa.String(500), nullable=True),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("is_valid", sa.Boolean(), server_default="true"),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
    )

    # Create indexes for ml_classification_feedback
    op.create_index(
        "ix_ml_feedback_config_entity",
        "ml_classification_feedback",
        ["config_id", "entity_type", "entity_id"],
    )
    op.create_index(
        "ix_ml_feedback_user",
        "ml_classification_feedback",
        ["user_id"],
    )
    op.create_index(
        "ix_ml_feedback_valid",
        "ml_classification_feedback",
        ["config_id", "is_valid"],
    )

    # 2. Add entity name columns to existing tables
    for table_name in EXISTING_TABLES:
        op.add_column(
            table_name,
            sa.Column("campaign_name", sa.String(255), nullable=True),
        )
        op.add_column(
            table_name,
            sa.Column("adset_name", sa.String(255), nullable=True),
        )
        op.add_column(
            table_name,
            sa.Column("ad_name", sa.String(255), nullable=True),
        )


def downgrade() -> None:
    """Remove entity name columns and drop ml_classification_feedback table."""

    # 1. Remove entity name columns from existing tables
    for table_name in EXISTING_TABLES:
        op.drop_column(table_name, "ad_name")
        op.drop_column(table_name, "adset_name")
        op.drop_column(table_name, "campaign_name")

    # 2. Drop ml_classification_feedback table
    op.drop_index("ix_ml_feedback_valid", table_name="ml_classification_feedback")
    op.drop_index("ix_ml_feedback_user", table_name="ml_classification_feedback")
    op.drop_index("ix_ml_feedback_config_entity", table_name="ml_classification_feedback")
    op.drop_table("ml_classification_feedback")
