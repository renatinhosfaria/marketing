"""Add multi-level ML support.

Revision ID: 010
Revises: 009
Create Date: 2026-02-03

Changes:
- Rename ml_campaign_classifications to ml_classifications
- Add entity_type and parent_id columns to classifications and features
- Rename campaign_id to entity_id in both tables
- Update indexes to support multi-level queries
"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "010"
down_revision = "009"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ==========================================================================
    # 1. ml_campaign_classifications -> ml_classifications
    # ==========================================================================

    # Drop old indexes (they reference old table/column names)
    op.drop_index(
        "ix_ml_classifications_config_campaign",
        table_name="ml_campaign_classifications",
        if_exists=True,
    )
    op.drop_index(
        "ix_ml_classifications_tier",
        table_name="ml_campaign_classifications",
        if_exists=True,
    )
    op.drop_index(
        "ix_ml_classifications_config_classified_at",
        table_name="ml_campaign_classifications",
        if_exists=True,
    )

    # Rename table
    op.rename_table("ml_campaign_classifications", "ml_classifications")

    # Add entity_type column with default 'campaign'
    op.add_column(
        "ml_classifications",
        sa.Column(
            "entity_type",
            sa.String(20),
            nullable=False,
            server_default="campaign",
        ),
    )

    # Add parent_id column (nullable)
    op.add_column(
        "ml_classifications",
        sa.Column("parent_id", sa.String(100), nullable=True),
    )

    # Rename campaign_id to entity_id
    op.alter_column(
        "ml_classifications",
        "campaign_id",
        new_column_name="entity_id",
    )

    # Create new indexes for multi-level queries
    op.create_index(
        "ix_ml_classifications_entity",
        "ml_classifications",
        ["config_id", "entity_type", "entity_id", "classified_at"],
    )
    op.create_index(
        "ix_ml_classifications_tier",
        "ml_classifications",
        ["config_id", "tier"],
    )
    op.create_index(
        "ix_ml_classifications_parent",
        "ml_classifications",
        ["config_id", "parent_id"],
    )

    # ==========================================================================
    # 2. ml_features
    # ==========================================================================

    # Drop old indexes
    op.drop_index(
        "ux_ml_features_unique",
        table_name="ml_features",
        if_exists=True,
    )
    op.drop_index(
        "ix_ml_features_config_date",
        table_name="ml_features",
        if_exists=True,
    )

    # Add entity_type column with default 'campaign'
    op.add_column(
        "ml_features",
        sa.Column(
            "entity_type",
            sa.String(20),
            nullable=False,
            server_default="campaign",
        ),
    )

    # Add parent_id column (nullable)
    op.add_column(
        "ml_features",
        sa.Column("parent_id", sa.String(100), nullable=True),
    )

    # Rename campaign_id to entity_id
    op.alter_column(
        "ml_features",
        "campaign_id",
        new_column_name="entity_id",
    )

    # Create new indexes for multi-level queries
    op.create_index(
        "ix_ml_features_entity",
        "ml_features",
        ["config_id", "entity_type", "entity_id", "feature_date"],
    )
    op.create_index(
        "ix_ml_features_window",
        "ml_features",
        ["config_id", "window_days"],
    )
    op.create_index(
        "ix_ml_features_parent",
        "ml_features",
        ["config_id", "parent_id"],
    )


def downgrade() -> None:
    # ==========================================================================
    # 1. ml_features - reverse changes
    # ==========================================================================

    # Drop new indexes
    op.drop_index("ix_ml_features_parent", table_name="ml_features", if_exists=True)
    op.drop_index("ix_ml_features_window", table_name="ml_features", if_exists=True)
    op.drop_index("ix_ml_features_entity", table_name="ml_features", if_exists=True)

    # Rename entity_id back to campaign_id
    op.alter_column(
        "ml_features",
        "entity_id",
        new_column_name="campaign_id",
    )

    # Drop new columns
    op.drop_column("ml_features", "parent_id")
    op.drop_column("ml_features", "entity_type")

    # Recreate old indexes
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
    )

    # ==========================================================================
    # 2. ml_classifications -> ml_campaign_classifications
    # ==========================================================================

    # Drop new indexes
    op.drop_index(
        "ix_ml_classifications_parent",
        table_name="ml_classifications",
        if_exists=True,
    )
    op.drop_index(
        "ix_ml_classifications_tier",
        table_name="ml_classifications",
        if_exists=True,
    )
    op.drop_index(
        "ix_ml_classifications_entity",
        table_name="ml_classifications",
        if_exists=True,
    )

    # Rename entity_id back to campaign_id
    op.alter_column(
        "ml_classifications",
        "entity_id",
        new_column_name="campaign_id",
    )

    # Drop new columns
    op.drop_column("ml_classifications", "parent_id")
    op.drop_column("ml_classifications", "entity_type")

    # Rename table back
    op.rename_table("ml_classifications", "ml_campaign_classifications")

    # Recreate old indexes
    op.create_index(
        "ix_ml_classifications_config_campaign",
        "ml_campaign_classifications",
        ["config_id", "campaign_id"],
    )
    op.create_index(
        "ix_ml_classifications_tier",
        "ml_campaign_classifications",
        ["tier"],
    )
    op.create_index(
        "ix_ml_classifications_config_classified_at",
        "ml_campaign_classifications",
        ["config_id", "classified_at"],
    )
