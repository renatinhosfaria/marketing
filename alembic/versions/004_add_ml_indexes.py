"""Add indexes for ML summary/list queries

Revision ID: 004_add_ml_indexes
Revises: 003_add_recommendation_id
Create Date: 2025-12-11
"""
from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = "004_add_ml_indexes"
down_revision: Union[str, None] = "003_add_recommendation_id"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_index(
        "ix_ml_recommendations_config_expires",
        "ml_recommendations",
        ["config_id", "expires_at"],
        unique=False,
    )
    op.create_index(
        "ix_ml_recommendations_config_created",
        "ml_recommendations",
        ["config_id", "created_at"],
        unique=False,
    )
    op.create_index(
        "ix_ml_anomalies_config_detected",
        "ml_anomalies",
        ["config_id", "detected_at"],
        unique=False,
    )
    op.create_index(
        "ix_ml_classifications_config_classified_at",
        "ml_campaign_classifications",
        ["config_id", "classified_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_ml_classifications_config_classified_at", table_name="ml_campaign_classifications")
    op.drop_index("ix_ml_anomalies_config_detected", table_name="ml_anomalies")
    op.drop_index("ix_ml_recommendations_config_created", table_name="ml_recommendations")
    op.drop_index("ix_ml_recommendations_config_expires", table_name="ml_recommendations")
