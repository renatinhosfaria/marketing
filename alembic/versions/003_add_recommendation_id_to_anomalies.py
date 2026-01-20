"""Add recommendation_id to ml_anomalies

Revision ID: 003_add_recommendation_id
Revises: 20241211_001
Create Date: 2025-12-11

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '003_add_recommendation_id'
down_revision: Union[str, None] = '20241211_001'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add recommendation_id column to ml_anomalies
    op.execute("""
        ALTER TABLE ml_anomalies 
        ADD COLUMN IF NOT EXISTS recommendation_id INTEGER 
        REFERENCES ml_recommendations(id) ON DELETE SET NULL
    """)


def downgrade() -> None:
    op.execute("""
        ALTER TABLE ml_anomalies DROP COLUMN IF EXISTS recommendation_id
    """)
