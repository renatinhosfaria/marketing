"""Add prediction_type to ml_forecasts

Revision ID: 007_add_prediction_type_to_forecasts
Revises: 006_create_agent_tables
Create Date: 2026-01-16
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "007"
down_revision: Union[str, None] = "006"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Criar o tipo enum se não existir
    op.execute("""
        DO $$ BEGIN
            CREATE TYPE predictiontype AS ENUM ('CPL_FORECAST', 'LEADS_FORECAST', 'SPEND_FORECAST');
        EXCEPTION
            WHEN duplicate_object THEN null;
        END $$;
    """)

    # Adicionar coluna prediction_type
    op.add_column(
        "ml_forecasts",
        sa.Column(
            "prediction_type",
            sa.Enum('CPL_FORECAST', 'LEADS_FORECAST', 'SPEND_FORECAST', name='predictiontype', create_type=False),
            nullable=True
        ),
    )


def downgrade() -> None:
    op.drop_column("ml_forecasts", "prediction_type")
    # Não removemos o tipo enum para evitar problemas com outras tabelas
