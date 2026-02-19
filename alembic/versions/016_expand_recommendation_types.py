"""Adiciona novos valores ao enum recommendationtype para suporte multi-nivel."""

from alembic import op

revision = "016"
down_revision = "011"
branch_labels = None
depends_on = None

# Novos valores que precisam ser adicionados ao enum PostgreSQL
NEW_VALUES = [
    "PAUSE",
    "AUDIENCE_EXPANSION",
    "AUDIENCE_NARROWING",
    "CREATIVE_TEST",
    "CREATIVE_WINNER",
]


def upgrade() -> None:
    for value in NEW_VALUES:
        op.execute(f"ALTER TYPE recommendationtype ADD VALUE IF NOT EXISTS '{value}'")


def downgrade() -> None:
    # PostgreSQL nao suporta remover valores de enum diretamente.
    # Para reverter seria necessario recriar o tipo, o que e destrutivo.
    pass
