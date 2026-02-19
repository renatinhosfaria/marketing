"""Adiciona coluna created_at na tabela checkpoints do LangGraph.

Revision ID: 019
Revises: 018
Create Date: 2026-02-17

Changes:
- Adiciona coluna created_at (TIMESTAMPTZ) com default now()
  na tabela checkpoints, necessaria para listar conversas
  ordenadas por data no endpoint GET /conversations.
- A tabela checkpoints e criada pelo LangGraph (checkpointer.setup()),
  mas nao inclui created_at por padrao.
"""

from alembic import op
import sqlalchemy as sa


revision = "019"
down_revision = "018"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # A tabela pode nao existir se o agent nunca rodou neste banco.
    # Nesse caso o checkpointer.setup() criara a tabela e o lifespan
    # adicionara a coluna. Usamos IF EXISTS para seguranca.
    op.execute(
        "ALTER TABLE IF EXISTS checkpoints "
        "ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ NOT NULL DEFAULT now()"
    )


def downgrade() -> None:
    op.execute(
        "ALTER TABLE IF EXISTS checkpoints "
        "DROP COLUMN IF EXISTS created_at"
    )
