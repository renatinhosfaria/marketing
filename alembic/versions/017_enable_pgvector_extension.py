"""Habilita extensao pgvector para suporte a embeddings vetoriais.

Revision ID: 017
Revises: 016
Create Date: 2026-02-11

Changes:
- Habilita a extensao pgvector no PostgreSQL
"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "017"
down_revision = "016"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Cria a extensao vector se nao existir."""
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")


def downgrade() -> None:
    """Remove a extensao vector."""
    op.execute("DROP EXTENSION IF EXISTS vector")
