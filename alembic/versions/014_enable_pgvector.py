"""Habilita extensao pgvector e cria tabela de embeddings."""

from alembic import op
import sqlalchemy as sa

revision = "014"
down_revision = "013"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Habilitar extensao pgvector
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "agent_memory_embeddings",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("user_id", sa.Integer, nullable=False, index=True),
        sa.Column("config_id", sa.Integer, nullable=False),
        sa.Column("thread_id", sa.String(255), nullable=False, index=True),
        sa.Column(
            "source_type",
            sa.String(50),
            nullable=False,
            comment="message | summary | entity",
        ),
        sa.Column("source_id", sa.String(255), nullable=True),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("metadata_", sa.JSON, nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime,
            server_default=sa.text("NOW()"),
            nullable=False,
        ),
    )

    # Criar coluna vector nativa via raw SQL (pgvector)
    op.execute(
        "ALTER TABLE agent_memory_embeddings "
        "ADD COLUMN IF NOT EXISTS embedding_vector vector(1536)"
    )

    # Indice HNSW para busca rapida de similaridade
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_memory_embeddings_vector "
        "ON agent_memory_embeddings USING hnsw (embedding_vector vector_cosine_ops)"
    )

    op.create_index(
        "idx_memory_embeddings_user_type",
        "agent_memory_embeddings",
        ["user_id", "source_type"],
    )


def downgrade() -> None:
    op.drop_table("agent_memory_embeddings")
    op.execute("DROP EXTENSION IF EXISTS vector")
