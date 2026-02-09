"""Cria tabela agent_user_entities para Entity Memory."""

from alembic import op
import sqlalchemy as sa

revision = "015"
down_revision = "014"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "agent_user_entities",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("user_id", sa.Integer, nullable=False, index=True),
        sa.Column("config_id", sa.Integer, nullable=False),
        sa.Column(
            "entity_type",
            sa.String(50),
            nullable=False,
            comment="campaign | metric | preference | threshold | insight",
        ),
        sa.Column("entity_key", sa.String(255), nullable=False),
        sa.Column("entity_value", sa.Text, nullable=False),
        sa.Column("confidence", sa.Float, nullable=False, server_default=sa.text("0.8")),
        sa.Column("source_thread_id", sa.String(255), nullable=True),
        sa.Column("mention_count", sa.Integer, nullable=False, server_default=sa.text("1")),
        sa.Column(
            "created_at",
            sa.DateTime,
            server_default=sa.text("NOW()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime,
            server_default=sa.text("NOW()"),
            nullable=False,
        ),
    )
    op.create_index(
        "idx_user_entities_user_type",
        "agent_user_entities",
        ["user_id", "entity_type"],
    )
    op.create_index(
        "idx_user_entities_key",
        "agent_user_entities",
        ["user_id", "entity_key"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_table("agent_user_entities")
