"""Cria tabela agent_conversation_summaries para memoria de sumarizacao."""

from alembic import op
import sqlalchemy as sa

revision = "013"
down_revision = "012"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "agent_conversation_summaries",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("thread_id", sa.String(255), nullable=False),
        sa.Column("user_id", sa.Integer, nullable=False),
        sa.Column("config_id", sa.Integer, nullable=False),
        sa.Column("summary_text", sa.Text, nullable=False),
        sa.Column("token_count", sa.Integer, nullable=False, server_default=sa.text("0")),
        sa.Column("messages_summarized", sa.Integer, nullable=False, server_default=sa.text("0")),
        sa.Column("last_message_index", sa.Integer, nullable=False, server_default=sa.text("0")),
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
        "idx_conv_summaries_thread",
        "agent_conversation_summaries",
        ["thread_id"],
        unique=True,
    )
    op.create_index(
        "idx_conv_summaries_user",
        "agent_conversation_summaries",
        ["user_id"],
    )


def downgrade() -> None:
    op.drop_table("agent_conversation_summaries")
