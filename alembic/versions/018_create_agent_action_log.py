"""Cria tabela agent_action_log para auditoria de acoes do agente.

Revision ID: 018
Revises: 017
Create Date: 2026-02-11

Changes:
- Cria tabela agent_action_log para registrar todas as acoes
  executadas pelo ecossistema multi-agente
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSON

# revision identifiers, used by Alembic.
revision = "018"
down_revision = "017"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Cria a tabela agent_action_log com indices."""
    op.create_table(
        "agent_action_log",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("campaign_id", sa.String(100), nullable=True),
        sa.Column("operation_type", sa.String(100), nullable=False),
        sa.Column("details", JSON, nullable=True),
        sa.Column(
            "executed_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("idempotency_key", sa.String(255), nullable=True, unique=True),
        sa.Column("user_id", sa.Integer(), nullable=True),
        sa.Column("account_id", sa.Integer(), nullable=True),
    )

    # Indice em campaign_id para consultas por campanha
    op.create_index(
        "ix_agent_action_log_campaign_id",
        "agent_action_log",
        ["campaign_id"],
    )

    # Indice em operation_type para filtros por tipo de operacao
    op.create_index(
        "ix_agent_action_log_operation_type",
        "agent_action_log",
        ["operation_type"],
    )

    # Indice em executed_at para consultas temporais
    op.create_index(
        "ix_agent_action_log_executed_at",
        "agent_action_log",
        ["executed_at"],
    )

    # Indice composto para consultas por usuario e conta
    op.create_index(
        "ix_agent_action_log_user_account",
        "agent_action_log",
        ["user_id", "account_id"],
    )


def downgrade() -> None:
    """Remove a tabela agent_action_log e seus indices."""
    op.drop_index("ix_agent_action_log_user_account", table_name="agent_action_log")
    op.drop_index("ix_agent_action_log_executed_at", table_name="agent_action_log")
    op.drop_index("ix_agent_action_log_operation_type", table_name="agent_action_log")
    op.drop_index("ix_agent_action_log_campaign_id", table_name="agent_action_log")
    op.drop_table("agent_action_log")
