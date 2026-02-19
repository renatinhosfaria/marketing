"""Reforca contrato de idempotencia/auditoria no agent_action_log.

Revision ID: 021
Revises: 020
Create Date: 2026-02-18

Changes:
- Adiciona coluna status para trilha de execucao (executing/executed/failed)
- Adiciona indice composto para consultas de cooldown por campanha/tipo/tempo/status
- Adiciona check constraint (NOT VALID) para exigir idempotency_key em registros executaveis
"""

from alembic import op
import sqlalchemy as sa


revision = "021"
down_revision = "020"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "agent_action_log",
        sa.Column(
            "status",
            sa.String(length=32),
            nullable=False,
            server_default=sa.text("'executed'"),
        ),
    )

    op.create_index(
        "ix_agent_action_log_campaign_op_executed_status",
        "agent_action_log",
        ["campaign_id", "operation_type", "executed_at", "status"],
    )

    op.execute(
        """
        ALTER TABLE agent_action_log
        ADD CONSTRAINT ck_agent_action_log_idempotency_required
        CHECK (
            status NOT IN ('executing', 'executed', 'success')
            OR idempotency_key IS NOT NULL
        ) NOT VALID
        """
    )


def downgrade() -> None:
    op.execute(
        "ALTER TABLE agent_action_log "
        "DROP CONSTRAINT IF EXISTS ck_agent_action_log_idempotency_required"
    )
    op.drop_index(
        "ix_agent_action_log_campaign_op_executed_status",
        table_name="agent_action_log",
    )
    op.drop_column("agent_action_log", "status")
