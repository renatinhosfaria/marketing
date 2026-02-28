"""Create audit table for FB Ads agent SQL queries.

Revision ID: 021
Revises: 020
Create Date: 2026-02-28
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB


revision = "021"
down_revision = "020"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "fbads_agent_query_audit",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column(
            "requested_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "requested_by",
            sa.String(length=100),
            nullable=False,
            server_default=sa.text("'fbads'"),
        ),
        sa.Column("prompt", sa.Text(), nullable=False),
        sa.Column("generated_sql", sa.Text(), nullable=False),
        sa.Column("operation_type", sa.String(length=20), nullable=False),
        sa.Column("execution_status", sa.String(length=20), nullable=False),
        sa.Column("rows_affected", sa.Integer(), nullable=True),
        sa.Column("duration_ms", sa.Integer(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("metadata", JSONB(), nullable=True),
    )

    op.create_index(
        "ix_fbads_agent_query_audit_requested_at",
        "fbads_agent_query_audit",
        ["requested_at"],
    )
    op.create_index(
        "ix_fbads_agent_query_audit_operation_type",
        "fbads_agent_query_audit",
        ["operation_type"],
    )
    op.create_index(
        "ix_fbads_agent_query_audit_execution_status",
        "fbads_agent_query_audit",
        ["execution_status"],
    )


def downgrade() -> None:
    op.drop_index("ix_fbads_agent_query_audit_execution_status", table_name="fbads_agent_query_audit")
    op.drop_index("ix_fbads_agent_query_audit_operation_type", table_name="fbads_agent_query_audit")
    op.drop_index("ix_fbads_agent_query_audit_requested_at", table_name="fbads_agent_query_audit")
    op.drop_table("fbads_agent_query_audit")
