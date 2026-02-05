"""Criar tabelas do módulo Facebook Ads (ML).

Revision ID: 008
Revises: 007_add_prediction_type_to_forecasts
Create Date: 2026-01-29

Tabelas criadas:
- sistema_facebook_ads_sync_history: Histórico de sincronizações
- ml_facebook_ads_management_log: Audit trail de ações
- ml_facebook_ads_rate_limit_log: Log de rate limiting
"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "008"
down_revision = "007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    # Tabela de histórico de sincronizações
    if not inspector.has_table("sistema_facebook_ads_sync_history"):
        op.create_table(
            "sistema_facebook_ads_sync_history",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column("config_id", sa.Integer(), nullable=False, index=True),
            sa.Column("sync_type", sa.String(50), nullable=False, server_default="full"),
            sa.Column("status", sa.String(50), nullable=False, server_default="pending"),
            sa.Column("campaigns_synced", sa.Integer(), server_default="0"),
            sa.Column("adsets_synced", sa.Integer(), server_default="0"),
            sa.Column("ads_synced", sa.Integer(), server_default="0"),
            sa.Column("insights_synced", sa.Integer(), server_default="0"),
            sa.Column("error_message", sa.Text()),
            sa.Column("details", sa.JSON()),
            sa.Column("started_at", sa.DateTime()),
            sa.Column("completed_at", sa.DateTime()),
            sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
            sa.Column("progress_pct", sa.Integer(), server_default="0"),
            sa.Column("current_step", sa.String(100)),
        )

    if inspector.has_table("sistema_facebook_ads_sync_history"):
        op.execute(
            "CREATE INDEX IF NOT EXISTS ix_fb_sync_history_config_status "
            "ON sistema_facebook_ads_sync_history (config_id, status)"
        )

    # Tabela de audit trail de ações de gerenciamento
    if not inspector.has_table("ml_facebook_ads_management_log"):
        op.create_table(
            "ml_facebook_ads_management_log",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column("config_id", sa.Integer(), nullable=False, index=True),
            sa.Column("user_id", sa.Integer(), nullable=False),
            sa.Column("action", sa.String(100), nullable=False),
            sa.Column("entity_type", sa.String(50), nullable=False),
            sa.Column("entity_id", sa.String(100), nullable=False),
            sa.Column("before_state", sa.JSON()),
            sa.Column("after_state", sa.JSON()),
            sa.Column("request_data", sa.JSON()),
            sa.Column("response_data", sa.JSON()),
            sa.Column("success", sa.Boolean(), server_default="true"),
            sa.Column("error_message", sa.Text()),
            sa.Column("source", sa.String(50), server_default="manual"),
            sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
        )

    if inspector.has_table("ml_facebook_ads_management_log"):
        op.execute(
            "CREATE INDEX IF NOT EXISTS ix_fb_mgmt_log_entity "
            "ON ml_facebook_ads_management_log (entity_type, entity_id)"
        )

    # Tabela de log de rate limiting
    if not inspector.has_table("ml_facebook_ads_rate_limit_log"):
        op.create_table(
            "ml_facebook_ads_rate_limit_log",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column("config_id", sa.Integer()),
            sa.Column("account_id", sa.String(100)),
            sa.Column("call_count_pct", sa.Numeric(5, 2)),
            sa.Column("total_cputime_pct", sa.Numeric(5, 2)),
            sa.Column("total_time_pct", sa.Numeric(5, 2)),
            sa.Column("max_usage_pct", sa.Numeric(5, 2)),
            sa.Column("action_taken", sa.String(50), nullable=False),
            sa.Column("wait_seconds", sa.Numeric(10, 2)),
            sa.Column("endpoint", sa.String(255)),
            sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
        )

    if inspector.has_table("ml_facebook_ads_rate_limit_log"):
        op.execute(
            "CREATE INDEX IF NOT EXISTS ix_fb_rate_limit_account "
            "ON ml_facebook_ads_rate_limit_log (account_id, created_at)"
        )


def downgrade() -> None:
    op.drop_table("ml_facebook_ads_rate_limit_log")
    op.drop_table("ml_facebook_ads_management_log")
    op.drop_table("sistema_facebook_ads_sync_history")
