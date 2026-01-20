"""
Criar tabelas do Agente de IA.

Revision ID: 006
Revises: 005_add_ml_features_forecasts
Create Date: 2026-01-11

Tabelas criadas:
- agent_conversations: Conversas do agente
- agent_messages: Mensagens das conversas
- agent_checkpoints: Checkpoints do LangGraph
- agent_writes: Writes incrementais do LangGraph
- agent_feedback: Feedback do usuário
"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '006'
down_revision = '005_add_ml_features_forecasts'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Criar tabelas do agente de IA."""

    # ==================== ENUM ====================
    op.execute("""
        DO $$ BEGIN
            CREATE TYPE messagerole AS ENUM ('user', 'assistant', 'tool');
        EXCEPTION
            WHEN duplicate_object THEN NULL;
        END $$;
    """)

    # ==================== TABELA: agent_conversations ====================
    op.execute("""
        CREATE TABLE IF NOT EXISTS agent_conversations (
            id SERIAL PRIMARY KEY,
            thread_id VARCHAR(255) UNIQUE NOT NULL,
            config_id INTEGER NOT NULL,
            user_id INTEGER NOT NULL,
            title VARCHAR(255),
            message_count INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT NOW(),
            updated_at TIMESTAMP DEFAULT NOW()
        );
    """)

    # Índices para agent_conversations
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_agent_conversations_thread ON agent_conversations(thread_id);
        CREATE INDEX IF NOT EXISTS idx_agent_conversations_user ON agent_conversations(user_id);
        CREATE INDEX IF NOT EXISTS idx_agent_conversations_config ON agent_conversations(config_id);
        CREATE INDEX IF NOT EXISTS idx_agent_conversations_created ON agent_conversations(created_at);
    """)

    # ==================== TABELA: agent_messages ====================
    op.execute("""
        CREATE TABLE IF NOT EXISTS agent_messages (
            id SERIAL PRIMARY KEY,
            conversation_id INTEGER NOT NULL REFERENCES agent_conversations(id) ON DELETE CASCADE,
            role messagerole NOT NULL,
            content TEXT NOT NULL,
            tool_calls JSONB,
            tool_results JSONB,
            tokens_used INTEGER,
            created_at TIMESTAMP DEFAULT NOW()
        );
    """)

    # Índices para agent_messages
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_agent_messages_conversation ON agent_messages(conversation_id);
        CREATE INDEX IF NOT EXISTS idx_agent_messages_created ON agent_messages(created_at);
    """)

    # ==================== TABELA: agent_checkpoints ====================
    op.execute("""
        CREATE TABLE IF NOT EXISTS agent_checkpoints (
            thread_id VARCHAR(255) NOT NULL,
            thread_ts TIMESTAMP NOT NULL,
            checkpoint BYTEA NOT NULL,
            metadata JSONB,
            PRIMARY KEY (thread_id, thread_ts)
        );
    """)

    # Índice para agent_checkpoints
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_agent_checkpoints_thread ON agent_checkpoints(thread_id);
    """)

    # ==================== TABELA: agent_writes ====================
    op.execute("""
        CREATE TABLE IF NOT EXISTS agent_writes (
            thread_id VARCHAR(255) NOT NULL,
            thread_ts TIMESTAMP NOT NULL,
            task_id VARCHAR(255) NOT NULL,
            idx INTEGER NOT NULL,
            channel VARCHAR(255) NOT NULL,
            value BYTEA,
            PRIMARY KEY (thread_id, thread_ts, task_id, idx)
        );
    """)

    # ==================== TABELA: agent_feedback ====================
    op.execute("""
        CREATE TABLE IF NOT EXISTS agent_feedback (
            id SERIAL PRIMARY KEY,
            message_id INTEGER NOT NULL UNIQUE REFERENCES agent_messages(id) ON DELETE CASCADE,
            user_id INTEGER NOT NULL,
            rating INTEGER NOT NULL CHECK (rating >= 1 AND rating <= 5),
            feedback_text TEXT,
            created_at TIMESTAMP DEFAULT NOW()
        );
    """)

    # Índice para agent_feedback
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_agent_feedback_user ON agent_feedback(user_id);
    """)

    # ==================== COMENTÁRIOS ====================
    op.execute("""
        COMMENT ON TABLE agent_conversations IS 'Conversas do agente de IA com usuários';
        COMMENT ON TABLE agent_messages IS 'Mensagens das conversas (user, assistant, tool)';
        COMMENT ON TABLE agent_checkpoints IS 'Checkpoints do LangGraph para persistência de estado';
        COMMENT ON TABLE agent_writes IS 'Writes incrementais do LangGraph';
        COMMENT ON TABLE agent_feedback IS 'Feedback dos usuários sobre respostas do agente';
    """)


def downgrade() -> None:
    """Remover tabelas do agente."""
    op.execute("DROP TABLE IF EXISTS agent_feedback CASCADE;")
    op.execute("DROP TABLE IF EXISTS agent_writes CASCADE;")
    op.execute("DROP TABLE IF EXISTS agent_checkpoints CASCADE;")
    op.execute("DROP TABLE IF EXISTS agent_messages CASCADE;")
    op.execute("DROP TABLE IF EXISTS agent_conversations CASCADE;")
    op.execute("DROP TYPE IF EXISTS messagerole CASCADE;")
