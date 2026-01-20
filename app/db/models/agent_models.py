"""
Modelos SQLAlchemy para o Agente de IA.
Tabelas de conversas, mensagens, checkpoints e feedback.
"""

import enum
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Integer,
    String,
    Text,
    Boolean,
    DateTime,
    JSON,
    LargeBinary,
    Enum,
    ForeignKey,
    Index,
    CheckConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.session import Base


# ==================== ENUMS ====================

class MessageRole(str, enum.Enum):
    """Roles de mensagens no chat."""
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


# ==================== MODELOS ====================

class AgentConversation(Base):
    """
    Conversas do agente de IA.
    Cada conversa pertence a um usuário e uma configuração de conta Facebook Ads.
    """
    __tablename__ = "agent_conversations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    thread_id: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    config_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    user_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    title: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    message_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relacionamentos
    messages: Mapped[list["AgentMessage"]] = relationship(
        "AgentMessage",
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="AgentMessage.created_at"
    )

    __table_args__ = (
        Index("idx_agent_conversations_user", "user_id"),
        Index("idx_agent_conversations_config", "config_id"),
        Index("idx_agent_conversations_created", "created_at"),
    )


class AgentMessage(Base):
    """
    Mensagens das conversas do agente.
    Armazena mensagens do usuário, respostas do assistente e resultados de tools.
    """
    __tablename__ = "agent_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    conversation_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("agent_conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    role: Mapped[MessageRole] = mapped_column(
        Enum(
            MessageRole,
            name='messagerole',
            create_type=False,  # Usar tipo existente no PostgreSQL
            values_callable=lambda x: [e.value for e in x]  # Usar valores (user, assistant) em vez de nomes (USER, ASSISTANT)
        ),
        nullable=False
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    tool_calls: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    tool_results: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    tokens_used: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Relacionamentos
    conversation: Mapped["AgentConversation"] = relationship(
        "AgentConversation",
        back_populates="messages"
    )
    feedback: Mapped[Optional["AgentFeedback"]] = relationship(
        "AgentFeedback",
        back_populates="message",
        uselist=False,
        cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("idx_agent_messages_conversation", "conversation_id"),
        Index("idx_agent_messages_created", "created_at"),
    )


class AgentCheckpoint(Base):
    """
    Checkpoints do LangGraph para persistência de estado.
    Usado pelo PostgresSaver para manter o estado do grafo entre requisições.
    """
    __tablename__ = "agent_checkpoints"

    thread_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    thread_ts: Mapped[datetime] = mapped_column(DateTime, primary_key=True)
    checkpoint: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    metadata_: Mapped[Optional[dict]] = mapped_column("metadata", JSON, nullable=True)

    __table_args__ = (
        Index("idx_agent_checkpoints_thread", "thread_id"),
    )


class AgentWrite(Base):
    """
    Writes incrementais do LangGraph.
    Armazena as mudanças de estado para checkpoint incremental.
    """
    __tablename__ = "agent_writes"

    thread_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    thread_ts: Mapped[datetime] = mapped_column(DateTime, primary_key=True)
    task_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    idx: Mapped[int] = mapped_column(Integer, primary_key=True)
    channel: Mapped[str] = mapped_column(String(255), nullable=False)
    value: Mapped[Optional[bytes]] = mapped_column(LargeBinary, nullable=True)


class AgentFeedback(Base):
    """
    Feedback do usuário sobre respostas do agente.
    Usado para métricas de qualidade e melhoria contínua.
    """
    __tablename__ = "agent_feedback"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    message_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("agent_messages.id", ondelete="CASCADE"),
        nullable=False,
        unique=True
    )
    user_id: Mapped[int] = mapped_column(Integer, nullable=False)
    rating: Mapped[int] = mapped_column(Integer, nullable=False)
    feedback_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Relacionamento
    message: Mapped["AgentMessage"] = relationship(
        "AgentMessage",
        back_populates="feedback"
    )

    __table_args__ = (
        CheckConstraint("rating >= 1 AND rating <= 5", name="check_rating_range"),
        Index("idx_agent_feedback_user", "user_id"),
    )
