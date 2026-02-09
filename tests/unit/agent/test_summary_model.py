"""Testes para o modelo ConversationSummary."""
import pytest
from projects.agent.db.models import ConversationSummary


class TestConversationSummaryModel:
    def test_create_summary(self):
        summary = ConversationSummary(
            thread_id="test-thread-123",
            user_id=1,
            config_id=1,
            summary_text="Resumo da conversa sobre performance de campanhas.",
            token_count=50,
            messages_summarized=15,
            last_message_index=15,
        )
        assert summary.thread_id == "test-thread-123"
        assert summary.summary_text.startswith("Resumo")
        assert summary.token_count == 50
        assert summary.messages_summarized == 15

    def test_tablename(self):
        assert ConversationSummary.__tablename__ == "agent_conversation_summaries"
