"""Testes para SummarizationService."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from projects.agent.memory.summarization import (
    SummarizationService,
    should_summarize,
    build_summarization_prompt,
)


class TestShouldSummarize:
    def test_below_threshold_returns_false(self):
        messages = [HumanMessage(content="oi")] * 10
        assert should_summarize(messages, threshold=20) is False

    def test_above_threshold_returns_true(self):
        messages = [HumanMessage(content="oi")] * 25
        assert should_summarize(messages, threshold=20) is True

    def test_exactly_at_threshold_returns_false(self):
        messages = [HumanMessage(content="oi")] * 20
        assert should_summarize(messages, threshold=20) is False

    def test_system_messages_not_counted(self):
        messages = [SystemMessage(content="system")] * 15 + [HumanMessage(content="oi")] * 10
        assert should_summarize(messages, threshold=20) is False


class TestBuildSummarizationPrompt:
    def test_includes_messages(self):
        messages = [
            HumanMessage(content="Como estao minhas campanhas?"),
            AIMessage(content="Suas campanhas estao indo bem."),
        ]
        prompt = build_summarization_prompt(messages)
        assert "campanhas" in prompt
        assert "Usuario:" in prompt
        assert "Assistente:" in prompt


@pytest.mark.asyncio
class TestSummarizationService:
    async def test_summarize_messages_calls_llm(self):
        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = MagicMock(
            content="Resumo: usuario perguntou sobre campanhas."
        )

        service = SummarizationService()
        messages = [
            HumanMessage(content="Como estao?"),
            AIMessage(content="Suas campanhas estao bem."),
        ]

        with patch("projects.agent.llm.provider.get_llm", return_value=mock_llm):
            result = await service.summarize_messages(messages)

        assert "Resumo" in result
        mock_llm.ainvoke.assert_called_once()

    async def test_get_or_create_summary_creates_new(self):
        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(
            return_value=MagicMock(scalar_one_or_none=MagicMock(return_value=None))
        )
        mock_session.add = MagicMock()
        mock_session.commit = AsyncMock()

        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = MagicMock(content="Resumo da conversa.")

        service = SummarizationService()

        messages = [HumanMessage(content=f"msg {i}") for i in range(25)]

        with patch("projects.agent.llm.provider.get_llm", return_value=mock_llm):
            with patch(
                "projects.agent.memory.summarization.async_session_maker",
                return_value=AsyncMock(
                    __aenter__=AsyncMock(return_value=mock_session),
                    __aexit__=AsyncMock(),
                ),
            ):
                summary, trimmed = await service.get_or_create_summary(
                    thread_id="t-1",
                    user_id=1,
                    config_id=1,
                    messages=messages,
                    threshold=20,
                    keep_recent=10,
                )

        assert summary is not None
        assert len(trimmed) == 10  # Apenas mensagens recentes mantidas
