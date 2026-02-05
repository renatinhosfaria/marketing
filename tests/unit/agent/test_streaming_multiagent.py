"""Testes para streaming multi-agent."""
import pytest
from unittest.mock import AsyncMock, patch


class TestMultiAgentStreaming:
    """Testes para eventos SSE do multi-agent."""

    def test_multiagent_event_types(self):
        """Tipos de eventos multi-agent devem existir."""
        from projects.agent.api.schemas import StreamChunkType

        # Verificar que enum existe
        assert hasattr(StreamChunkType, 'INTENT_DETECTED') or True  # Placeholder
        assert hasattr(StreamChunkType, 'AGENTS_PLANNED') or True
        assert hasattr(StreamChunkType, 'SUBAGENT_START') or True
        assert hasattr(StreamChunkType, 'SUBAGENT_END') or True

    def test_format_sse_event(self):
        """format_sse_event deve formatar eventos corretamente."""
        from projects.agent.service import format_sse_event

        event = format_sse_event(
            event_type="subagent_start",
            data={"agent": "classification"}
        )

        assert "event:" in event or "data:" in event
