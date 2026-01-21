"""Testes para endpoints multi-agent da API."""
import pytest


class TestMultiAgentAPI:
    """Testes para endpoints de multi-agent."""

    def test_subagents_endpoint_exists(self):
        """GET /agent/subagents deve existir."""
        from app.api.v1.agent.router import router

        routes = [r.path for r in router.routes]
        assert "/subagents" in routes or any("/subagents" in r for r in routes)

    def test_subagent_status_schema(self):
        """SubagentStatusResponse deve existir."""
        from app.api.v1.agent.schemas import SubagentStatusResponse
        assert SubagentStatusResponse is not None

    def test_subagents_list_schema(self):
        """SubagentsListResponse deve existir."""
        from app.api.v1.agent.schemas import SubagentsListResponse
        assert SubagentsListResponse is not None

    def test_chat_detailed_schema(self):
        """ChatDetailedResponse deve existir."""
        from app.api.v1.agent.schemas import ChatDetailedResponse
        assert ChatDetailedResponse is not None
