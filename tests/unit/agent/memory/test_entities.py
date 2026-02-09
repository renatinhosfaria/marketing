"""Testes para EntityMemoryService."""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from projects.agent.memory.entities import EntityMemoryService, extract_entities_prompt


class TestExtractEntitiesPrompt:
    def test_includes_response_text(self):
        prompt = extract_entities_prompt("Campanha XYZ tem CPL de R$ 25,00")
        assert "XYZ" in prompt
        assert "CPL" in prompt


@pytest.mark.asyncio
class TestEntityMemoryService:
    async def test_extract_entities_from_response(self):
        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = MagicMock(
            content='[{"type":"campaign","key":"Campanha XYZ","value":"CPL R$ 25,00","confidence":0.9}]'
        )

        service = EntityMemoryService()

        with patch("projects.agent.llm.provider.get_llm", return_value=mock_llm):
            entities = await service.extract_entities("Campanha XYZ tem CPL de R$ 25,00")

        assert len(entities) == 1
        assert entities[0]["key"] == "Campanha XYZ"

    async def test_extract_entities_returns_empty_for_invalid_json(self):
        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = MagicMock(
            content="Nao consegui extrair entidades do texto."
        )

        service = EntityMemoryService()

        with patch("projects.agent.llm.provider.get_llm", return_value=mock_llm):
            entities = await service.extract_entities("texto qualquer")

        assert entities == []

    async def test_extract_entities_returns_empty_array(self):
        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = MagicMock(content="[]")

        service = EntityMemoryService()

        with patch("projects.agent.llm.provider.get_llm", return_value=mock_llm):
            entities = await service.extract_entities("texto sem entidades")

        assert entities == []

    async def test_extract_entities_filters_invalid_types(self):
        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = MagicMock(
            content='[{"type":"invalid_type","key":"x","value":"y"},{"type":"campaign","key":"A","value":"B"}]'
        )

        service = EntityMemoryService()

        with patch("projects.agent.llm.provider.get_llm", return_value=mock_llm):
            entities = await service.extract_entities("texto")

        assert len(entities) == 1
        assert entities[0]["type"] == "campaign"

    async def test_load_user_entities(self):
        service = EntityMemoryService()

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [
            MagicMock(
                entity_type="campaign",
                entity_key="Campanha XYZ",
                entity_value="CPL R$ 25,00",
                confidence=0.9,
                mention_count=3,
            )
        ]
        mock_session.execute = AsyncMock(return_value=mock_result)

        with patch(
            "projects.agent.memory.entities.async_session_maker",
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_session),
                __aexit__=AsyncMock(),
            ),
        ):
            entities = await service.load_user_entities(user_id=1, limit=10)

        assert len(entities) == 1
        assert entities[0]["key"] == "Campanha XYZ"
