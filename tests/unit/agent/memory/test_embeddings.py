"""Testes para EmbeddingService."""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from projects.agent.memory.embeddings import EmbeddingService


@pytest.mark.asyncio
class TestEmbeddingService:
    async def test_generate_embedding(self):
        mock_embeddings = AsyncMock()
        mock_embeddings.aembed_query.return_value = [0.1] * 1536

        service = EmbeddingService()
        with patch.object(service, "_get_embeddings_model", return_value=mock_embeddings):
            vector = await service.generate_embedding("Como estao minhas campanhas?")

        assert len(vector) == 1536
        assert vector[0] == 0.1

    async def test_search_similar_returns_results(self):
        service = EmbeddingService()

        mock_results = [
            {"content": "Conversa sobre CPL", "similarity": 0.92, "thread_id": "t-1"},
            {"content": "Analise de performance", "similarity": 0.85, "thread_id": "t-2"},
        ]

        with patch.object(service, "_search_by_vector", return_value=mock_results):
            with patch.object(service, "generate_embedding", return_value=[0.1] * 1536):
                results = await service.search_similar(
                    query="CPL das campanhas",
                    user_id=1,
                    limit=5,
                )

        assert len(results) == 2
        assert results[0]["similarity"] > results[1]["similarity"]
