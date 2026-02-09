"""Testes para o modelo MemoryEmbedding."""
import pytest
from projects.agent.db.models import MemoryEmbedding


class TestMemoryEmbeddingModel:
    def test_create_embedding(self):
        emb = MemoryEmbedding(
            user_id=1,
            config_id=1,
            thread_id="t-1",
            source_type="message",
            content="Como estao minhas campanhas?",
        )
        assert emb.source_type == "message"
        assert emb.thread_id == "t-1"

    def test_tablename(self):
        assert MemoryEmbedding.__tablename__ == "agent_memory_embeddings"
