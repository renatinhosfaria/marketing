"""
Servico de embeddings para busca semantica na memoria do agente.

Gera e armazena embeddings vetoriais de mensagens e sumarios,
permitindo busca por similaridade semantica no historico.
"""
from typing import Optional

from sqlalchemy import text

from projects.agent.config import get_agent_settings
from projects.agent.db.models import MemoryEmbedding
from shared.db.session import async_session_maker
from shared.core.logging import get_logger

logger = get_logger(__name__)


class EmbeddingService:
    """Servico para geracao e busca de embeddings vetoriais."""

    def _get_embeddings_model(self):
        """Retorna o modelo de embeddings configurado."""
        settings = get_agent_settings()

        from langchain_openai import OpenAIEmbeddings

        api_key = settings.openai_api_key
        if not api_key:
            raise ValueError(
                "OpenAI API key necessaria para embeddings. "
                "Configure AGENT_OPENAI_API_KEY."
            )

        return OpenAIEmbeddings(
            model=settings.embedding_model,
            openai_api_key=api_key,
        )

    async def generate_embedding(self, content: str) -> list[float]:
        """Gera embedding vetorial para um texto."""
        model = self._get_embeddings_model()
        return await model.aembed_query(content)

    async def store_embedding(
        self,
        user_id: int,
        config_id: int,
        thread_id: str,
        content: str,
        source_type: str,
        source_id: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        """Gera e armazena embedding no banco."""
        vector = await self.generate_embedding(content)
        vector_str = "[" + ",".join(str(v) for v in vector) + "]"

        async with async_session_maker() as session:
            record = MemoryEmbedding(
                user_id=user_id,
                config_id=config_id,
                thread_id=thread_id,
                source_type=source_type,
                source_id=source_id,
                content=content,
                metadata_=metadata,
            )
            session.add(record)
            await session.flush()

            # Atualizar coluna vector nativa via raw SQL
            await session.execute(
                text(
                    "UPDATE agent_memory_embeddings "
                    "SET embedding_vector = :vec "
                    "WHERE id = :id"
                ),
                {"vec": vector_str, "id": record.id},
            )
            await session.commit()

    async def _search_by_vector(
        self,
        vector: list[float],
        user_id: int,
        limit: int = 5,
        min_similarity: float = 0.7,
        exclude_thread_id: Optional[str] = None,
    ) -> list[dict]:
        """Busca registros similares por vetor."""
        vector_str = "[" + ",".join(str(v) for v in vector) + "]"

        query = """
            SELECT
                content,
                thread_id,
                source_type,
                metadata_,
                1 - (embedding_vector <=> :vec::vector) AS similarity
            FROM agent_memory_embeddings
            WHERE user_id = :user_id
              AND 1 - (embedding_vector <=> :vec::vector) >= :min_sim
        """
        params = {
            "vec": vector_str,
            "user_id": user_id,
            "min_sim": min_similarity,
        }

        if exclude_thread_id:
            query += " AND thread_id != :exclude_tid"
            params["exclude_tid"] = exclude_thread_id

        query += " ORDER BY similarity DESC LIMIT :lim"
        params["lim"] = limit

        async with async_session_maker() as session:
            result = await session.execute(text(query), params)
            rows = result.fetchall()

        return [
            {
                "content": row[0],
                "thread_id": row[1],
                "source_type": row[2],
                "metadata": row[3],
                "similarity": float(row[4]),
            }
            for row in rows
        ]

    async def search_similar(
        self,
        query: str,
        user_id: int,
        limit: int = 5,
        min_similarity: float = 0.7,
        exclude_thread_id: Optional[str] = None,
    ) -> list[dict]:
        """Busca conteudos semanticamente similares a query."""
        vector = await self.generate_embedding(query)
        return await self._search_by_vector(
            vector=vector,
            user_id=user_id,
            limit=limit,
            min_similarity=min_similarity,
            exclude_thread_id=exclude_thread_id,
        )
