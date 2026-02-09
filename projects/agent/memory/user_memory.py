"""
Servico unificado de User Memory.

Combina Entity Memory e Vector Store para fornecer contexto
cross-thread personalizado por usuario.
"""
from typing import Optional

from projects.agent.memory.entities import EntityMemoryService
from projects.agent.memory.embeddings import EmbeddingService
from projects.agent.config import get_agent_settings
from shared.core.logging import get_logger

logger = get_logger(__name__)


class UserMemoryService:
    """Servico que unifica todas as camadas de memoria do usuario."""

    def __init__(self):
        self._entity_service = EntityMemoryService()
        self._embedding_service = EmbeddingService()

    async def get_user_context(
        self,
        user_id: int,
        current_query: str,
        current_thread_id: str,
    ) -> dict:
        """Obtem contexto completo do usuario para enriquecer a conversa.

        Combina:
        - Entidades persistidas (preferencias, campanhas, metricas)
        - Conversas anteriores semanticamente relevantes (RAG cross-thread)

        Args:
            user_id: ID do usuario.
            current_query: Pergunta atual do usuario.
            current_thread_id: Thread atual (excluida da busca).

        Returns:
            Dict com entities e related_conversations.
        """
        settings = get_agent_settings()
        result = {
            "entities": [],
            "related_conversations": [],
        }

        # 1. Carregar entidades do usuario
        if settings.entity_memory_enabled:
            try:
                entities = await self._entity_service.load_user_entities(
                    user_id=user_id,
                    limit=settings.entity_max_per_user,
                )
                result["entities"] = entities
            except Exception as e:
                logger.warning("Falha ao carregar entidades", error=str(e))

        # 2. Buscar conversas relevantes (cross-thread)
        if settings.vector_store_enabled and settings.cross_thread_enabled:
            try:
                related = await self._embedding_service.search_similar(
                    query=current_query,
                    user_id=user_id,
                    limit=settings.rag_top_k,
                    min_similarity=settings.rag_min_similarity,
                    exclude_thread_id=current_thread_id,
                )
                result["related_conversations"] = related
            except Exception as e:
                logger.warning("Falha ao buscar conversas", error=str(e))

        return result

    def format_context_for_injection(self, context: dict) -> Optional[str]:
        """Formata o contexto do usuario para injecao como SystemMessage.

        Args:
            context: Dict retornado por get_user_context.

        Returns:
            Texto formatado ou None se nao houver contexto.
        """
        parts = []

        entities = context.get("entities", [])
        if entities:
            entity_lines = []
            for e in entities[:10]:
                entity_lines.append(f"- [{e['type']}] {e['key']}: {e['value']}")
            parts.append(
                "### Conhecimento sobre o usuario\n" + "\n".join(entity_lines)
            )

        conversations = context.get("related_conversations", [])
        if conversations:
            conv_lines = []
            for c in conversations[:3]:
                conv_lines.append(
                    f"- (similaridade {c['similarity']:.0%}) {c['content'][:200]}"
                )
            parts.append(
                "### Conversas anteriores relevantes\n" + "\n".join(conv_lines)
            )

        if not parts:
            return None

        return "## Memoria do usuario\n\n" + "\n\n".join(parts)
