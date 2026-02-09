"""No persist_memory do Orchestrator.

Persiste memoria de longo prazo apos a sintese (sumarios, entidades, embeddings).
"""
from projects.agent.config import get_agent_settings
from projects.agent.orchestrator.state import OrchestratorState
from shared.core.logging import get_logger

logger = get_logger("orchestrator.persist_memory")


async def persist_memory(state: OrchestratorState) -> dict:
    """Persiste memoria apos a sintese da resposta.

    Persiste:
    - Sumario de conversas longas (Phase 1)
    - Embeddings da resposta sintetizada (Phase 2)
    """
    settings = get_agent_settings()

    thread_id = state.get("thread_id")
    if not thread_id:
        return {}

    # Phase 1: Sumarizacao
    if settings.summarization_enabled:
        messages = list(state.get("messages", []))
        try:
            from projects.agent.memory.summarization import SummarizationService

            service = SummarizationService()
            summary, _ = await service.get_or_create_summary(
                thread_id=thread_id,
                user_id=state.get("user_id", 0),
                config_id=state.get("config_id", 0),
                messages=messages,
                threshold=settings.summarization_threshold,
                keep_recent=settings.summarization_keep_recent,
            )

            if summary:
                logger.info("Sumario persistido", thread_id=thread_id)

        except Exception as e:
            logger.warning("Falha ao persistir memoria", error=str(e))

    # Phase 2: Salvar embeddings
    if settings.vector_store_enabled:
        try:
            from projects.agent.memory.embeddings import EmbeddingService

            embedding_service = EmbeddingService()
            synthesized = state.get("synthesized_response", "")

            if synthesized:
                await embedding_service.store_embedding(
                    user_id=state.get("user_id", 0),
                    config_id=state.get("config_id", 0),
                    thread_id=thread_id,
                    content=synthesized[:1000],
                    source_type="summary",
                    metadata={
                        "intent": state.get("user_intent"),
                        "agents_used": list(state.get("agent_results", {}).keys()),
                    },
                )
                logger.info("Embedding persistido", thread_id=thread_id)

        except Exception as e:
            logger.warning("Falha ao salvar embedding", error=str(e))

    # Phase 3: Extrair e salvar entidades
    if settings.entity_memory_enabled:
        try:
            from projects.agent.memory.entities import EntityMemoryService

            entity_service = EntityMemoryService()
            synthesized = state.get("synthesized_response", "")

            if synthesized:
                entities = await entity_service.extract_entities(synthesized)
                if entities:
                    await entity_service.save_entities(
                        user_id=state.get("user_id", 0),
                        config_id=state.get("config_id", 0),
                        thread_id=thread_id,
                        entities=entities,
                    )
                    logger.info(
                        "Entidades extraidas e salvas",
                        count=len(entities),
                        thread_id=thread_id,
                    )
        except Exception as e:
            logger.warning("Falha ao extrair entidades", error=str(e))

    return {}
