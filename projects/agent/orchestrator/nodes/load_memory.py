"""No load_memory do Orchestrator.

Carrega memoria de longo prazo (sumarios, entidades, contexto cross-thread)
e injeta no estado antes do processamento da requisicao.
"""
from typing import Optional

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from projects.agent.config import get_agent_settings
from projects.agent.orchestrator.state import OrchestratorState
from shared.core.logging import get_logger

logger = get_logger("orchestrator.load_memory")


def _get_last_user_message(messages) -> Optional[str]:
    """Extrai texto da ultima mensagem do usuario."""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg.content
    return None


async def load_memory(state: OrchestratorState) -> dict:
    """Carrega toda memoria persistente e injeta no estado."""
    settings = get_agent_settings()
    updates: dict = {}

    thread_id = state.get("thread_id")
    if not thread_id:
        return updates

    # Phase 1: Sumario de conversa
    if settings.summarization_enabled:
        try:
            from projects.agent.memory.summarization import SummarizationService

            service = SummarizationService()
            summary = await service.load_summary(thread_id)

            if summary:
                logger.info("Sumario carregado", thread_id=thread_id)
                updates["conversation_summary"] = summary
                updates["messages"] = [
                    SystemMessage(
                        content=(
                            f"## Contexto desta conversa\n\n{summary}\n\n"
                            "Use este contexto para dar respostas mais relevantes."
                        )
                    )
                ]

        except Exception as e:
            logger.warning("Falha ao carregar sumario", error=str(e))

    # Phases 2+3+4: User Memory (entidades + RAG cross-thread)
    if settings.entity_memory_enabled or settings.vector_store_enabled:
        try:
            from projects.agent.memory.user_memory import UserMemoryService

            user_memory = UserMemoryService()
            user_message = _get_last_user_message(state.get("messages", []))

            if user_message:
                context = await user_memory.get_user_context(
                    user_id=state.get("user_id", 0),
                    current_query=user_message,
                    current_thread_id=thread_id,
                )

                formatted = user_memory.format_context_for_injection(context)
                if formatted:
                    if context.get("entities"):
                        updates["user_entities"] = context["entities"]
                    if context.get("related_conversations"):
                        updates["retrieved_context"] = context["related_conversations"]

                    msgs = updates.get("messages", [])
                    msgs.append(SystemMessage(content=formatted))
                    updates["messages"] = msgs

        except Exception as e:
            logger.warning("Falha ao carregar user memory", error=str(e))

    return updates
