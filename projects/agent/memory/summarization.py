"""
Servico de sumarizacao para memoria de longo prazo.

Quando uma conversa excede um threshold de mensagens, as mais antigas
sao resumidas em um sumario compacto pelo LLM. O sumario e persistido
no banco e injetado como contexto nas proximas requisicoes.
"""
from typing import Optional, Tuple, List

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from sqlalchemy import select

from projects.agent.db.models import ConversationSummary
from shared.db.session import async_session_maker
from shared.core.logging import get_logger

logger = get_logger(__name__)


def should_summarize(messages: list[BaseMessage], threshold: int = 20) -> bool:
    """Verifica se as mensagens devem ser resumidas.

    Conta apenas mensagens nao-system (HumanMessage, AIMessage).
    """
    non_system = [m for m in messages if not isinstance(m, SystemMessage)]
    return len(non_system) > threshold


def build_summarization_prompt(messages: list[BaseMessage]) -> str:
    """Constroi prompt para o LLM resumir mensagens."""
    lines = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            lines.append(f"Usuario: {msg.content}")
        elif isinstance(msg, AIMessage):
            lines.append(f"Assistente: {msg.content}")

    conversation = "\n".join(lines)

    return (
        "Voce e um assistente de sumarizacao. Resuma a conversa abaixo de forma "
        "concisa, preservando:\n"
        "- Perguntas e topicos discutidos pelo usuario\n"
        "- Campanhas, metricas e valores mencionados\n"
        "- Decisoes tomadas ou recomendacoes aceitas\n"
        "- Qualquer preferencia expressa pelo usuario\n\n"
        "Mantenha o resumo em no maximo 500 tokens.\n\n"
        f"--- CONVERSA ---\n{conversation}\n--- FIM ---\n\n"
        "Resumo:"
    )


class SummarizationService:
    """Servico para sumarizacao de conversas longas."""

    async def summarize_messages(self, messages: list[BaseMessage]) -> str:
        """Resume uma lista de mensagens usando o LLM."""
        from projects.agent.llm.provider import get_llm
        from projects.agent.config import get_agent_settings

        settings = get_agent_settings()
        prompt = build_summarization_prompt(messages)
        llm = get_llm(temperature=0.1, max_tokens=settings.summarization_max_tokens)
        response = await llm.ainvoke(prompt)
        return response.content.strip()

    async def load_summary(self, thread_id: str) -> Optional[str]:
        """Carrega sumario existente do banco."""
        async with async_session_maker() as session:
            result = await session.execute(
                select(ConversationSummary).where(
                    ConversationSummary.thread_id == thread_id
                )
            )
            summary = result.scalar_one_or_none()
            return summary.summary_text if summary else None

    async def save_summary(
        self,
        thread_id: str,
        user_id: int,
        config_id: int,
        summary_text: str,
        messages_summarized: int,
        last_message_index: int,
    ) -> None:
        """Salva ou atualiza sumario no banco."""
        async with async_session_maker() as session:
            result = await session.execute(
                select(ConversationSummary).where(
                    ConversationSummary.thread_id == thread_id
                )
            )
            existing = result.scalar_one_or_none()

            if existing:
                existing.summary_text = summary_text
                existing.messages_summarized = messages_summarized
                existing.last_message_index = last_message_index
                existing.token_count = len(summary_text) // 4
            else:
                summary = ConversationSummary(
                    thread_id=thread_id,
                    user_id=user_id,
                    config_id=config_id,
                    summary_text=summary_text,
                    token_count=len(summary_text) // 4,
                    messages_summarized=messages_summarized,
                    last_message_index=last_message_index,
                )
                session.add(summary)

            await session.commit()

    async def get_or_create_summary(
        self,
        thread_id: str,
        user_id: int,
        config_id: int,
        messages: list[BaseMessage],
        threshold: int = 20,
        keep_recent: int = 10,
    ) -> Tuple[Optional[str], List[BaseMessage]]:
        """Obtem sumario existente ou cria um novo se necessario.

        Retorna o sumario e a lista de mensagens recentes (ja truncada).
        """
        if not should_summarize(messages, threshold):
            return None, messages

        # Separar system messages
        system_msgs = [m for m in messages if isinstance(m, SystemMessage)]
        non_system = [m for m in messages if not isinstance(m, SystemMessage)]

        # Mensagens antigas (para resumir) e recentes (manter intactas)
        old_messages = non_system[:-keep_recent]
        recent_messages = non_system[-keep_recent:]

        if not old_messages:
            return None, messages

        logger.info(
            "Sumarizando mensagens antigas",
            thread_id=thread_id,
            old_count=len(old_messages),
            recent_count=len(recent_messages),
        )

        # Gerar sumario
        summary_text = await self.summarize_messages(old_messages)

        # Persistir
        await self.save_summary(
            thread_id=thread_id,
            user_id=user_id,
            config_id=config_id,
            summary_text=summary_text,
            messages_summarized=len(old_messages),
            last_message_index=len(non_system) - keep_recent,
        )

        return summary_text, system_msgs + recent_messages
