"""
No Synthesizer: fan-in dos reports dos agentes + resposta final.

Recebe todos os AgentReports acumulados via operator.add e sintetiza
uma resposta coerente em portugues para o usuario.
"""

from langchain_core.runnables import RunnableConfig
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langgraph.store.base import BaseStore

import structlog

from projects.agent.graph.state import SupervisorState
from projects.agent.llm.provider import get_model
from projects.agent.memory.namespaces import StoreNamespace

logger = structlog.get_logger()


async def synthesizer_node(
    state: SupervisorState,
    config: RunnableConfig,
    *,
    store: BaseStore,
):
    """Sintetiza reports dos agentes em resposta final."""
    reports = state.get("agent_reports", [])

    # Se nenhum agente respondeu, resposta de fallback
    if not reports:
        return {"messages": [AIMessage(
            content="Desculpe, nao consegui analisar seus dados neste momento. "
                    "Por favor, tente novamente."
        )]}

    # Se alguns agentes falharam, sintetiza com os que responderam
    successful = [r for r in reports if r["status"] == "completed"]
    failed = [r for r in reports if r["status"] == "error"]

    user_question = state["messages"][-1].content if state["messages"] else ""
    prompt = _build_synthesis_prompt(successful, failed, user_question)

    model = get_model("synthesizer", config)
    response = await model.ainvoke([
        SystemMessage(content=prompt),
        HumanMessage(content=user_question),
    ])

    # Gerar titulo na primeira interacao
    await _maybe_generate_title(state, config, store, user_question)

    return {"messages": [response], "synthesis": response.content}


async def _maybe_generate_title(
    state: SupervisorState,
    config: RunnableConfig,
    store: BaseStore,
    user_question: str,
):
    """Gera titulo da conversa se for a primeira interacao."""
    human_messages = [
        m for m in state["messages"]
        if isinstance(m, HumanMessage)
    ]
    if len(human_messages) != 1:
        return  # Nao e a primeira interacao

    user_ctx = state.get("user_context", {})
    user_id = user_ctx.get("user_id", "")
    account_id = user_ctx.get("account_id", "")
    if not user_id or not account_id:
        return

    full_thread_id = config.get("configurable", {}).get("thread_id", "")
    prefix = f"{user_id}:{account_id}:"
    frontend_thread_id = full_thread_id.removeprefix(prefix) if full_thread_id.startswith(prefix) else full_thread_id

    try:
        model = get_model("title_generator", config)
        title_response = await model.ainvoke([
            SystemMessage(
                content="Gere um titulo curto (5-8 palavras) em portugues para esta conversa. "
                        "Retorne APENAS o titulo, sem aspas nem pontuacao extra."
            ),
            HumanMessage(content=user_question),
        ])
        title = title_response.content.strip()[:80]

        ns = StoreNamespace.conversation_titles(user_id, account_id)
        await store.aput(ns, frontend_thread_id, {"title": title})
        logger.info("title.generated", thread_id=frontend_thread_id, title=title)
    except Exception as e:
        logger.warning("title.generation_failed", error=str(e))


def _build_synthesis_prompt(
    successful: list,
    failed: list,
    user_question: str,
) -> str:
    """Constroi o prompt de sintese a partir dos reports dos agentes."""
    parts = [
        f"Voce e o sintetizador do FamaChat AI Agent. "
        f"O usuario perguntou: '{user_question}'"
    ]
    parts.append(
        "\nResuma os resultados dos agentes abaixo em uma resposta clara e util "
        "em portugues:"
    )

    for r in successful:
        parts.append(
            f"\n--- {r['agent_id']} (confianca: {r['confidence']:.0%}) ---\n"
            f"{r['summary']}"
        )

    if failed:
        parts.append(
            "\nAgentes com falha (mencione brevemente que houve problemas):"
        )
        for r in failed:
            parts.append(f"- {r['agent_id']}: {r['summary']}")

    parts.append(
        "\nResponda em portugues (Brasil), de forma clara e acionavel. "
        "Use formatacao markdown quando apropriado."
    )

    return "\n".join(parts)
