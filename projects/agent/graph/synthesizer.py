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
    """Sintetiza reports dos agentes em resposta final.

    Garante sintese parcial mesmo quando:
    - Alguns agentes falharam (usa reports bem-sucedidos)
    - O LLM de sintese falha (retorna resumo textual dos reports sem LLM)
    """
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

    try:
        model = get_model("synthesizer", config)
        response = await model.ainvoke([
            SystemMessage(content=prompt),
            HumanMessage(content=user_question),
        ])
        synthesis_text = response.content
    except Exception as e:
        # LLM indisponivel: montar sintese textual a partir dos reports sem LLM
        logger.warning(
            "synthesizer.llm_failed",
            error=str(e),
            successful_agents=len(successful),
            failed_agents=len(failed),
        )
        synthesis_text = _build_plain_synthesis(successful, failed)
        response = AIMessage(content=synthesis_text)

    # Gerar titulo na primeira interacao (best-effort — nao bloqueia resposta)
    await _maybe_generate_title(state, config, store, user_question)

    return {"messages": [response], "synthesis": synthesis_text}


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


def _build_plain_synthesis(successful: list, failed: list) -> str:
    """Sintese textual sem LLM: usado como fallback quando o modelo nao responde.

    Formata os summaries dos agentes bem-sucedidos como lista estruturada.
    Inclui aviso sobre agentes com falha.
    """
    parts = []

    if successful:
        parts.append("**Analise parcial dos agentes disponíveis:**\n")
        for r in successful:
            agent_label = r.get("agent_id", "agente").replace("_", " ").title()
            summary = (r.get("summary") or "Sem dados disponiveis.").strip()
            parts.append(f"**{agent_label}:** {summary}\n")

    if failed:
        agents_str = ", ".join(r.get("agent_id", "?") for r in failed)
        parts.append(
            f"\n*Nota: {len(failed)} agente(s) nao responderam nesta execucao "
            f"({agents_str}). Os dados acima refletem apenas as analises disponiveis.*"
        )

    if not parts:
        return (
            "Nao foi possivel obter analises neste momento. "
            "Por favor, tente novamente em instantes."
        )

    return "\n".join(parts)
