"""
No Supervisor: classificacao de intencao + dispatch via Send() fan-out.

O Supervisor usa Haiku 3.5 com Structured Output para classificar a intencao
do usuario e despachar agentes em paralelo. Retorna Command(goto=[Send()])
que o LangGraph 1.x executa no mesmo super-step.
"""

from typing import List

from langgraph.types import Command, Send
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

from projects.agent.graph.state import SupervisorState, AgentReport
from projects.agent.graph.routing import RoutingDecision
from projects.agent.llm.provider import get_model
from projects.agent.prompts.supervisor import SYSTEM_PROMPT
from projects.agent.config import agent_settings
from projects.agent.observability.metrics import agent_dispatches_total

import structlog

logger = structlog.get_logger()

MAX_HISTORY_MESSAGES = agent_settings.supervisor_max_history_messages


def _build_context_messages(
    messages: list, system_prompt: str
) -> List[SystemMessage | HumanMessage | AIMessage]:
    """Monta lista de mensagens para o LLM com sliding window.

    Filtra apenas HumanMessage e AIMessage com conteudo (ignora ToolMessage e vazias).
    Se > MAX_HISTORY_MESSAGES, pega as ultimas MAX_HISTORY_MESSAGES.
    """
    filtered = [
        m for m in messages
        if isinstance(m, (HumanMessage, AIMessage))
        and getattr(m, "content", None)
    ]
    if len(filtered) > MAX_HISTORY_MESSAGES:
        filtered = filtered[-MAX_HISTORY_MESSAGES:]
    return [SystemMessage(content=system_prompt)] + filtered


def _build_reports_context(agent_reports: List[AgentReport]) -> str:
    """Resume agent_reports previos em texto compacto.

    Retorna string vazia se nao ha reports (sem poluir prompt).
    """
    if not agent_reports:
        return ""
    lines = ["## Analises ja realizadas\n"]
    for report in agent_reports:
        summary = (report.get("summary") or "")[:200]
        status = report.get("status", "?")
        agent_id = report.get("agent_id", "unknown")
        lines.append(f"- {agent_id} [{status.upper()}]: {summary}")
    return "\n".join(lines)


async def supervisor_node(state: SupervisorState, config: RunnableConfig):
    """Classifica intencao e despacha agentes em paralelo via Send().

    IMPORTANTE: Este no e async e usa ainvoke() para nao bloquear o event-loop.
    Mesmo Haiku sendo rapido, a chamada e I/O de rede e DEVE ser non-blocking.

    Retorna Command(goto=[Send()]) que o LangGraph 1.x executa em paralelo.
    Nao precisa de add_conditional_edges — o Send() e o mecanismo de dispatch.
    """
    # RemainingSteps e managed pelo LangGraph; avalia como int em comparacoes.
    # Em testes unitarios (node chamado fora do grafo), usar int literal no mock.
    if state["remaining_steps"] <= 3:
        return {
            "agent_reports": [{
                "agent_id": "supervisor",
                "status": "completed",
                "summary": "Limite de processamento atingido.",
                "data": None,
                "confidence": 0.5,
            }],
            "messages": [AIMessage(
                content="Atingi o limite de processamento desta execucao. "
                        "Respondi com os dados ja disponiveis."
            )],
        }

    messages = state["messages"]
    last_message = messages[-1]
    thread_id = config["configurable"]["thread_id"]

    logger.info(
        "supervisor.classify_start",
        thread_id=thread_id,
        message_preview=last_message.content[:100],
    )

    model = get_model("supervisor", config).with_structured_output(RoutingDecision)

    # Monta historico filtrado com sliding window para contexto multi-turn
    context_messages = _build_context_messages(state["messages"], SYSTEM_PROMPT)
    reports_ctx = _build_reports_context(state.get("agent_reports", []))
    if reports_ctx:
        context_messages.insert(1, SystemMessage(content=reports_ctx))

    decision = await model.ainvoke(context_messages)

    if not decision.selected_agents:
        return {"messages": [AIMessage(
            content="Ola! Sou o assistente de marketing da FamaChat. "
                    "Posso analisar a saude das suas campanhas, comparar periodos, "
                    "detectar anomalias, gerar previsoes e muito mais. "
                    "Como posso ajudar com seus anuncios hoje?"
        )]}

    logger.info(
        "supervisor.dispatch",
        thread_id=thread_id,
        selected_agents=decision.selected_agents,
        urgency=decision.urgency,
    )

    # Send() despacha cada agente com messages + scope como input.
    # O arg deve ser compativel com AgentInput (input_schema de todos os subgraphs).
    scope_dict = decision.scope.model_dump()

    # Metricas: contar despachos por agente
    for agent_id in decision.selected_agents:
        agent_dispatches_total.labels(agent_id=agent_id).inc()

    # Contexto multi-turn: enviar ultimas N mensagens (Human+AI) aos agentes.
    # Apenas a pergunta atual nao basta para conversas longas.
    n_ctx = agent_settings.agent_context_messages
    recent = [
        m for m in messages
        if isinstance(m, (HumanMessage, AIMessage)) and getattr(m, "content", None)
    ][-n_ctx:]
    # Garantir que a ultima mensagem do usuario esta sempre incluida
    if recent and recent[-1] is not last_message:
        recent.append(last_message)
    agent_messages = recent or [last_message]

    return Command(goto=[
        Send(node=agent_id, arg={
            "messages": agent_messages,
            "scope": scope_dict,
        })
        for agent_id in decision.selected_agents
    ])
