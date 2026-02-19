"""
Nos do subgraph Gerente de Operacoes.

Fluxo: propose_action -> (interrupt) -> execute_action
O interrupt() pausa para aprovacao humana. O ultimo no produz AgentReport.

Arquitetura: Tools Puras + Interrupt no Node.
"""

import secrets

from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer
from langgraph.store.base import BaseStore
from langgraph.types import interrupt

from projects.agent.subgraphs.operations_manager.state import OperationsSubgraphState
from projects.agent.tools.operations_tools import (
    get_recommendations,
    prepare_budget_change,
    prepare_status_change,
    execute_budget_change,
    execute_status_change,
    build_approval_token,
)
from projects.agent.llm.provider import get_model
from projects.agent.graph.routing import ActionDecision
from projects.agent.prompts.operations_manager import SYSTEM_PROMPT
from projects.agent.observability.metrics import (
    agent_interrupts_total,
    approval_token_failures,
)

import structlog

logger = structlog.get_logger()


def _approved_budget_override(approved_overrides: dict) -> float | None:
    """Extrai override de budget aprovado (canonico ou legado)."""
    if not isinstance(approved_overrides, dict):
        return None

    raw_value = approved_overrides.get("new_budget_override")
    if raw_value is None:
        raw_value = approved_overrides.get("override_value")

    if raw_value in (None, ""):
        return None

    try:
        return float(raw_value)
    except (TypeError, ValueError):
        logger.warning(
            "Override de budget invalido no resume_payload",
            raw_value=raw_value,
        )
        return None


async def propose_action_node(
    state: OperationsSubgraphState,
    config: RunnableConfig,
    *,
    store: BaseStore,
):
    """Analisa a situacao e propoe uma acao ao usuario.

    O LLM decide se uma acao e necessaria. Se sim, usa tools prepare_*
    para validar e construir a proposta. Em seguida, interrompe para
    aprovacao humana via interrupt().
    """
    writer = get_stream_writer()
    writer({"agent": "operations_manager", "status": "running", "progress": 0})

    user_question = state["messages"][-1].content if state.get("messages") else ""
    scope = state.get("scope") or {}

    # Buscar recomendacoes do sistema ML
    writer({"agent": "operations_manager", "status": "fetching_recommendations", "progress": 20})

    entity_type = scope.get("entity_type", "campaign")
    recs_result = await get_recommendations.ainvoke(
        {"entity_type": entity_type},
        config=config,
    )

    recommendations = None
    if isinstance(recs_result, dict) and recs_result.get("ok"):
        recommendations = recs_result.get("data")

    # LLM analisa com structured output e decide se propoe acao
    writer({"agent": "operations_manager", "status": "analyzing", "progress": 40})

    model = get_model("operations", config).with_structured_output(ActionDecision)
    prompt = _build_operations_prompt(user_question, recommendations, scope)
    decision: ActionDecision = await model.ainvoke(prompt)

    if not decision.action_needed:
        # LLM decidiu que nenhuma acao e necessaria
        return {
            "proposed_action": None,
            "approval_status": None,
        }

    # LLM decidiu que acao e necessaria — preparar proposta via tool
    writer({"agent": "operations_manager", "status": "preparing_action", "progress": 50})

    proposal = None
    if decision.action_type == "budget_change" and decision.campaign_id and decision.new_value:
        try:
            budget_value = float(decision.new_value)
        except (ValueError, TypeError):
            logger.warning(
                "LLM gerou new_value nao numerico para budget_change",
                new_value=decision.new_value,
            )
            return {"proposed_action": None, "approval_status": None}
        proposal = await prepare_budget_change.ainvoke(
            {
                "campaign_id": decision.campaign_id,
                "new_daily_budget": budget_value,
                "reason": decision.reason,
            },
            config=config,
        )
    elif decision.action_type == "status_change" and decision.campaign_id and decision.new_value:
        proposal = await prepare_status_change.ainvoke(
            {
                "campaign_id": decision.campaign_id,
                "new_status": decision.new_value,
                "reason": decision.reason,
            },
            config=config,
        )

    if not proposal or not isinstance(proposal, dict) or not proposal.get("ok"):
        # Preparacao falhou (validacao, ownership, etc.)
        error_msg = "Preparacao falhou"
        if proposal and isinstance(proposal, dict) and proposal.get("error"):
            error_msg = proposal["error"].get("message", error_msg)
        logger.warning("Proposta rejeitada na preparacao", error=error_msg)
        return {
            "proposed_action": proposal,
            "approval_status": None,
        }

    # Proposta valida — interromper para aprovacao humana
    writer({"agent": "operations_manager", "status": "awaiting_approval", "progress": 60})

    cfg = config.get("configurable", {})
    thread_id = cfg.get("thread_id", "")
    idempotency_key = proposal.get("data", {}).get("idempotency_key", "")
    approval_token = build_approval_token(thread_id, idempotency_key)

    action_type = proposal["data"].get("action_type", "unknown")
    agent_interrupts_total.labels(
        interrupt_type=action_type, resolution="triggered",
    ).inc()

    approval = interrupt({
        "type": action_type,
        "approval_token": approval_token,
        "details": proposal["data"],
    })

    # Anti-forgery
    if not secrets.compare_digest(
        approval.get("approval_token", ""), approval_token,
    ):
        approval_token_failures.labels(action_type=action_type).inc()
        agent_interrupts_total.labels(
            interrupt_type=action_type, resolution="token_invalid",
        ).inc()
        return {
            "approval_status": "rejected",
            "execution_result": {"error": "Token invalido."},
        }

    if approval.get("approved"):
        agent_interrupts_total.labels(
            interrupt_type=action_type, resolution="approved",
        ).inc()
        return {
            "approval_status": "approved",
            "proposed_action": {
                **proposal,
                "approved_overrides": approval,
            },
        }

    agent_interrupts_total.labels(
        interrupt_type=action_type, resolution="rejected",
    ).inc()
    return {"approval_status": "rejected"}


async def execute_action_node(
    state: OperationsSubgraphState,
    config: RunnableConfig,
    *,
    store: BaseStore,
):
    """Executa acao aprovada ou gera relatorio se cancelada.

    Chama execute_budget_change / execute_status_change com dados aprovados.
    """
    writer = get_stream_writer()
    writer({"agent": "operations_manager", "status": "executing", "progress": 80})

    if state.get("approval_status") != "approved":
        # Acao cancelada ou nenhuma acao proposta — gerar relatorio
        user_question = state["messages"][-1].content if state.get("messages") else ""
        model = get_model("operations", config)

        prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"Pergunta do usuario: {user_question}\n\n"
            "Nenhuma acao foi executada. Forneca um resumo das recomendacoes "
            "disponiveis e proximos passos. Responda em portugues (Brasil)."
        )
        response = await model.ainvoke(prompt)

        summary = response.content
        if state.get("approval_status") == "rejected":
            summary = "Acao cancelada pelo usuario. " + summary

        writer({"agent": "operations_manager", "status": "completed", "progress": 100})

        return {"agent_reports": [{
            "agent_id": "operations_manager",
            "status": "completed",
            "summary": summary,
            "data": {"action_executed": False},
            "confidence": 1.0,
        }]}

    # Acao aprovada — executar via tool
    proposal = state.get("proposed_action") or {}
    data = proposal.get("data", {})
    approved_overrides = proposal.get("approved_overrides") or {}
    action_type = data.get("action_type")
    campaign_id = data.get("campaign_id")
    idempotency_key = data.get("idempotency_key")

    if not campaign_id or not idempotency_key or not action_type:
        logger.error(
            "Dados da proposta aprovada incompletos",
            action_type=action_type,
            campaign_id=campaign_id,
        )
        writer({"agent": "operations_manager", "status": "completed", "progress": 100})
        return {
            "execution_result": None,
            "agent_reports": [{
                "agent_id": "operations_manager",
                "status": "completed",
                "summary": "Falha na execucao: dados da proposta incompletos.",
                "data": {"action_executed": False, "error": "Dados da proposta incompletos"},
                "confidence": 1.0,
            }],
        }

    execution_result = None
    if action_type == "budget_change":
        new_value = _approved_budget_override(approved_overrides)
        if new_value is None:
            new_value = data.get("new_value")
        if new_value is None:
            logger.error("new_value ausente para budget_change")
            execution_result = {"ok": False, "data": None, "error": {"code": "INVALID_DATA", "message": "new_value ausente na proposta."}}
        else:
            execution_result = await execute_budget_change.ainvoke(
                {
                    "campaign_id": campaign_id,
                    "new_daily_budget": float(new_value),
                    "idempotency_key": idempotency_key,
                },
                config=config,
            )
    elif action_type == "status_change":
        new_status = data.get("new_status")
        if not new_status:
            logger.error("new_status ausente para status_change")
            execution_result = {"ok": False, "data": None, "error": {"code": "INVALID_DATA", "message": "new_status ausente na proposta."}}
        else:
            execution_result = await execute_status_change.ainvoke(
                {
                    "campaign_id": campaign_id,
                    "new_status": new_status,
                    "idempotency_key": idempotency_key,
                },
                config=config,
            )

    if not execution_result:
        execution_result = {"ok": False, "data": None, "error": {"code": "UNKNOWN", "message": "Tipo de acao desconhecido."}}

    writer({"agent": "operations_manager", "status": "completed", "progress": 100})

    if execution_result.get("ok"):
        message = execution_result.get("data", {}).get("message", "OK")
        return {
            "execution_result": execution_result,
            "agent_reports": [{
                "agent_id": "operations_manager",
                "status": "completed",
                "summary": f"Acao executada: {message}",
                "data": {"action_executed": True, "result": execution_result},
                "confidence": 1.0,
            }],
        }

    # Execucao falhou
    error_msg = execution_result.get("error", {}).get("message", "Erro desconhecido")
    return {
        "execution_result": execution_result,
        "agent_reports": [{
            "agent_id": "operations_manager",
            "status": "completed",
            "summary": f"Falha na execucao: {error_msg}",
            "data": {"action_executed": False, "error": error_msg},
            "confidence": 1.0,
        }],
    }


def _build_operations_prompt(
    user_question: str,
    recommendations: dict,
    scope: dict,
) -> str:
    """Constroi prompt de analise operacional."""
    import json

    parts = [SYSTEM_PROMPT]
    parts.append(f"Pergunta do usuario: {user_question}")

    if recommendations:
        recs_json = json.dumps(recommendations, indent=2, ensure_ascii=False)
        parts.append(f"Recomendacoes do sistema ML:\n{recs_json}")

    if scope:
        scope_json = json.dumps(scope, indent=2, ensure_ascii=False)
        parts.append(f"Escopo: {scope_json}")

    parts.append(
        "Analise as recomendacoes e determine se alguma acao e necessaria. "
        "Se sim, descreva a acao proposta com detalhes. "
        "Responda em portugues (Brasil)."
    )
    return "\n\n".join(parts)
