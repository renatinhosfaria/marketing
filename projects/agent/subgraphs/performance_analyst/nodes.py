"""
Nos do subgraph Analista de Performance & Impacto.

Fluxo: analyze_metrics -> compare_periods -> generate_report
O ultimo no (generate_report) produz AgentReport para o synthesizer.
"""

from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer

from projects.agent.subgraphs.performance_analyst.state import PerformanceSubgraphState
from projects.agent.tools.performance_tools import (
    get_insights_summary,
    compare_periods,
)
from projects.agent.llm.provider import get_model
from projects.agent.prompts.performance_analyst import SYSTEM_PROMPT

import structlog

logger = structlog.get_logger()


async def analyze_metrics_node(
    state: PerformanceSubgraphState,
    config: RunnableConfig,
):
    """Busca metricas agregadas da conta."""
    writer = get_stream_writer()
    writer({"agent": "performance_analyst", "status": "running", "progress": 0})

    writer({"agent": "performance_analyst", "status": "fetching_metrics", "progress": 10})

    summary_result = await get_insights_summary.ainvoke({}, config=config)

    metrics_data = None
    metrics_error = None
    if isinstance(summary_result, dict) and summary_result.get("ok"):
        metrics_data = summary_result.get("data")
    elif isinstance(summary_result, dict) and summary_result.get("error"):
        err = summary_result["error"]
        metrics_error = f"[{err.get('code', 'UNKNOWN')}] {err.get('message', 'Erro desconhecido')}"

    writer({"agent": "performance_analyst", "status": "metrics_loaded", "progress": 30})

    return {"metrics_data": metrics_data, "metrics_error": metrics_error}


async def compare_periods_node(
    state: PerformanceSubgraphState,
    config: RunnableConfig,
):
    """Compara periodos (semana atual vs anterior) para identificar tendencias."""
    writer = get_stream_writer()
    writer({"agent": "performance_analyst", "status": "comparing_periods", "progress": 50})

    scope = state.get("scope") or {}
    entity_ids = scope.get("entity_ids") or []

    comparison = None
    if entity_ids:
        # Comparar ultima semana vs semana anterior para a primeira entidade
        from datetime import datetime, timedelta
        today = datetime.now().strftime("%Y-%m-%d")
        week_ago = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        two_weeks_ago = (datetime.now() - timedelta(days=14)).strftime("%Y-%m-%d")

        comp_result = await compare_periods.ainvoke(
            {
                "entity_id": entity_ids[0],
                "period_a_start": two_weeks_ago,
                "period_a_end": week_ago,
                "period_b_start": week_ago,
                "period_b_end": today,
            },
            config=config,
        )

        if isinstance(comp_result, dict) and comp_result.get("ok"):
            comparison = comp_result.get("data")

    writer({"agent": "performance_analyst", "status": "comparison_done", "progress": 70})

    return {"comparison": comparison}


async def generate_report_node(
    state: PerformanceSubgraphState,
    config: RunnableConfig,
):
    """Gera relatorio de performance e produz AgentReport."""
    writer = get_stream_writer()
    writer({"agent": "performance_analyst", "status": "generating_report", "progress": 80})

    metrics_data = state.get("metrics_data") or {}
    metrics_error = state.get("metrics_error")
    comparison = state.get("comparison")
    user_question = state["messages"][-1].content if state.get("messages") else ""

    model = get_model("analyst", config)
    prompt = _build_report_prompt(metrics_data, comparison, user_question, metrics_error)
    response = await model.ainvoke(prompt)

    writer({"agent": "performance_analyst", "status": "completed", "progress": 100})

    return {
        "report": response.content,
        "agent_reports": [{
            "agent_id": "performance_analyst",
            "status": "completed",
            "summary": response.content,
            "data": {
                "metrics_summary": metrics_data,
                "comparison": comparison,
            },
            "confidence": 0.85,
        }],
    }


def _build_report_prompt(
    metrics: dict,
    comparison: dict,
    user_question: str,
    metrics_error: str = None,
) -> str:
    """Constroi prompt de analise de performance."""
    parts = [SYSTEM_PROMPT]
    parts.append(f"\n\nPergunta do usuario: {user_question}")

    if metrics_error:
        parts.append(
            f"\n\n**ERRO ao buscar metricas**: {metrics_error}. "
            "Nao foi possivel obter o resumo de KPIs. Informe ao usuario."
        )
    elif metrics:
        parts.append(f"\n\nResumo de metricas (ultimos 7 dias): {metrics}")
    else:
        parts.append(
            "\n\nBusca de metricas executada com sucesso, mas nao ha dados "
            "de insights disponiveis nos ultimos 7 dias para esta conta."
        )

    if comparison:
        parts.append(f"\nComparacao entre periodos: {comparison}")

    parts.append(
        "\n\nAnalise a performance das campanhas, identifique tendencias, "
        "problemas e oportunidades. Se houve erros na obtencao dos dados, "
        "informe ao usuario e sugira tentar novamente. "
        "Responda em portugues (Brasil)."
    )
    return "\n".join(parts)
