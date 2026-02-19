"""
No simples do Especialista em Audiencias (sem subgraph).

Funcao unica que analisa audiencias e retorna AgentReport.
"""

from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer

from projects.agent.tools.audience_tools import (
    get_adset_audiences,
    detect_audience_saturation,
    get_audience_performance,
)
from projects.agent.llm.provider import get_model
from projects.agent.prompts.audience_specialist import SYSTEM_PROMPT

import structlog

logger = structlog.get_logger()


async def audience_node(state: dict, config: RunnableConfig):
    """Analisa audiencias e retorna AgentReport.

    No simples (sem subgraph) pois a analise de audiencias e direta:
    busca dados, analisa saturacao, gera relatorio.
    """
    writer = get_stream_writer()
    writer({"agent": "audience_specialist", "status": "running", "progress": 0})

    scope = state.get("scope") or {}
    campaign_id = None
    entity_ids = scope.get("entity_ids")
    if entity_ids and scope.get("entity_type") == "campaign":
        campaign_id = entity_ids[0]

    # Buscar audiencias
    writer({"agent": "audience_specialist", "status": "fetching_audiences", "progress": 10})
    audiences_result = await get_adset_audiences.ainvoke(
        {"campaign_id": campaign_id},
        config=config,
    )

    audiences = []
    if isinstance(audiences_result, dict) and audiences_result.get("ok"):
        audiences = audiences_result.get("data", [])

    # Detectar saturacao
    saturation_data = None
    if audiences:
        writer({"agent": "audience_specialist", "status": "detecting_saturation", "progress": 40})
        adset_ids = [a.get("adset_id") for a in audiences if a.get("adset_id")][:20]

        if adset_ids:
            saturation_result = await detect_audience_saturation.ainvoke(
                {"adset_ids": adset_ids},
                config=config,
            )
            if isinstance(saturation_result, dict) and saturation_result.get("ok"):
                saturation_data = saturation_result.get("data")

    # Performance por audiencia
    performance_data = None
    if audiences:
        writer({"agent": "audience_specialist", "status": "analyzing_performance", "progress": 60})
        adset_ids = [a.get("adset_id") for a in audiences if a.get("adset_id")][:20]

        if adset_ids:
            perf_result = await get_audience_performance.ainvoke(
                {"adset_ids": adset_ids},
                config=config,
            )
            if isinstance(perf_result, dict) and perf_result.get("ok"):
                performance_data = perf_result.get("data")

    # Gerar relatorio com LLM
    writer({"agent": "audience_specialist", "status": "generating_report", "progress": 80})
    user_question = state["messages"][-1].content if state.get("messages") else ""

    model = get_model("analyst", config)
    prompt = _build_audience_prompt(
        audiences, saturation_data, performance_data, user_question,
    )
    response = await model.ainvoke(prompt)

    saturated_count = 0
    if saturation_data:
        saturated_count = len(saturation_data.get("saturated", []))

    writer({"agent": "audience_specialist", "status": "completed", "progress": 100})

    return {"agent_reports": [{
        "agent_id": "audience_specialist",
        "status": "completed",
        "summary": response.content,
        "data": {
            "total_audiences": len(audiences),
            "saturated_count": saturated_count,
            "saturation_data": saturation_data,
            "performance_data": performance_data,
        },
        "confidence": 0.80,
    }]}


def _build_audience_prompt(
    audiences: list,
    saturation: dict,
    performance: list,
    user_question: str,
) -> str:
    """Constroi prompt de analise de audiencias."""
    parts = [SYSTEM_PROMPT]
    parts.append(f"\n\nPergunta do usuario: {user_question}")
    parts.append(f"\n\nAudiencias encontradas: {len(audiences)}")

    if saturation:
        saturated = saturation.get("saturated", [])
        parts.append(f"\nAudiencias saturadas: {len(saturated)}")
        if saturated:
            parts.append(f"\nDetalhes saturacao: {saturated[:5]}")

    if performance:
        parts.append(f"\nPerformance por audiencia (top 5): {performance[:5]}")

    parts.append(
        "\n\nAnalise a segmentacao das audiencias, identifique saturacao, "
        "e recomende acoes para otimizar. Responda em portugues (Brasil)."
    )
    return "\n".join(parts)
