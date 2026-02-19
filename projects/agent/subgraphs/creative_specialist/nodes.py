"""
Nos do subgraph Especialista em Criativos.

Fluxo: fetch_ads -> analyze_fatigue -> recommend
O ultimo no (recommend) produz AgentReport para o synthesizer.
"""

from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer

from projects.agent.subgraphs.creative_specialist.state import CreativeSubgraphState
from projects.agent.tools.creative_tools import (
    get_ad_creatives,
    detect_creative_fatigue,
)
from projects.agent.llm.provider import get_model
from projects.agent.prompts.creative_specialist import SYSTEM_PROMPT

import structlog

logger = structlog.get_logger()


async def fetch_ads_node(
    state: CreativeSubgraphState,
    config: RunnableConfig,
):
    """Busca metadados dos anuncios da conta."""
    writer = get_stream_writer()
    writer({"agent": "creative_specialist", "status": "running", "progress": 0})

    scope = state.get("scope") or {}
    campaign_id = None
    entity_ids = scope.get("entity_ids")
    if entity_ids and scope.get("entity_type") == "campaign":
        campaign_id = entity_ids[0]

    writer({"agent": "creative_specialist", "status": "fetching_ads", "progress": 10})

    creatives_result = await get_ad_creatives.ainvoke(
        {"campaign_id": campaign_id},
        config=config,
    )

    ad_creatives = []
    if isinstance(creatives_result, dict) and creatives_result.get("ok"):
        ad_creatives = creatives_result.get("data", [])

    writer({"agent": "creative_specialist", "status": "ads_loaded", "progress": 30})

    return {"ad_creatives": ad_creatives}


async def analyze_fatigue_node(
    state: CreativeSubgraphState,
    config: RunnableConfig,
):
    """Analisa fadiga criativa dos anuncios."""
    writer = get_stream_writer()
    writer({"agent": "creative_specialist", "status": "analyzing_fatigue", "progress": 40})

    ad_creatives = state.get("ad_creatives", [])

    fatigue_analysis = None
    if ad_creatives:
        ad_ids = [ad.get("ad_id") for ad in ad_creatives if ad.get("ad_id")][:20]

        if ad_ids:
            scope = state.get("scope") or {}
            window_days = scope.get("lookback_days", 14)

            fatigue_result = await detect_creative_fatigue.ainvoke(
                {"ad_ids": ad_ids, "window_days": window_days},
                config=config,
            )

            if isinstance(fatigue_result, dict) and fatigue_result.get("ok"):
                fatigue_analysis = fatigue_result.get("data")

    writer({"agent": "creative_specialist", "status": "fatigue_analyzed", "progress": 70})

    return {"fatigue_analysis": fatigue_analysis}


async def recommend_node(
    state: CreativeSubgraphState,
    config: RunnableConfig,
):
    """Gera recomendacoes sobre criativos e produz AgentReport."""
    writer = get_stream_writer()
    writer({"agent": "creative_specialist", "status": "generating_recommendations", "progress": 80})

    ad_creatives = state.get("ad_creatives", [])
    fatigue_analysis = state.get("fatigue_analysis")
    user_question = state["messages"][-1].content if state.get("messages") else ""

    model = get_model("analyst", config)
    prompt = _build_creative_prompt(ad_creatives, fatigue_analysis, user_question)
    response = await model.ainvoke(prompt)

    fatigued_count = 0
    if fatigue_analysis:
        fatigued_count = len(fatigue_analysis.get("fatigued_ads", []))

    writer({"agent": "creative_specialist", "status": "completed", "progress": 100})

    return {
        "recommendation": response.content,
        "agent_reports": [{
            "agent_id": "creative_specialist",
            "status": "completed",
            "summary": response.content,
            "data": {
                "total_ads": len(ad_creatives),
                "fatigued_count": fatigued_count,
                "fatigue_analysis": fatigue_analysis,
            },
            "confidence": 0.80,
        }],
    }


def _build_creative_prompt(
    ad_creatives: list,
    fatigue_analysis: dict,
    user_question: str,
) -> str:
    """Constroi prompt de analise criativa."""
    parts = [SYSTEM_PROMPT]
    parts.append(f"\n\nPergunta do usuario: {user_question}")
    parts.append(f"\n\nAnuncios encontrados: {len(ad_creatives)}")

    if fatigue_analysis:
        fatigued = fatigue_analysis.get("fatigued_ads", [])
        healthy = fatigue_analysis.get("healthy_ads", [])
        parts.append(f"\nAnuncios com fadiga: {len(fatigued)}")
        parts.append(f"\nAnuncios saudaveis: {len(healthy)}")
        if fatigued:
            parts.append(f"\nDetalhes fadiga: {fatigued[:5]}")

    parts.append(
        "\n\nAnalise a saude dos criativos, identifique fadiga criativa, "
        "e recomende acoes. Responda em portugues (Brasil)."
    )
    return "\n".join(parts)
