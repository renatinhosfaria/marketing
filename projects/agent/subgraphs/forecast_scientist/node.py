"""
No simples do Cientista de Previsao (sem subgraph).

Funcao unica que gera previsoes e retorna AgentReport.
"""

from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer

from projects.agent.tools.forecast_tools import generate_forecast
from projects.agent.llm.provider import get_model
from projects.agent.prompts.forecast_scientist import SYSTEM_PROMPT

import structlog

logger = structlog.get_logger()


async def forecast_node(state: dict, config: RunnableConfig):
    """Gera previsoes e retorna AgentReport.

    No simples (sem subgraph) pois o fluxo e direto:
    busca entidade, gera forecast, interpreta resultado.
    """
    writer = get_stream_writer()
    writer({"agent": "forecast_scientist", "status": "running", "progress": 0})

    scope = state.get("scope") or {}
    entity_ids = scope.get("entity_ids") or []
    user_question = state["messages"][-1].content if state.get("messages") else ""

    forecasts = []
    forecast_errors = []

    # Gerar previsoes para entidades no scope
    if entity_ids:
        for i, entity_id in enumerate(entity_ids[:5]):  # Max 5 entidades
            writer({
                "agent": "forecast_scientist",
                "status": f"generating_forecast_{i + 1}",
                "progress": 10 + (i * 15),
            })

            for metric in ["cpl", "leads"]:
                result = await generate_forecast.ainvoke(
                    {
                        "entity_id": entity_id,
                        "metric": metric,
                        "horizon_days": scope.get("lookback_days", 7),
                    },
                    config=config,
                )

                if isinstance(result, dict) and result.get("ok"):
                    forecasts.append({
                        "entity_id": entity_id,
                        "metric": metric,
                        "data": result.get("data"),
                    })
                else:
                    forecast_errors.append({
                        "entity_id": entity_id,
                        "metric": metric,
                        "error": result.get("error", {}).get("message", "Erro desconhecido"),
                    })
    else:
        writer({"agent": "forecast_scientist", "status": "no_entities", "progress": 50})

    # Gerar interpretacao com LLM
    writer({"agent": "forecast_scientist", "status": "interpreting_results", "progress": 80})

    model = get_model("analyst", config)
    prompt = _build_forecast_prompt(forecasts, forecast_errors, user_question)
    response = await model.ainvoke(prompt)

    writer({"agent": "forecast_scientist", "status": "completed", "progress": 100})

    return {"agent_reports": [{
        "agent_id": "forecast_scientist",
        "status": "completed",
        "summary": response.content,
        "data": {
            "forecasts_generated": len(forecasts),
            "forecast_errors": len(forecast_errors),
            "forecasts": forecasts,
        },
        "confidence": 0.75,
    }]}


def _build_forecast_prompt(
    forecasts: list,
    errors: list,
    user_question: str,
) -> str:
    """Constroi prompt de interpretacao de previsoes."""
    parts = [SYSTEM_PROMPT]
    parts.append(f"\n\nPergunta do usuario: {user_question}")

    if forecasts:
        parts.append(f"\n\nPrevisoes geradas ({len(forecasts)}):")
        for f in forecasts[:10]:
            parts.append(f"\n- {f['entity_id']} ({f['metric']}): {f['data']}")

    if errors:
        parts.append(f"\n\nErros de previsao ({len(errors)}):")
        for e in errors[:5]:
            parts.append(f"\n- {e['entity_id']} ({e['metric']}): {e['error']}")

    if not forecasts and not errors:
        parts.append(
            "\n\nNenhuma entidade especificada para previsao. "
            "Sugira ao usuario especificar campanhas ou metricas."
        )

    parts.append(
        "\n\nInterprete as previsoes, destaque tendencias relevantes "
        "e faca recomendacoes baseadas nos dados. Responda em portugues (Brasil)."
    )
    return "\n".join(parts)
