"""
Nos do subgraph Monitor de Saude & Anomalias.

Fluxo: fetch_metrics -> anomaly_detection -> diagnose
O ultimo no (diagnose) produz AgentReport para o synthesizer.
"""

from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer

from projects.agent.subgraphs.health_monitor.state import HealthSubgraphState
from projects.agent.tools.health_tools import (
    detect_anomalies,
    get_classifications,
)
from projects.agent.llm.provider import get_model
from projects.agent.prompts.health_monitor import SYSTEM_PROMPT

import structlog

logger = structlog.get_logger()


async def fetch_metrics_node(
    state: HealthSubgraphState,
    config: RunnableConfig,
):
    """Busca metricas iniciais e classificacoes das entidades.

    Usa tools de classificacao para obter tiers atuais.
    """
    writer = get_stream_writer()
    writer({"agent": "health_monitor", "status": "running", "progress": 0})

    scope = state.get("scope") or {}
    entity_type = scope.get("entity_type", "campaign")

    writer({"agent": "health_monitor", "status": "fetching_classifications", "progress": 10})

    # Buscar classificacoes atuais
    classifications_result = await get_classifications.ainvoke(
        {"entity_type": entity_type},
        config=config,
    )

    classifications = None
    classifications_error = None
    if isinstance(classifications_result, dict) and classifications_result.get("ok"):
        classifications = classifications_result.get("data")
    elif isinstance(classifications_result, dict) and classifications_result.get("error"):
        err = classifications_result["error"]
        classifications_error = f"[{err.get('code', 'UNKNOWN')}] {err.get('message', 'Erro desconhecido')}"

    writer({"agent": "health_monitor", "status": "classifications_loaded", "progress": 30})

    return {"classifications": classifications, "classifications_error": classifications_error}


async def anomaly_detection_node(
    state: HealthSubgraphState,
    config: RunnableConfig,
):
    """Executa deteccao de anomalias usando ML API.

    Usa IsolationForest + Z-score + IQR via endpoint /anomalies/detect.
    """
    writer = get_stream_writer()
    writer({"agent": "health_monitor", "status": "running_anomaly_detection", "progress": 40})

    scope = state.get("scope") or {}
    entity_type = scope.get("entity_type", "campaign")
    entity_ids = scope.get("entity_ids")
    lookback_days = scope.get("lookback_days", 1)

    # Detectar anomalias
    anomaly_result = await detect_anomalies.ainvoke(
        {
            "entity_type": entity_type,
            "entity_ids": entity_ids,
            "days": lookback_days,
        },
        config=config,
    )

    anomaly_results = None
    anomaly_error = None
    if isinstance(anomaly_result, dict) and anomaly_result.get("ok"):
        anomaly_results = anomaly_result.get("data")
    elif isinstance(anomaly_result, dict) and anomaly_result.get("error"):
        err = anomaly_result["error"]
        anomaly_error = f"[{err.get('code', 'UNKNOWN')}] {err.get('message', 'Erro desconhecido')}"

    writer({"agent": "health_monitor", "status": "anomalies_detected", "progress": 70})

    return {"anomaly_results": anomaly_results, "anomaly_error": anomaly_error}


async def diagnose_node(
    state: HealthSubgraphState,
    config: RunnableConfig,
):
    """Diagnostica anomalias e produz AgentReport para o synthesizer.

    Usa LLM para gerar diagnostico textual baseado nas anomalias e classificacoes.
    """
    writer = get_stream_writer()
    writer({"agent": "health_monitor", "status": "diagnosing", "progress": 80})

    anomalies = state.get("anomaly_results") or {}
    classifications = state.get("classifications") or {}
    anomaly_error = state.get("anomaly_error")
    classifications_error = state.get("classifications_error")

    # LLM gera diagnostico textual
    model = get_model("analyst", config)

    prompt = _build_diagnosis_prompt(
        anomalies, classifications, anomaly_error, classifications_error
    )
    diagnosis = await model.ainvoke(prompt)

    anomaly_count = 0
    if isinstance(anomalies, dict):
        anomaly_count = anomalies.get("detected_count", 0)
        if not anomaly_count and "anomalies" in anomalies:
            anomaly_count = len(anomalies["anomalies"])

    writer({"agent": "health_monitor", "status": "completed", "progress": 100})

    return {
        "diagnosis": diagnosis.content,
        "agent_reports": [{
            "agent_id": "health_monitor",
            "status": "completed",
            "summary": diagnosis.content,
            "data": {
                "anomaly_count": anomaly_count,
                "classifications": classifications,
            },
            "confidence": 0.85,
        }],
    }


def _build_diagnosis_prompt(
    anomalies: dict,
    classifications: dict,
    anomaly_error: str = None,
    classifications_error: str = None,
) -> str:
    """Constroi prompt de diagnostico a partir dos dados coletados."""
    parts = [SYSTEM_PROMPT]
    parts.append("\n\n--- Dados para diagnostico ---")

    if anomaly_error:
        parts.append(
            f"\n**ERRO ao buscar anomalias**: {anomaly_error}. "
            "Nao foi possivel verificar anomalias neste momento."
        )
    elif anomalies:
        parts.append(f"\nAnomalias detectadas: {anomalies}")
    else:
        parts.append(
            "\nDeteccao de anomalias executada com sucesso: nenhuma anomalia encontrada."
        )

    if classifications_error:
        parts.append(
            f"\n**ERRO ao buscar classificacoes**: {classifications_error}. "
            "Nao foi possivel obter as classificacoes neste momento."
        )
    elif classifications:
        parts.append(f"\nClassificacoes atuais: {classifications}")
    else:
        parts.append(
            "\nBusca de classificacoes executada com sucesso: "
            "nenhuma classificacao disponivel no periodo."
        )

    parts.append(
        "\n\nCom base nos dados acima, faca um diagnostico de saude das campanhas. "
        "Se houve erros na coleta de dados, informe ao usuario que parte da analise "
        "ficou indisponivel e sugira tentar novamente em instantes. "
        "Identifique problemas criticos, tendencias preocupantes e oportunidades. "
        "Responda em portugues (Brasil)."
    )

    return "\n".join(parts)
