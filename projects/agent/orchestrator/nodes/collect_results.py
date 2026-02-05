"""No collect_results do Orchestrator.

Responsavel por coletar e agregar resultados dos subagentes.
"""
from typing import Any

from projects.agent.orchestrator.state import OrchestratorState
from projects.agent.subagents.state import AgentResult
from shared.core.logging import get_logger

logger = get_logger("orchestrator.collect_results")


def merge_subagent_results(
    existing: dict[str, AgentResult],
    new_results: list[dict]
) -> dict[str, AgentResult]:
    """Combina resultados existentes com novos resultados.

    Args:
        existing: Resultados ja coletados
        new_results: Novos resultados para adicionar

    Returns:
        Dicionario combinado de resultados
    """
    merged = dict(existing)

    for item in new_results:
        agent_name = item.get("agent_name")
        result = item.get("result", {})

        if agent_name:
            merged[agent_name] = result

    return merged


def calculate_confidence_score(results: dict[str, AgentResult]) -> float:
    """Calcula score de confianca baseado nos resultados.

    O score e calculado como proporcao de agentes que tiveram sucesso.

    Args:
        results: Resultados dos subagentes

    Returns:
        Score de 0.0 a 1.0
    """
    if not results:
        return 0.0

    successful = sum(
        1 for r in results.values()
        if isinstance(r, dict) and r.get("success", False)
    )

    return successful / len(results)


def collect_results(state: OrchestratorState) -> dict:
    """No que coleta resultados dos subagentes.

    Agrega todos os resultados retornados pelos subagentes
    apos execucao paralela.

    Args:
        state: Estado atual do orchestrator

    Returns:
        Atualizacoes para o estado
    """
    logger.info("Coletando resultados dos subagentes")

    agent_results = state.get("agent_results", {})

    # Log dos resultados
    successful = [
        name for name, r in agent_results.items()
        if isinstance(r, dict) and r.get("success", False)
    ]
    failed = [
        name for name, r in agent_results.items()
        if isinstance(r, dict) and not r.get("success", True)
    ]

    logger.info(
        f"Resultados coletados: {len(successful)} sucesso, {len(failed)} falha"
    )

    if failed:
        logger.warning("Agentes com falha: {failed}")

    # Calcular confidence score
    confidence = calculate_confidence_score(agent_results)

    logger.info("Confidence score: {confidence:.2f}")

    return {
        "agent_results": agent_results,
        "confidence_score": confidence,
    }


def reduce_agent_results(
    left: dict[str, AgentResult],
    right: dict[str, AgentResult]
) -> dict[str, AgentResult]:
    """Reducer para combinar resultados de multiplos subagentes.

    Usado pelo LangGraph para agregar resultados de nos paralelos.

    Args:
        left: Resultados anteriores
        right: Novos resultados

    Returns:
        Resultados combinados
    """
    result = dict(left) if left else {}

    if right:
        result.update(right)

    return result
