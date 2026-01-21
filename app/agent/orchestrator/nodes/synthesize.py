"""No synthesize do Orchestrator.

Responsavel por sintetizar resultados dos subagentes em resposta unificada.
"""
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.agent.orchestrator.state import OrchestratorState, PRIORITY_ORDER
from app.agent.orchestrator.prompts import get_synthesis_prompt
from app.agent.config import get_agent_settings
from app.agent.llm.provider import get_llm
from app.agent.subagents.state import AgentResult
from app.core.logging import get_logger

logger = get_logger("orchestrator.synthesize")


def prioritize_results(
    results: dict[str, AgentResult]
) -> list[tuple[str, AgentResult]]:
    """Ordena resultados por prioridade.

    Args:
        results: Dicionario de resultados por agente

    Returns:
        Lista de tuplas (nome, resultado) ordenada por prioridade
    """
    items = list(results.items())

    # Ordenar por PRIORITY_ORDER (menor = maior prioridade)
    items.sort(key=lambda x: PRIORITY_ORDER.get(x[0], 10))

    return items


def format_results_for_synthesis(results: dict[str, AgentResult]) -> str:
    """Formata resultados para o prompt de sintese.

    Args:
        results: Dicionario de resultados por agente

    Returns:
        String formatada com todos os resultados
    """
    sections = []

    # Ordenar por prioridade
    ordered = prioritize_results(results)

    for agent_name, result in ordered:
        if not isinstance(result, dict):
            continue

        success = result.get("success", False)
        data = result.get("data", {})
        error = result.get("error")
        tool_calls = result.get("tool_calls", [])

        # Header do agente
        status = "OK" if success else "ERROR"
        section = f"\n## {agent_name.upper()} {status}\n"

        if success and data:
            response = data.get("response", "")
            if response:
                section += f"\n{response}\n"

            # Adicionar info de tools usadas
            if tool_calls:
                section += f"\nTools utilizadas: {', '.join(tool_calls)}\n"

        elif error:
            section += f"\nErro: {error}\n"

        sections.append(section)

    return "\n---\n".join(sections)


async def synthesize(state: OrchestratorState) -> dict:
    """No que sintetiza resultados em resposta unificada.

    Usa LLM para combinar analises de multiplos subagentes
    em uma resposta coerente e acionavel.

    Args:
        state: Estado atual do orchestrator

    Returns:
        Atualizacoes para o estado
    """
    logger.info("Iniciando sintese de resultados")

    agent_results = state.get("agent_results", {})
    user_intent = state.get("user_intent", "general")

    if not agent_results:
        logger.warning("Nenhum resultado para sintetizar")
        return {
            "synthesized_response": "Nao foi possivel obter analises. Por favor, tente novamente.",
            "messages": [AIMessage(content="Nao foi possivel obter analises.")]
        }

    # Formatar resultados
    formatted_results = format_results_for_synthesis(agent_results)

    # Obter LLM para sintese
    settings = get_agent_settings()
    llm = get_llm(
        provider=settings.llm_provider,
        model=settings.llm_model,
        temperature=settings.synthesis_temperature,
        max_tokens=settings.synthesis_max_tokens
    )

    # Construir prompt
    synthesis_prompt = get_synthesis_prompt()

    messages = [
        SystemMessage(content=synthesis_prompt),
        HumanMessage(content=f"""
Intencao do usuario: {user_intent}

Resultados dos agentes especialistas:

{formatted_results}

Por favor, sintetize esses resultados em uma resposta clara e util.
""")
    ]

    try:
        # Chamar LLM
        response = await llm.ainvoke(messages)
        synthesized = response.content

        logger.info(f"Sintese concluida: {len(synthesized)} caracteres")

        return {
            "synthesized_response": synthesized,
            "messages": [AIMessage(content=synthesized)]
        }

    except Exception as e:
        logger.error(f"Erro na sintese: {e}")

        # Fallback: concatenar resultados
        fallback = _create_fallback_response(agent_results)

        return {
            "synthesized_response": fallback,
            "messages": [AIMessage(content=fallback)],
            "error": str(e)
        }


def _create_fallback_response(results: dict[str, AgentResult]) -> str:
    """Cria resposta fallback quando sintese falha.

    Args:
        results: Resultados dos agentes

    Returns:
        Resposta formatada basica
    """
    parts = ["Resultados da Analise:\n"]

    for agent_name, result in results.items():
        if not isinstance(result, dict):
            continue

        if result.get("success") and result.get("data"):
            response = result["data"].get("response", "")
            if response:
                parts.append(f"\n{agent_name.title()}:\n{response}\n")

    return "\n".join(parts) if len(parts) > 1 else "Analise nao disponivel."
