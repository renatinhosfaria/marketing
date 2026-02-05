"""No synthesize do Orchestrator.

Responsavel por sintetizar resultados dos subagentes em resposta unificada.
"""
from typing import Any
import time

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from projects.agent.orchestrator.state import OrchestratorState, PRIORITY_ORDER
from projects.agent.orchestrator.prompts import get_synthesis_prompt
from projects.agent.config import get_agent_settings
from projects.agent.llm.provider import get_llm
from projects.agent.subagents.state import AgentResult
from shared.core.logging import get_logger
from shared.core.tracing.decorators import log_span
from shared.core.tracing.events import (
    log_synthesis_start,
    log_synthesis_completed,
    log_llm_call
)

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


@log_span("synthesis", log_args=False, log_result=False)
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

    # Logar início da síntese
    log_synthesis_start(
        subagent_results_count=len(agent_results),
        strategy="comprehensive"
    )

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

    prompt_content = f"""
Intencao do usuario: {user_intent}

Resultados dos agentes especialistas:

{formatted_results}

Por favor, sintetize esses resultados em uma resposta clara e util.
"""

    messages = [
        SystemMessage(content=synthesis_prompt),
        HumanMessage(content=prompt_content)
    ]

    start_time = time.time()

    try:
        # Chamar LLM
        response = await llm.ainvoke(messages)
        synthesized = response.content

        duration_ms = (time.time() - start_time) * 1000

        # Logar chamada ao LLM
        prompt_tokens = getattr(response, "response_metadata", {}).get("token_usage", {}).get("prompt_tokens", 0)
        response_tokens = getattr(response, "response_metadata", {}).get("token_usage", {}).get("completion_tokens", 0)
        total_tokens = prompt_tokens + response_tokens

        # Construir prompt completo para logging
        full_prompt = f"{synthesis_prompt}\n\n{prompt_content}"

        log_llm_call(
            prompt=full_prompt,
            response=synthesized,
            prompt_tokens=prompt_tokens,
            response_tokens=response_tokens,
            duration_ms=duration_ms,
            prompt_type="synthesis",
            model=settings.llm_model
        )

        # Logar conclusão da síntese
        log_synthesis_completed(
            success=True,
            duration_ms=duration_ms,
            response_length=len(synthesized),
            model=settings.llm_model,
            tokens_used=total_tokens
        )

        logger.info("Sintese concluida: {len(synthesized)} caracteres")

        return {
            "synthesized_response": synthesized,
            "messages": [AIMessage(content=synthesized)]
        }

    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000

        logger.error("Erro na sintese", error=str(e))

        # Logar falha na síntese
        log_synthesis_completed(
            success=False,
            duration_ms=duration_ms,
            response_length=0,
            model=settings.llm_model,
            tokens_used=0
        )

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
