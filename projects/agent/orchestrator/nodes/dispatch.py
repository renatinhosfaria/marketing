"""No dispatch_agents do Orchestrator.

Responsavel por disparar subagentes em paralelo usando Send().
"""
from typing import Any
import asyncio
import time

from langchain_core.messages import HumanMessage
from langgraph.types import Send

from projects.agent.config import get_agent_settings
from projects.agent.orchestrator.state import OrchestratorState
from shared.core.logging import get_logger
from shared.core.tracing.decorators import log_span
from shared.core.tracing.events import (
    log_subagent_dispatched,
    log_subagent_started,
    log_subagent_completed,
    log_subagent_failed
)

logger = get_logger("orchestrator.dispatch")


@log_span("dispatch_agents", log_args=False, log_result=False)
def dispatch_agents(state: OrchestratorState) -> list[Send]:
    """No que dispara subagentes em paralelo.

    Usa Send() do LangGraph para executar multiplos subagentes
    simultaneamente.

    Args:
        state: Estado atual do orchestrator

    Returns:
        Lista de objetos Send para cada subagente
    """
    required_agents = state.get("required_agents", [])
    execution_plan = state.get("execution_plan")

    # Extrair pergunta original do usuario
    user_question = ""
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, HumanMessage):
            user_question = str(msg.content).strip() if msg.content else ""
            break
        if isinstance(msg, dict) and msg.get("role") == "user":
            user_question = str(msg.get("content", "")).strip()
            break

    if not required_agents:
        logger.warning("Nenhum agente para disparar")
        return []

    if not execution_plan:
        logger.warning("Sem plano de execucao")
        return []

    sends = []
    tasks = execution_plan.get("tasks", {})
    max_parallel = get_agent_settings().max_parallel_subagents

    if len(required_agents) > max_parallel:
        ordered_agents = sorted(
            required_agents,
            key=lambda name: tasks.get(name, {}).get("priority", 10),
        )
        required_agents = ordered_agents[:max_parallel]
        logger.warning(
            "Limitando dispatch de subagentes por max_parallel_subagents",
            selected_agents=required_agents,
            max_parallel=max_parallel,
        )

    for agent_name in required_agents:
        task = tasks.get(agent_name, {})

        # Criar argumento para o subagente
        task_dict = {
            "description": task.get("description", f"Execute {agent_name}"),
            "context": task.get("context", {}),
            "priority": task.get("priority", 10),
            "user_question": user_question,
        }

        arg = {
            "task": task_dict,
            "config_id": state.get("config_id"),
            "user_id": state.get("user_id"),
            "thread_id": state.get("thread_id"),
            "messages": list(state.get("messages", [])),
        }

        # Criar Send para o subagente
        send = Send(
            node=f"subagent_{agent_name}",
            arg=arg
        )
        sends.append(send)

        # Logar dispatch do subagente
        log_subagent_dispatched(
            subagent=agent_name,
            task=task_dict
        )

        logger.debug(f"Dispatch criado para: {agent_name}")

    logger.info(f"Disparando {len(sends)} subagentes em paralelo")

    return sends


def create_subagent_node(agent_name: str):
    """Factory para criar no de subagente.

    Args:
        agent_name: Nome do subagente

    Returns:
        Funcao async que executa o subagente
    """
    async def subagent_node(state: dict) -> dict:
        """Executa um subagente especifico com logica de retry.

        Args:
            state: Estado passado pelo Send()

        Returns:
            Resultado do subagente
        """
        from projects.agent.subagents import get_subagent

        settings = get_agent_settings()
        max_retries = settings.subagent_max_retries
        retry_delay = settings.subagent_retry_delay

        task = state.get("task", {})
        task_description = task.get("description", f"Execute {agent_name}")

        # Logar início da execução
        log_subagent_started(
            subagent=agent_name,
            task_description=task_description
        )

        logger.info("Executando subagente", detail=str(agent_name))

        last_result = None

        for attempt in range(max_retries + 1):
            start_time = time.time()

            try:
                # Obter instancia do subagente
                agent = get_subagent(agent_name)

                # Executar
                result = await agent.run(
                    task=task,
                    config_id=state.get("config_id", 0),
                    user_id=state.get("user_id", 0),
                    thread_id=state.get("thread_id", ""),
                    messages=state.get("messages", [])
                )

                duration_ms = (time.time() - start_time) * 1000

                if result.get("success"):
                    # Logar conclusão com sucesso
                    log_subagent_completed(
                        subagent=agent_name,
                        success=True,
                        duration_ms=result.get("duration_ms", duration_ms),
                        tool_calls=result.get("tool_calls", [])
                    )

                    logger.info(
                        f"Subagente {agent_name} concluido na tentativa "
                        f"{attempt + 1}: success=True, "
                        f"duration={result.get('duration_ms', duration_ms)}ms"
                    )

                    return {
                        "agent_results": {agent_name: result}
                    }

                # Resultado sem sucesso
                last_result = {
                    "agent_results": {agent_name: result}
                }

                if attempt < max_retries:
                    logger.warning(
                        f"Subagente {agent_name} falhou na tentativa "
                        f"{attempt + 1}/{max_retries + 1}, "
                        f"retentando em {retry_delay * (attempt + 1)}s"
                    )
                    await asyncio.sleep(retry_delay * (attempt + 1))
                else:
                    # Ultima tentativa, logar falha final
                    log_subagent_completed(
                        subagent=agent_name,
                        success=False,
                        duration_ms=result.get("duration_ms", duration_ms),
                        tool_calls=result.get("tool_calls", [])
                    )

                    logger.error(
                        f"Subagente {agent_name} falhou apos "
                        f"{max_retries + 1} tentativas: "
                        f"error={result.get('error')}"
                    )

            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000

                last_result = {
                    "agent_results": {
                        agent_name: {
                            "agent_name": agent_name,
                            "success": False,
                            "data": None,
                            "error": str(e),
                            "duration_ms": duration_ms,
                            "tool_calls": []
                        }
                    }
                }

                if attempt < max_retries:
                    logger.warning(
                        f"Excecao no subagente {agent_name} na tentativa "
                        f"{attempt + 1}/{max_retries + 1}: {e}, "
                        f"retentando em {retry_delay * (attempt + 1)}s"
                    )
                    await asyncio.sleep(retry_delay * (attempt + 1))
                else:
                    # Ultima tentativa, logar falha
                    log_subagent_failed(
                        subagent=agent_name,
                        error_type=type(e).__name__,
                        error_message=str(e),
                        duration_ms=duration_ms
                    )

                    logger.error(
                        f"Subagente {agent_name} falhou com excecao apos "
                        f"{max_retries + 1} tentativas: {e}"
                    )

        return last_result

    return subagent_node
