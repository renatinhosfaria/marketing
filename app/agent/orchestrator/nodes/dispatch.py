"""No dispatch_agents do Orchestrator.

Responsavel por disparar subagentes em paralelo usando Send().
"""
from typing import Any
import time

from langgraph.types import Send

from app.agent.orchestrator.state import OrchestratorState
from app.core.logging import get_logger
from app.core.tracing.decorators import log_span
from app.core.tracing.events import (
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

    if not required_agents:
        logger.warning("Nenhum agente para disparar")
        return []

    if not execution_plan:
        logger.warning("Sem plano de execucao")
        return []

    sends = []
    tasks = execution_plan.get("tasks", {})

    for agent_name in required_agents:
        task = tasks.get(agent_name, {})

        # Criar argumento para o subagente
        task_dict = {
            "description": task.get("description", f"Execute {agent_name}"),
            "context": task.get("context", {}),
            "priority": task.get("priority", 10),
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
        """Executa um subagente especifico.

        Args:
            state: Estado passado pelo Send()

        Returns:
            Resultado do subagente
        """
        from app.agent.subagents import get_subagent

        task = state.get("task", {})
        task_description = task.get("description", f"Execute {agent_name}")

        # Logar início da execução
        log_subagent_started(
            subagent=agent_name,
            task_description=task_description
        )

        logger.info(f"Executando subagente: {agent_name}")

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

            # Logar conclusão com sucesso
            log_subagent_completed(
                subagent=agent_name,
                success=result.get("success", False),
                duration_ms=result.get("duration_ms", duration_ms),
                tool_calls=result.get("tool_calls", [])
            )

            logger.info(
                f"Subagente {agent_name} concluido: "
                f"success={result.get('success')}, "
                f"duration={result.get('duration_ms')}ms"
            )

            return {
                "agent_name": agent_name,
                "result": result
            }

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000

            # Logar falha
            log_subagent_failed(
                subagent=agent_name,
                error_type=type(e).__name__,
                error_message=str(e),
                duration_ms=duration_ms
            )

            logger.error(f"Erro no subagente {agent_name}: {e}")

            return {
                "agent_name": agent_name,
                "result": {
                    "agent_name": agent_name,
                    "success": False,
                    "data": None,
                    "error": str(e),
                    "duration_ms": duration_ms,
                    "tool_calls": []
                }
            }

    return subagent_node
