"""No dispatch_agents do Orchestrator.

Responsavel por disparar subagentes em paralelo usando Send().
"""
from typing import Any

from langgraph.types import Send

from app.agent.orchestrator.state import OrchestratorState
from app.core.logging import get_logger

logger = get_logger("orchestrator.dispatch")


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
        arg = {
            "task": {
                "description": task.get("description", f"Execute {agent_name}"),
                "context": task.get("context", {}),
                "priority": task.get("priority", 10),
            },
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

        logger.info(f"Executando subagente: {agent_name}")

        try:
            # Obter instancia do subagente
            agent = get_subagent(agent_name)

            # Executar
            result = await agent.run(
                task=state.get("task", {}),
                config_id=state.get("config_id", 0),
                user_id=state.get("user_id", 0),
                thread_id=state.get("thread_id", ""),
                messages=state.get("messages", [])
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
            logger.error(f"Erro no subagente {agent_name}: {e}")
            return {
                "agent_name": agent_name,
                "result": {
                    "agent_name": agent_name,
                    "success": False,
                    "data": None,
                    "error": str(e),
                    "duration_ms": 0,
                    "tool_calls": []
                }
            }

    return subagent_node
