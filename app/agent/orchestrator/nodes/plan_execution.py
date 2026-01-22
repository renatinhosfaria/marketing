"""No plan_execution do Orchestrator.

Responsavel por criar o plano de execucao baseado na intencao detectada.
"""
from typing import Any

from app.agent.orchestrator.state import (
    OrchestratorState,
    ExecutionPlan,
    INTENT_TO_AGENTS,
    get_agents_for_intent
)
from app.agent.config import get_agent_settings
from app.core.logging import get_logger
from app.core.tracing.decorators import log_span
from app.core.tracing.events import log_subagents_selected

logger = get_logger("orchestrator.plan_execution")


# Descricoes de tarefas por agente
AGENT_TASK_DESCRIPTIONS = {
    "classification": "Analisar classificacao de performance das campanhas por tier",
    "anomaly": "Identificar anomalias e problemas nas campanhas",
    "forecast": "Gerar previsoes de CPL e leads para os proximos dias",
    "recommendation": "Fornecer recomendacoes de otimizacao priorizadas",
    "campaign": "Coletar dados detalhados das campanhas",
    "analysis": "Realizar analises avancadas e comparacoes",
}

# Prioridades por agente (menor = maior prioridade)
AGENT_PRIORITIES = {
    "anomaly": 1,
    "classification": 2,
    "recommendation": 3,
    "forecast": 4,
    "campaign": 5,
    "analysis": 6,
}


def get_agent_timeout(agent_name: str) -> int:
    """Retorna timeout para um agente especifico.

    Args:
        agent_name: Nome do agente

    Returns:
        Timeout em segundos
    """
    settings = get_agent_settings()
    timeout_attr = f"timeout_{agent_name}"
    return getattr(settings, timeout_attr, 30)


def create_execution_plan(
    intent: str,
    config_id: int,
    context: dict[str, Any] = None
) -> ExecutionPlan:
    """Cria plano de execucao para uma intencao.

    Args:
        intent: Intencao detectada
        config_id: ID da configuracao
        context: Contexto adicional (opcional)

    Returns:
        Plano de execucao
    """
    # Obter agentes necessarios
    agents = get_agents_for_intent(intent)

    # Criar tasks para cada agente
    tasks = {}
    max_timeout = 0

    for agent_name in agents:
        timeout = get_agent_timeout(agent_name)
        max_timeout = max(max_timeout, timeout)

        tasks[agent_name] = {
            "description": AGENT_TASK_DESCRIPTIONS.get(
                agent_name,
                f"Executar analise de {agent_name}"
            ),
            "context": {
                "config_id": config_id,
                "intent": intent,
                **(context or {})
            },
            "priority": AGENT_PRIORITIES.get(agent_name, 10),
        }

    # Calcular timeout total (max dos agentes + margem)
    total_timeout = max_timeout + 30  # 30s de margem para sintese

    return ExecutionPlan(
        agents=agents,
        tasks=tasks,
        parallel=True,  # Sempre paralelo quando possivel
        timeout=total_timeout
    )


@log_span("execution_planning", log_args=False, log_result=False)
def plan_execution(state: OrchestratorState) -> dict:
    """No que cria o plano de execucao.

    Baseado na intencao detectada, seleciona os subagentes necessarios
    e cria um plano de execucao.

    Args:
        state: Estado atual do orchestrator

    Returns:
        Atualizacoes para o estado
    """
    intent = state.get("user_intent", "general")
    config_id = state.get("config_id", 0)

    logger.info(f"Criando plano de execucao para intencao: {intent}")

    # Criar plano
    plan = create_execution_plan(intent, config_id)

    # Logar seleção de subagentes
    log_subagents_selected(
        subagents=plan["agents"],
        reasoning=f"Intent '{intent}' mapped to {len(plan['agents'])} agents: {', '.join(plan['agents'])}",
        parallel=plan["parallel"]
    )

    logger.info(
        f"Plano criado: {len(plan['agents'])} agentes, "
        f"parallel={plan['parallel']}, timeout={plan['timeout']}s"
    )
    logger.debug(f"Agentes selecionados: {plan['agents']}")

    return {
        "required_agents": plan["agents"],
        "execution_plan": plan,
    }
