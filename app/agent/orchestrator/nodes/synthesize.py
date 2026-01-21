"""No synthesize do Orchestrator.

Responsavel por sintetizar os resultados dos subagentes em uma
resposta final coerente para o usuario.
"""
import os
import importlib.util
import json
from typing import Any


# Carregar state.py diretamente para evitar problemas de import circular
_state_path = os.path.join(
    os.path.dirname(__file__),
    '..', 'state.py'
)
_state_path = os.path.abspath(_state_path)

_spec = importlib.util.spec_from_file_location("orchestrator_state", _state_path)
_state_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_state_module)

OrchestratorState = _state_module.OrchestratorState
AgentResult = _state_module.AgentResult


# Ordem de prioridade para sintese (menor = maior prioridade)
SYNTHESIS_PRIORITY: dict[str, int] = {
    "anomaly": 1,        # Problemas criticos primeiro
    "recommendation": 2,  # Acoes sugeridas
    "classification": 3,  # Contexto de performance
    "forecast": 4,        # Projecoes futuras
    "campaign": 5,        # Dados de campanhas
    "analysis": 6,        # Analises complementares
}

# Templates de secoes para resposta
SECTION_TEMPLATES: dict[str, str] = {
    "anomaly": "## Alertas e Problemas\n{content}",
    "recommendation": "## Recomendacoes\n{content}",
    "classification": "## Performance\n{content}",
    "forecast": "## Previsoes\n{content}",
    "campaign": "## Campanhas\n{content}",
    "analysis": "## Analise\n{content}",
}

# Template padrao para agentes desconhecidos
DEFAULT_SECTION_TEMPLATE = "## {agent_name}\n{content}"


def calculate_confidence_score(agent_results: dict[str, AgentResult]) -> float:
    """Calcula score de confianca baseado na taxa de sucesso.

    Args:
        agent_results: Dicionario com resultados dos subagentes

    Returns:
        Float entre 0.0 e 1.0 representando a taxa de sucesso
    """
    if not agent_results:
        return 0.0

    total = len(agent_results)
    successful = sum(1 for result in agent_results.values() if result.get("success", False))

    return successful / total


def _format_data_content(data: dict[str, Any]) -> str:
    """Formata dados do resultado em texto legivel.

    Args:
        data: Dicionario de dados do resultado

    Returns:
        String formatada com os dados
    """
    if not data:
        return "Sem dados disponiveis."

    lines = []
    for key, value in data.items():
        if isinstance(value, list):
            if value:
                # Lista com itens
                items_str = ", ".join(str(item) for item in value[:5])  # Limita a 5 itens
                if len(value) > 5:
                    items_str += f" ... (+{len(value) - 5} mais)"
                lines.append(f"- **{key}**: {items_str}")
            else:
                lines.append(f"- **{key}**: Nenhum item encontrado")
        elif isinstance(value, dict):
            # Dicionario aninhado - formatar como JSON simplificado
            try:
                json_str = json.dumps(value, ensure_ascii=False, indent=2)
                lines.append(f"- **{key}**:\n```json\n{json_str}\n```")
            except (TypeError, ValueError):
                lines.append(f"- **{key}**: {value}")
        else:
            lines.append(f"- **{key}**: {value}")

    return "\n".join(lines)


def format_agent_section(agent_name: str, result: AgentResult) -> str:
    """Formata o resultado de um agente em uma secao markdown.

    Args:
        agent_name: Nome do agente
        result: Resultado do agente (AgentResult)

    Returns:
        String formatada como secao markdown
    """
    # Verificar se o resultado tem erro
    if not result.get("success", False):
        error_msg = result.get("error", "Erro desconhecido")
        content = f"*Erro na execucao:* {error_msg}"
    else:
        # Formatar dados do resultado
        data = result.get("data")
        if data:
            content = _format_data_content(data)
        else:
            content = "Execucao concluida sem dados retornados."

    # Obter template apropriado
    if agent_name in SECTION_TEMPLATES:
        template = SECTION_TEMPLATES[agent_name]
        return template.format(content=content)
    else:
        # Template padrao para agentes desconhecidos
        return DEFAULT_SECTION_TEMPLATE.format(
            agent_name=agent_name.replace("_", " ").title(),
            content=content
        )


def order_results_by_priority(
    agent_results: dict[str, AgentResult]
) -> list[tuple[str, AgentResult]]:
    """Ordena resultados por prioridade de sintese.

    Args:
        agent_results: Dicionario com resultados dos subagentes

    Returns:
        Lista de tuplas (agent_name, result) ordenada por prioridade
    """
    if not agent_results:
        return []

    # Converter para lista de tuplas com prioridade
    results_with_priority = []
    for agent_name, result in agent_results.items():
        # Agentes desconhecidos recebem prioridade 99 (ultima posicao)
        priority = SYNTHESIS_PRIORITY.get(agent_name, 99)
        results_with_priority.append((priority, agent_name, result))

    # Ordenar por prioridade (menor primeiro)
    results_with_priority.sort(key=lambda x: x[0])

    # Retornar apenas (agent_name, result)
    return [(name, result) for _, name, result in results_with_priority]


def synthesize_response(agent_results: dict[str, AgentResult]) -> str:
    """Sintetiza resultados dos subagentes em uma resposta unificada.

    Args:
        agent_results: Dicionario com resultados dos subagentes

    Returns:
        String com resposta sintetizada em formato markdown
    """
    if not agent_results:
        return ""

    # Ordenar resultados por prioridade
    ordered_results = order_results_by_priority(agent_results)

    # Formatar cada secao
    sections = []
    for agent_name, result in ordered_results:
        section = format_agent_section(agent_name, result)
        sections.append(section)

    # Combinar secoes com separador
    return "\n\n".join(sections)


def synthesize(state: OrchestratorState) -> dict[str, Any]:
    """No principal de sintese do orchestrator.

    Combina resultados de todos os subagentes em uma resposta
    final coerente para o usuario.

    Args:
        state: Estado atual do orchestrator

    Returns:
        Dict com 'synthesized_response' e 'confidence_score'
    """
    # Obter resultados dos subagentes
    agent_results = state.get("agent_results", {})

    # Calcular score de confianca
    confidence_score = calculate_confidence_score(agent_results)

    # Sintetizar resposta
    synthesized_response = synthesize_response(agent_results)

    return {
        "synthesized_response": synthesized_response,
        "confidence_score": confidence_score
    }
