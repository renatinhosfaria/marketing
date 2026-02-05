"""AnomalyAgent - Especialista em deteccao de anomalias."""
from typing import List

from langchain_core.tools import BaseTool

from projects.agent.subagents.base import BaseSubagent
from projects.agent.subagents.anomaly.prompts import get_anomaly_prompt
from projects.agent.tools.anomaly_tools import (
    get_anomalies,
    get_critical_anomalies,
    get_anomalies_by_type
)


class AnomalyAgent(BaseSubagent):
    """Subagente especializado em deteccao de anomalias.

    Responsavel por:
    - Identificar problemas em campanhas
    - Priorizar por severidade
    - Alertar sobre situacoes criticas

    Attributes:
        AGENT_NAME: Nome identificador do subagente ("anomaly")
        AGENT_DESCRIPTION: Descricao das capacidades do agente
    """

    AGENT_NAME = "anomaly"
    AGENT_DESCRIPTION = "Detecta anomalias e problemas em campanhas"

    def get_tools(self) -> List[BaseTool]:
        """Retorna as 3 tools de anomalia.

        Returns:
            Lista com get_anomalies, get_critical_anomalies,
            get_anomalies_by_type
        """
        return [
            get_anomalies,
            get_critical_anomalies,
            get_anomalies_by_type
        ]

    def get_system_prompt(self) -> str:
        """Retorna o system prompt do AnomalyAgent.

        Returns:
            System prompt string com instrucoes especializadas
        """
        return get_anomaly_prompt()
