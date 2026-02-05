"""RecommendationAgent - Especialista em recomendacoes."""
from typing import List

from langchain_core.tools import BaseTool

from projects.agent.subagents.base import BaseSubagent
from projects.agent.subagents.recommendation.prompts import get_recommendation_prompt
from projects.agent.tools.recommendation_tools import (
    get_recommendations,
    get_recommendations_by_type,
    get_high_priority_recommendations
)


class RecommendationAgent(BaseSubagent):
    """Subagente especializado em recomendacoes.

    Responsavel por:
    - Fornecer acoes acionaveis
    - Priorizar por impacto
    - Sugerir otimizacoes especificas

    Attributes:
        AGENT_NAME: Nome identificador do subagente ("recommendation")
        AGENT_DESCRIPTION: Descricao das capacidades do agente
    """

    AGENT_NAME = "recommendation"
    AGENT_DESCRIPTION = "Fornece recomendacoes de acoes para campanhas"

    def get_tools(self) -> List[BaseTool]:
        """Retorna as 3 tools de recomendacao.

        Returns:
            Lista com get_recommendations, get_recommendations_by_type,
            get_high_priority_recommendations
        """
        return [
            get_recommendations,
            get_recommendations_by_type,
            get_high_priority_recommendations
        ]

    def get_system_prompt(self) -> str:
        """Retorna o system prompt do RecommendationAgent.

        Returns:
            System prompt string
        """
        return get_recommendation_prompt()
