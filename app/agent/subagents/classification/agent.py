"""ClassificationAgent - Especialista em classificacao de campanhas."""
from typing import List

from langchain_core.tools import BaseTool

from app.agent.subagents.base import BaseSubagent
from app.agent.subagents.classification.prompts import get_classification_prompt
from app.agent.tools.classification_tools import (
    get_classifications,
    get_campaign_tier,
    get_high_performers,
    get_underperformers
)


class ClassificationAgent(BaseSubagent):
    """Subagente especializado em classificacao de performance.

    Responsavel por:
    - Analisar tiers de campanhas (HIGH_PERFORMER, MODERATE, LOW, UNDERPERFORMER)
    - Identificar melhores e piores performers
    - Fornecer contexto de performance geral

    Attributes:
        AGENT_NAME: Nome identificador do subagente ("classification")
        AGENT_DESCRIPTION: Descricao das capacidades do agente
    """

    AGENT_NAME = "classification"
    AGENT_DESCRIPTION = "Analisa e classifica campanhas por tier de performance"

    def get_tools(self) -> List[BaseTool]:
        """Retorna as 4 tools de classificacao.

        Returns:
            Lista com get_classifications, get_campaign_tier,
            get_high_performers, get_underperformers
        """
        return [
            get_classifications,
            get_campaign_tier,
            get_high_performers,
            get_underperformers
        ]

    def get_system_prompt(self) -> str:
        """Retorna o system prompt do ClassificationAgent.

        Returns:
            System prompt string com instrucoes especializadas
        """
        return get_classification_prompt()
