"""AnalysisAgent - Especialista em analises avancadas."""
from typing import List

from langchain_core.tools import BaseTool

from projects.agent.subagents.base import BaseSubagent
from projects.agent.subagents.analysis.prompts import get_analysis_prompt
from projects.agent.tools.analysis_tools import (
    compare_campaigns,
    analyze_trends,
    get_account_summary,
    calculate_roi,
    get_top_campaigns
)


class AnalysisAgent(BaseSubagent):
    """Subagente especializado em analises avancadas.

    Responsavel por:
    - Comparacoes entre campanhas
    - Analise de tendencias
    - Calculos de ROI/ROAS
    - Rankings e sumarios

    Attributes:
        AGENT_NAME: Nome identificador do subagente ("analysis")
        AGENT_DESCRIPTION: Descricao das capacidades do agente
    """

    AGENT_NAME = "analysis"
    AGENT_DESCRIPTION = "Realiza analises avancadas e comparacoes"

    def get_tools(self) -> List[BaseTool]:
        """Retorna as 5 tools de analise.

        Returns:
            Lista com compare_campaigns, analyze_trends, get_account_summary,
            calculate_roi, get_top_campaigns
        """
        return [
            compare_campaigns,
            analyze_trends,
            get_account_summary,
            calculate_roi,
            get_top_campaigns
        ]

    def get_system_prompt(self) -> str:
        """Retorna o system prompt do AnalysisAgent.

        Returns:
            System prompt string
        """
        return get_analysis_prompt()
