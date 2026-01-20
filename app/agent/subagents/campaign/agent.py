"""CampaignAgent - Especialista em dados de campanhas."""
from typing import List

from langchain_core.tools import BaseTool

from app.agent.subagents.base import BaseSubagent
from app.agent.subagents.campaign.prompts import get_campaign_prompt
from app.agent.tools.campaign_tools import (
    get_campaign_details,
    list_campaigns
)


class CampaignAgent(BaseSubagent):
    """Subagente especializado em dados de campanhas.

    Responsavel por:
    - Fornecer detalhes de campanhas especificas
    - Listar campanhas com filtros
    - Apresentar metricas formatadas

    Attributes:
        AGENT_NAME: Nome identificador do subagente ("campaign")
        AGENT_DESCRIPTION: Descricao das capacidades do agente
    """

    AGENT_NAME = "campaign"
    AGENT_DESCRIPTION = "Fornece dados detalhados de campanhas"

    def get_tools(self) -> List[BaseTool]:
        """Retorna as 2 tools de campanha.

        Returns:
            Lista com get_campaign_details, list_campaigns
        """
        return [
            get_campaign_details,
            list_campaigns
        ]

    def get_system_prompt(self) -> str:
        """Retorna o system prompt do CampaignAgent.

        Returns:
            System prompt string
        """
        return get_campaign_prompt()
