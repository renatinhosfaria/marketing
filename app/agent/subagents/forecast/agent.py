"""ForecastAgent - Especialista em previsoes."""
from typing import List

from langchain_core.tools import BaseTool

from app.agent.subagents.base import BaseSubagent
from app.agent.subagents.forecast.prompts import get_forecast_prompt
from app.agent.tools.forecast_tools import (
    get_forecasts,
    predict_campaign_cpl,
    predict_campaign_leads
)


class ForecastAgent(BaseSubagent):
    """Subagente especializado em previsoes.

    Responsavel por:
    - Analisar previsoes de CPL e Leads
    - Identificar tendencias futuras
    - Alertar sobre mudancas esperadas

    Attributes:
        AGENT_NAME: Nome identificador do subagente ("forecast")
        AGENT_DESCRIPTION: Descricao das capacidades do agente
    """

    AGENT_NAME = "forecast"
    AGENT_DESCRIPTION = "Analisa previsoes de CPL e Leads"

    def get_tools(self) -> List[BaseTool]:
        """Retorna as 3 tools de previsao.

        Returns:
            Lista com get_forecasts, predict_campaign_cpl,
            predict_campaign_leads
        """
        return [
            get_forecasts,
            predict_campaign_cpl,
            predict_campaign_leads
        ]

    def get_system_prompt(self) -> str:
        """Retorna o system prompt do ForecastAgent.

        Returns:
            System prompt string
        """
        return get_forecast_prompt()
