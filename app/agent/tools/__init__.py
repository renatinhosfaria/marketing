"""
Tools do agente para acesso aos dados de ML.
"""

from app.agent.tools.classification_tools import (
    get_classifications,
    get_campaign_tier,
    get_high_performers,
    get_underperformers,
)
from app.agent.tools.recommendation_tools import (
    get_recommendations,
    get_recommendations_by_type,
    get_high_priority_recommendations,
)
from app.agent.tools.anomaly_tools import (
    get_anomalies,
    get_critical_anomalies,
    get_anomalies_by_type,
)
from app.agent.tools.forecast_tools import (
    get_forecasts,
    predict_campaign_cpl,
    predict_campaign_leads,
)
from app.agent.tools.campaign_tools import (
    get_campaign_details,
    list_campaigns,
)
from app.agent.tools.analysis_tools import (
    compare_campaigns,
    analyze_trends,
    get_account_summary,
    calculate_roi,
    get_top_campaigns,
)

# Lista de todas as tools disponíveis
ALL_TOOLS = [
    # Classificação
    get_classifications,
    get_campaign_tier,
    get_high_performers,
    get_underperformers,
    # Recomendações
    get_recommendations,
    get_recommendations_by_type,
    get_high_priority_recommendations,
    # Anomalias
    get_anomalies,
    get_critical_anomalies,
    get_anomalies_by_type,
    # Previsões
    get_forecasts,
    predict_campaign_cpl,
    predict_campaign_leads,
    # Campanhas
    get_campaign_details,
    list_campaigns,
    # Análise
    compare_campaigns,
    analyze_trends,
    get_account_summary,
    calculate_roi,
    get_top_campaigns,
]

def get_all_tools() -> list:
    """
    Retorna a lista completa de tools do agente.
    """
    return ALL_TOOLS

__all__ = [
    "ALL_TOOLS",
    "get_all_tools",
    # Classificação
    "get_classifications",
    "get_campaign_tier",
    "get_high_performers",
    "get_underperformers",
    # Recomendações
    "get_recommendations",
    "get_recommendations_by_type",
    "get_high_priority_recommendations",
    # Anomalias
    "get_anomalies",
    "get_critical_anomalies",
    "get_anomalies_by_type",
    # Previsões
    "get_forecasts",
    "predict_campaign_cpl",
    "predict_campaign_leads",
    # Campanhas
    "get_campaign_details",
    "list_campaigns",
    # Análise
    "compare_campaigns",
    "analyze_trends",
    "get_account_summary",
    "calculate_roi",
    "get_top_campaigns",
]
