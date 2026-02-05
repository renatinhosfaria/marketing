"""Specialized subagents for multi-agent system.

Re-exports from the original subagents/ location.
"""
from projects.agent.subagents.base import BaseSubagent
from projects.agent.subagents.state import SubagentState
from projects.agent.subagents.analysis.agent import AnalysisSubagent
from projects.agent.subagents.anomaly.agent import AnomalySubagent
from projects.agent.subagents.campaign.agent import CampaignSubagent
from projects.agent.subagents.classification.agent import ClassificationSubagent
from projects.agent.subagents.forecast.agent import ForecastSubagent
from projects.agent.subagents.recommendation.agent import RecommendationSubagent

__all__ = [
    "BaseSubagent",
    "SubagentState",
    "AnalysisSubagent",
    "AnomalySubagent",
    "CampaignSubagent",
    "ClassificationSubagent",
    "ForecastSubagent",
    "RecommendationSubagent",
]
