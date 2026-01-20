"""CampaignAgent - Dados detalhados de campanhas."""
from app.agent.subagents.campaign.agent import CampaignAgent
from app.agent.subagents.campaign.prompts import get_campaign_prompt

__all__ = ["CampaignAgent", "get_campaign_prompt"]
