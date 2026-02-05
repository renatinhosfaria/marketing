"""RecommendationAgent - Recomendacoes de otimizacao de campanhas."""
from projects.agent.subagents.recommendation.agent import RecommendationAgent
from projects.agent.subagents.recommendation.prompts import get_recommendation_prompt

__all__ = ["RecommendationAgent", "get_recommendation_prompt"]
