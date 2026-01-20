"""RecommendationAgent - Recomendacoes de otimizacao de campanhas."""
from app.agent.subagents.recommendation.agent import RecommendationAgent
from app.agent.subagents.recommendation.prompts import get_recommendation_prompt

__all__ = ["RecommendationAgent", "get_recommendation_prompt"]
