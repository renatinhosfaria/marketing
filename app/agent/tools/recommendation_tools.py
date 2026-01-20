"""
Tools para acesso a recomendações de otimização.
"""

from typing import Optional
from langchain_core.tools import tool
from sqlalchemy import select, desc, and_
from datetime import datetime, timedelta

from app.db.session import get_db_session
from app.db.models import MLRecommendation, RecommendationType
from app.core.logging import get_logger
from app.agent.tools.base import format_currency

logger = get_logger(__name__)


# Mapeamento de tipos de recomendação para descrições em PT-BR
RECOMMENDATION_DESCRIPTIONS = {
    RecommendationType.SCALE_UP: "Escalar campanha (aumentar budget significativamente)",
    RecommendationType.BUDGET_INCREASE: "Aumentar budget moderadamente",
    RecommendationType.BUDGET_DECREASE: "Reduzir budget",
    RecommendationType.PAUSE_CAMPAIGN: "Pausar campanha",
    RecommendationType.CREATIVE_REFRESH: "Renovar criativos",
    RecommendationType.AUDIENCE_REVIEW: "Revisar segmentação de público",
    RecommendationType.REACTIVATE: "Reativar campanha pausada",
    RecommendationType.OPTIMIZE_SCHEDULE: "Otimizar horários de veiculação",
}


@tool
def get_recommendations(
    config_id: int,
    limit: int = 20,
    active_only: bool = True
) -> dict:
    """
    Lista todas as recomendações de otimização ativas.

    Use esta ferramenta para obter sugestões de ações para melhorar
    a performance das campanhas.

    Args:
        config_id: ID da configuração Facebook Ads
        limit: Número máximo de recomendações (padrão: 20)
        active_only: Se True, retorna apenas recomendações não expiradas

    Returns:
        Lista de recomendações ordenadas por prioridade
    """
    try:
        with get_db_session() as db:
            query = select(MLRecommendation).where(
                MLRecommendation.config_id == config_id
            )

            if active_only:
                query = query.where(
                    MLRecommendation.expires_at > datetime.utcnow()
                )

            query = query.order_by(desc(MLRecommendation.priority)).limit(limit)

            results = db.execute(query).scalars().all()

            recommendations = []
            type_counts = {}

            for r in results:
                rec_type = r.recommendation_type.value
                type_counts[rec_type] = type_counts.get(rec_type, 0) + 1

                recommendations.append({
                    "id": r.id,
                    "campaign_id": r.entity_id,
                    "type": rec_type,
                    "type_description": RECOMMENDATION_DESCRIPTIONS.get(
                        r.recommendation_type, rec_type
                    ),
                    "priority": r.priority,
                    "title": r.title,
                    "description": r.description,
                    "confidence": round(r.confidence_score * 100, 1),
                    "suggested_action": r.suggested_action,
                    "reasoning": r.reasoning,
                    "expires_at": r.expires_at.isoformat() if r.expires_at else None,
                })

            return {
                "total": len(recommendations),
                "by_type": type_counts,
                "recommendations": recommendations,
                "summary": _build_recommendations_summary(type_counts)
            }
    except Exception as e:
        logger.error("Erro ao buscar recomendações", error=str(e))
        return {"error": str(e), "recommendations": []}


@tool
def get_recommendations_by_type(
    config_id: int,
    recommendation_type: str,
    limit: int = 10
) -> dict:
    """
    Filtra recomendações por tipo específico.

    Tipos disponíveis:
    - SCALE_UP: Escalar campanhas de alta performance
    - BUDGET_INCREASE: Aumentar budget moderadamente
    - BUDGET_DECREASE: Reduzir budget de campanhas com CPL alto
    - PAUSE_CAMPAIGN: Pausar campanhas com baixa performance
    - CREATIVE_REFRESH: Renovar criativos (frequência alta)
    - AUDIENCE_REVIEW: Revisar segmentação
    - REACTIVATE: Reativar campanhas pausadas
    - OPTIMIZE_SCHEDULE: Otimizar horários

    Args:
        config_id: ID da configuração Facebook Ads
        recommendation_type: Tipo de recomendação (ex: "SCALE_UP", "PAUSE_CAMPAIGN")
        limit: Número máximo de resultados (padrão: 10)

    Returns:
        Lista de recomendações do tipo especificado
    """
    try:
        # Validar tipo
        try:
            rec_type = RecommendationType(recommendation_type.upper())
        except ValueError:
            return {
                "error": f"Tipo de recomendação inválido: {recommendation_type}",
                "valid_types": [t.value for t in RecommendationType],
                "recommendations": []
            }

        with get_db_session() as db:
            results = db.execute(
                select(MLRecommendation).where(
                    MLRecommendation.config_id == config_id,
                    MLRecommendation.recommendation_type == rec_type,
                    MLRecommendation.expires_at > datetime.utcnow()
                ).order_by(desc(MLRecommendation.priority)).limit(limit)
            ).scalars().all()

            recommendations = []
            for r in results:
                recommendations.append({
                    "id": r.id,
                    "campaign_id": r.entity_id,
                    "priority": r.priority,
                    "title": r.title,
                    "description": r.description,
                    "confidence": round(r.confidence_score * 100, 1),
                    "suggested_action": r.suggested_action,
                })

            type_desc = RECOMMENDATION_DESCRIPTIONS.get(rec_type, recommendation_type)

            return {
                "type": recommendation_type,
                "type_description": type_desc,
                "total": len(recommendations),
                "recommendations": recommendations,
                "summary": f"Encontradas {len(recommendations)} recomendações do tipo '{type_desc}'."
            }
    except Exception as e:
        logger.error("Erro ao buscar recomendações por tipo", error=str(e))
        return {"error": str(e), "recommendations": []}


@tool
def get_high_priority_recommendations(
    config_id: int,
    min_priority: int = 7,
    limit: int = 10
) -> dict:
    """
    Retorna recomendações de alta prioridade que requerem ação urgente.

    Use esta ferramenta para identificar as ações mais importantes
    que devem ser tomadas primeiro.

    Args:
        config_id: ID da configuração Facebook Ads
        min_priority: Prioridade mínima (1-10, padrão: 7)
        limit: Número máximo de resultados (padrão: 10)

    Returns:
        Lista de recomendações urgentes ordenadas por prioridade
    """
    try:
        with get_db_session() as db:
            results = db.execute(
                select(MLRecommendation).where(
                    MLRecommendation.config_id == config_id,
                    MLRecommendation.priority >= min_priority,
                    MLRecommendation.expires_at > datetime.utcnow()
                ).order_by(desc(MLRecommendation.priority)).limit(limit)
            ).scalars().all()

            recommendations = []
            for r in results:
                # Determinar urgência baseada na prioridade
                if r.priority >= 9:
                    urgency = "CRÍTICA"
                    urgency_emoji = "🔴"
                elif r.priority >= 8:
                    urgency = "ALTA"
                    urgency_emoji = "🟠"
                else:
                    urgency = "MÉDIA-ALTA"
                    urgency_emoji = "🟡"

                recommendations.append({
                    "id": r.id,
                    "campaign_id": r.entity_id,
                    "type": r.recommendation_type.value,
                    "priority": r.priority,
                    "urgency": urgency,
                    "urgency_emoji": urgency_emoji,
                    "title": r.title,
                    "description": r.description,
                    "suggested_action": r.suggested_action,
                    "confidence": round(r.confidence_score * 100, 1),
                })

            critical_count = sum(1 for r in recommendations if r["priority"] >= 9)
            high_count = sum(1 for r in recommendations if 8 <= r["priority"] < 9)

            return {
                "total": len(recommendations),
                "critical_count": critical_count,
                "high_count": high_count,
                "recommendations": recommendations,
                "summary": _build_priority_summary(critical_count, high_count, len(recommendations))
            }
    except Exception as e:
        logger.error("Erro ao buscar recomendações prioritárias", error=str(e))
        return {"error": str(e), "recommendations": []}


def _build_recommendations_summary(type_counts: dict) -> str:
    """Constrói resumo das recomendações por tipo."""
    if not type_counts:
        return "Nenhuma recomendação ativa no momento."

    parts = []
    if type_counts.get("SCALE_UP", 0):
        parts.append(f"{type_counts['SCALE_UP']} para escalar")
    if type_counts.get("PAUSE_CAMPAIGN", 0):
        parts.append(f"{type_counts['PAUSE_CAMPAIGN']} para pausar")
    if type_counts.get("BUDGET_DECREASE", 0):
        parts.append(f"{type_counts['BUDGET_DECREASE']} para reduzir budget")
    if type_counts.get("CREATIVE_REFRESH", 0):
        parts.append(f"{type_counts['CREATIVE_REFRESH']} para renovar criativos")

    total = sum(type_counts.values())
    summary = f"Total de {total} recomendações ativas"
    if parts:
        summary += f": {', '.join(parts)}."
    else:
        summary += "."

    return summary


def _build_priority_summary(critical: int, high: int, total: int) -> str:
    """Constrói resumo das recomendações prioritárias."""
    if total == 0:
        return "Nenhuma recomendação de alta prioridade no momento. Suas campanhas estão bem!"

    parts = []
    if critical > 0:
        parts.append(f"🔴 {critical} crítica(s)")
    if high > 0:
        parts.append(f"🟠 {high} de alta prioridade")

    return f"Atenção! {', '.join(parts)} requerem ação imediata."
