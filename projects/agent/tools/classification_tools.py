"""
Tools para acesso a classificações de campanhas.
"""

from datetime import datetime
from typing import Optional
from langchain_core.tools import tool
from sqlalchemy import select, desc, or_
from sqlalchemy.orm import Session

from shared.db.session import get_db_session
from shared.db.models.ml_readonly import MLCampaignClassification, CampaignTier
from shared.core.logging import get_logger
from projects.agent.tools.base import format_currency, format_percentage

logger = get_logger(__name__)


def _is_classification_active(classification: MLCampaignClassification) -> bool:
    """Verifica se a classificação ainda é válida."""
    if classification.valid_until is None:
        return True
    return classification.valid_until > datetime.utcnow()


def _get_metric_from_snapshot(snapshot: dict, key: str, default=None):
    """Extrai métrica do snapshot de métricas."""
    if not snapshot:
        return default
    return snapshot.get(key, default)


@tool
def get_classifications(
    config_id: int,
    limit: int = 50,
    active_only: bool = True
) -> dict:
    """
    Lista todas as classificações de campanhas.

    Use esta ferramenta para obter uma visão geral de como as campanhas
    estão classificadas (HIGH_PERFORMER, MODERATE, LOW, UNDERPERFORMER).

    Args:
        config_id: ID da configuração Facebook Ads
        limit: Número máximo de classificações a retornar (padrão: 50)
        active_only: Se True, retorna apenas campanhas com classificação válida

    Returns:
        Dicionário com lista de classificações e estatísticas
    """
    try:
        with get_db_session() as db:
            query = select(MLCampaignClassification).where(
                MLCampaignClassification.config_id == config_id
            )

            if active_only:
                # Filtrar por valid_until (NULL ou futuro)
                query = query.where(
                    or_(
                        MLCampaignClassification.valid_until.is_(None),
                        MLCampaignClassification.valid_until > datetime.utcnow()
                    )
                )

            query = query.order_by(desc(MLCampaignClassification.classified_at)).limit(limit)

            results = db.execute(query).scalars().all()

            classifications = []
            tier_counts = {tier.value: 0 for tier in CampaignTier}

            for c in results:
                tier_counts[c.tier.value] += 1
                metrics = c.metrics_snapshot or {}
                classifications.append({
                    "campaign_id": c.campaign_id,
                    "campaign_name": _get_metric_from_snapshot(metrics, "campaign_name", c.campaign_id),
                    "tier": c.tier.value,
                    "confidence": round(c.confidence_score * 100, 1),
                    "cpl_7d": round(metrics.get("cpl_7d", 0), 2) if metrics.get("cpl_7d") else None,
                    "leads_7d": metrics.get("leads_7d"),
                    "spend_7d": round(metrics.get("spend_7d", 0), 2) if metrics.get("spend_7d") else None,
                    "is_valid": _is_classification_active(c),
                    "classified_at": c.classified_at.isoformat() if c.classified_at else None,
                })

            return {
                "total": len(classifications),
                "by_tier": tier_counts,
                "classifications": classifications,
                "summary": f"Total de {len(classifications)} campanhas classificadas: "
                          f"{tier_counts.get('HIGH_PERFORMER', 0)} high performers, "
                          f"{tier_counts.get('MODERATE', 0)} moderadas, "
                          f"{tier_counts.get('LOW', 0)} baixas, "
                          f"{tier_counts.get('UNDERPERFORMER', 0)} underperformers."
            }
    except Exception as e:
        logger.error("Erro ao buscar classificações", error=str(e))
        return {"error": str(e), "classifications": []}


@tool
def get_campaign_tier(
    config_id: int,
    campaign_id: str
) -> dict:
    """
    Retorna o tier de classificação de uma campanha específica.

    Use esta ferramenta quando precisar saber a classificação de uma campanha
    em particular, incluindo métricas detalhadas.

    Args:
        config_id: ID da configuração Facebook Ads
        campaign_id: ID da campanha

    Returns:
        Dicionário com detalhes da classificação da campanha
    """
    try:
        with get_db_session() as db:
            result = db.execute(
                select(MLCampaignClassification).where(
                    MLCampaignClassification.config_id == config_id,
                    MLCampaignClassification.campaign_id == campaign_id
                ).order_by(desc(MLCampaignClassification.classified_at)).limit(1)
            ).scalar_one_or_none()

            if not result:
                return {
                    "found": False,
                    "message": f"Campanha {campaign_id} não encontrada nas classificações."
                }

            metrics = result.metrics_snapshot or {}
            features = result.feature_importances or {}

            return {
                "found": True,
                "campaign_id": result.campaign_id,
                "campaign_name": _get_metric_from_snapshot(metrics, "campaign_name", result.campaign_id),
                "tier": result.tier.value,
                "confidence": round(result.confidence_score * 100, 1),
                "metrics": {
                    "cpl_7d": format_currency(metrics.get("cpl_7d")) if metrics.get("cpl_7d") else "N/A",
                    "leads_7d": metrics.get("leads_7d"),
                    "spend_7d": format_currency(metrics.get("spend_7d")) if metrics.get("spend_7d") else "N/A",
                    "ctr_7d": format_percentage(metrics.get("ctr_7d")) if metrics.get("ctr_7d") else "N/A",
                },
                "features": features,
                "is_valid": _is_classification_active(result),
                "classified_at": result.classified_at.isoformat() if result.classified_at else None,
            }
    except Exception as e:
        logger.error("Erro ao buscar tier da campanha", error=str(e), campaign_id=campaign_id)
        return {"error": str(e), "found": False}


@tool
def get_high_performers(
    config_id: int,
    limit: int = 10
) -> dict:
    """
    Lista as campanhas classificadas como HIGH_PERFORMER.

    Use esta ferramenta para identificar as melhores campanhas que podem
    ser escaladas ou usadas como referência.

    Args:
        config_id: ID da configuração Facebook Ads
        limit: Número máximo de campanhas a retornar (padrão: 10)

    Returns:
        Lista de campanhas high performer com métricas
    """
    try:
        with get_db_session() as db:
            results = db.execute(
                select(MLCampaignClassification).where(
                    MLCampaignClassification.config_id == config_id,
                    MLCampaignClassification.tier == CampaignTier.HIGH_PERFORMER,
                    or_(
                        MLCampaignClassification.valid_until.is_(None),
                        MLCampaignClassification.valid_until > datetime.utcnow()
                    )
                ).order_by(
                    desc(MLCampaignClassification.confidence_score)
                ).limit(limit)
            ).scalars().all()

            campaigns = []
            for c in results:
                metrics = c.metrics_snapshot or {}
                cpl_7d = metrics.get("cpl_7d")
                campaigns.append({
                    "campaign_id": c.campaign_id,
                    "campaign_name": _get_metric_from_snapshot(metrics, "campaign_name", c.campaign_id),
                    "confidence": round(c.confidence_score * 100, 1),
                    "cpl_7d": format_currency(cpl_7d) if cpl_7d else "N/A",
                    "leads_7d": metrics.get("leads_7d"),
                    "spend_7d": format_currency(metrics.get("spend_7d")) if metrics.get("spend_7d") else "N/A",
                    "reason": f"CPL {format_percentage((1 - cpl_7d/50) * 100) if cpl_7d and cpl_7d < 50 else 'competitivo'}"
                })

            return {
                "total": len(campaigns),
                "campaigns": campaigns,
                "summary": f"Encontradas {len(campaigns)} campanhas HIGH_PERFORMER ativas." if campaigns
                          else "Nenhuma campanha HIGH_PERFORMER encontrada."
            }
    except Exception as e:
        logger.error("Erro ao buscar high performers", error=str(e))
        return {"error": str(e), "campaigns": []}


@tool
def get_underperformers(
    config_id: int,
    limit: int = 10
) -> dict:
    """
    Lista as campanhas classificadas como UNDERPERFORMER.

    Use esta ferramenta para identificar campanhas com baixa performance
    que precisam de atenção ou podem ser pausadas.

    Args:
        config_id: ID da configuração Facebook Ads
        limit: Número máximo de campanhas a retornar (padrão: 10)

    Returns:
        Lista de campanhas underperformer com métricas e possíveis causas
    """
    try:
        with get_db_session() as db:
            results = db.execute(
                select(MLCampaignClassification).where(
                    MLCampaignClassification.config_id == config_id,
                    MLCampaignClassification.tier == CampaignTier.UNDERPERFORMER,
                    or_(
                        MLCampaignClassification.valid_until.is_(None),
                        MLCampaignClassification.valid_until > datetime.utcnow()
                    )
                ).order_by(
                    desc(MLCampaignClassification.classified_at)
                ).limit(limit)
            ).scalars().all()

            campaigns = []
            for c in results:
                metrics = c.metrics_snapshot or {}
                cpl_7d = metrics.get("cpl_7d")
                leads_7d = metrics.get("leads_7d")
                ctr_7d = metrics.get("ctr_7d")

                # Identificar possíveis causas
                causes = []
                if cpl_7d and cpl_7d > 100:
                    causes.append(f"CPL muito alto ({format_currency(cpl_7d)})")
                if leads_7d is not None and leads_7d == 0:
                    causes.append("Sem leads nos últimos 7 dias")
                if ctr_7d and ctr_7d < 0.5:
                    causes.append(f"CTR muito baixo ({format_percentage(ctr_7d)})")

                campaigns.append({
                    "campaign_id": c.campaign_id,
                    "campaign_name": _get_metric_from_snapshot(metrics, "campaign_name", c.campaign_id),
                    "confidence": round(c.confidence_score * 100, 1),
                    "cpl_7d": format_currency(cpl_7d) if cpl_7d else "N/A",
                    "leads_7d": leads_7d,
                    "spend_7d": format_currency(metrics.get("spend_7d")) if metrics.get("spend_7d") else "N/A",
                    "possible_causes": causes if causes else ["Performance geral abaixo do esperado"],
                })

            return {
                "total": len(campaigns),
                "campaigns": campaigns,
                "summary": f"Encontradas {len(campaigns)} campanhas UNDERPERFORMER que precisam de atenção." if campaigns
                          else "Nenhuma campanha UNDERPERFORMER encontrada. Bom trabalho!",
                "recommendation": "Considere pausar ou otimizar estas campanhas para melhorar o ROI geral."
            }
    except Exception as e:
        logger.error("Erro ao buscar underperformers", error=str(e))
        return {"error": str(e), "campaigns": []}
