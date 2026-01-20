"""
Tools para acesso a anomalias detectadas.
"""

from typing import Optional
from langchain_core.tools import tool
from sqlalchemy import select, desc, and_
from datetime import datetime, timedelta

from app.db.session import get_db_session
from app.db.models import MLAnomaly, AnomalySeverity
from app.core.logging import get_logger
from app.agent.tools.base import format_currency, format_percentage

logger = get_logger(__name__)


# Mapeamento de tipos de anomalia para descrições em PT-BR
ANOMALY_DESCRIPTIONS = {
    "CPL_SPIKE": "Aumento súbito no CPL",
    "CPL_DROP": "Queda súbita no CPL",
    "SPEND_SPIKE": "Aumento súbito no gasto",
    "SPEND_DROP": "Gasto zerado ou muito baixo",
    "LEADS_SPIKE": "Aumento súbito nos leads",
    "LEADS_DROP": "Queda súbita nos leads",
    "CTR_DROP": "Queda no CTR",
    "FREQUENCY_HIGH": "Frequência muito alta",
    "PERFORMANCE_DROP": "Queda geral na performance",
    "BUDGET_EXHAUSTED": "Budget esgotado",
    "ZERO_IMPRESSIONS": "Sem impressões",
    "CONVERSION_ANOMALY": "Anomalia na conversão",
}

# Emojis por severidade
SEVERITY_EMOJIS = {
    AnomalySeverity.CRITICAL: "🔴",
    AnomalySeverity.HIGH: "🟠",
    AnomalySeverity.MEDIUM: "🟡",
    AnomalySeverity.LOW: "🟢",
}


@tool
def get_anomalies(
    config_id: int,
    days: int = 7,
    limit: int = 50
) -> dict:
    """
    Lista anomalias detectadas nas campanhas.

    Use esta ferramenta para identificar problemas ou comportamentos
    incomuns nas campanhas que precisam de atenção.

    Args:
        config_id: ID da configuração Facebook Ads
        days: Número de dias para buscar (padrão: 7)
        limit: Número máximo de anomalias (padrão: 50)

    Returns:
        Lista de anomalias com detalhes e severidade
    """
    try:
        since = datetime.utcnow() - timedelta(days=days)

        with get_db_session() as db:
            results = db.execute(
                select(MLAnomaly).where(
                    MLAnomaly.config_id == config_id,
                    MLAnomaly.detected_at >= since
                ).order_by(
                    desc(MLAnomaly.severity),
                    desc(MLAnomaly.detected_at)
                ).limit(limit)
            ).scalars().all()

            anomalies = []
            severity_counts = {s.value: 0 for s in AnomalySeverity}
            type_counts = {}

            for a in results:
                severity_counts[a.severity.value] += 1
                anomaly_type = a.anomaly_type
                type_counts[anomaly_type] = type_counts.get(anomaly_type, 0) + 1

                anomalies.append({
                    "id": a.id,
                    "campaign_id": a.entity_id,
                    "type": anomaly_type,
                    "type_description": ANOMALY_DESCRIPTIONS.get(anomaly_type, anomaly_type),
                    "severity": a.severity.value,
                    "severity_emoji": SEVERITY_EMOJIS.get(a.severity, "⚪"),
                    "description": a.description,
                    "metric_name": a.metric_name,
                    "metric_value": a.metric_value,
                    "expected_value": a.expected_value,
                    "deviation": round(a.deviation_score, 2) if a.deviation_score else None,
                    "detected_at": a.detected_at.isoformat() if a.detected_at else None,
                    "is_resolved": a.is_resolved,
                })

            return {
                "total": len(anomalies),
                "period_days": days,
                "by_severity": severity_counts,
                "by_type": type_counts,
                "anomalies": anomalies,
                "summary": _build_anomaly_summary(severity_counts, days)
            }
    except Exception as e:
        logger.error("Erro ao buscar anomalias", error=str(e))
        return {"error": str(e), "anomalies": []}


@tool
def get_critical_anomalies(
    config_id: int,
    days: int = 3,
    limit: int = 20
) -> dict:
    """
    Retorna anomalias de severidade CRITICAL e HIGH que requerem ação imediata.

    Use esta ferramenta para identificar problemas graves que podem
    estar causando desperdício de budget ou perda de leads.

    Args:
        config_id: ID da configuração Facebook Ads
        days: Número de dias para buscar (padrão: 3)
        limit: Número máximo de anomalias (padrão: 20)

    Returns:
        Lista de anomalias críticas ordenadas por severidade
    """
    try:
        since = datetime.utcnow() - timedelta(days=days)

        with get_db_session() as db:
            results = db.execute(
                select(MLAnomaly).where(
                    MLAnomaly.config_id == config_id,
                    MLAnomaly.severity.in_([AnomalySeverity.CRITICAL, AnomalySeverity.HIGH]),
                    MLAnomaly.detected_at >= since,
                    MLAnomaly.is_resolved == False
                ).order_by(
                    desc(MLAnomaly.severity),
                    desc(MLAnomaly.detected_at)
                ).limit(limit)
            ).scalars().all()

            anomalies = []
            critical_count = 0
            high_count = 0

            for a in results:
                if a.severity == AnomalySeverity.CRITICAL:
                    critical_count += 1
                else:
                    high_count += 1

                # Construir ação sugerida
                suggested_action = _get_suggested_action(a.anomaly_type, a.severity)

                anomalies.append({
                    "id": a.id,
                    "campaign_id": a.entity_id,
                    "type": a.anomaly_type,
                    "type_description": ANOMALY_DESCRIPTIONS.get(a.anomaly_type, a.anomaly_type),
                    "severity": a.severity.value,
                    "severity_emoji": SEVERITY_EMOJIS.get(a.severity, "⚪"),
                    "description": a.description,
                    "metric_name": a.metric_name,
                    "current_value": a.metric_value,
                    "expected_value": a.expected_value,
                    "deviation_percent": round(abs(a.deviation_score) * 100, 1) if a.deviation_score else None,
                    "detected_at": a.detected_at.isoformat() if a.detected_at else None,
                    "suggested_action": suggested_action,
                })

            if not anomalies:
                return {
                    "total": 0,
                    "critical_count": 0,
                    "high_count": 0,
                    "anomalies": [],
                    "summary": "Nenhuma anomalia crítica ou de alta severidade nos últimos " +
                              f"{days} dias. Suas campanhas estão saudáveis!"
                }

            return {
                "total": len(anomalies),
                "critical_count": critical_count,
                "high_count": high_count,
                "anomalies": anomalies,
                "summary": f"⚠️ ATENÇÃO: {critical_count} anomalia(s) CRÍTICA(S) e {high_count} de "
                          f"ALTA severidade detectadas nos últimos {days} dias. Ação imediata recomendada."
            }
    except Exception as e:
        logger.error("Erro ao buscar anomalias críticas", error=str(e))
        return {"error": str(e), "anomalies": []}


@tool
def get_anomalies_by_type(
    config_id: int,
    anomaly_type: str,
    days: int = 7,
    limit: int = 20
) -> dict:
    """
    Filtra anomalias por tipo específico.

    Tipos disponíveis:
    - CPL_SPIKE: Aumento súbito no CPL
    - CPL_DROP: Queda no CPL
    - SPEND_SPIKE: Aumento no gasto
    - SPEND_DROP: Gasto zerado
    - LEADS_DROP: Queda nos leads
    - CTR_DROP: Queda no CTR
    - FREQUENCY_HIGH: Frequência alta
    - PERFORMANCE_DROP: Queda geral na performance

    Args:
        config_id: ID da configuração Facebook Ads
        anomaly_type: Tipo de anomalia (ex: "CPL_SPIKE", "SPEND_DROP")
        days: Número de dias para buscar (padrão: 7)
        limit: Número máximo de resultados (padrão: 20)

    Returns:
        Lista de anomalias do tipo especificado
    """
    try:
        since = datetime.utcnow() - timedelta(days=days)

        with get_db_session() as db:
            results = db.execute(
                select(MLAnomaly).where(
                    MLAnomaly.config_id == config_id,
                    MLAnomaly.anomaly_type == anomaly_type.upper(),
                    MLAnomaly.detected_at >= since
                ).order_by(
                    desc(MLAnomaly.severity),
                    desc(MLAnomaly.detected_at)
                ).limit(limit)
            ).scalars().all()

            anomalies = []
            for a in results:
                anomalies.append({
                    "id": a.id,
                    "campaign_id": a.entity_id,
                    "severity": a.severity.value,
                    "severity_emoji": SEVERITY_EMOJIS.get(a.severity, "⚪"),
                    "description": a.description,
                    "metric_value": a.metric_value,
                    "expected_value": a.expected_value,
                    "detected_at": a.detected_at.isoformat() if a.detected_at else None,
                    "is_resolved": a.is_resolved,
                })

            type_desc = ANOMALY_DESCRIPTIONS.get(anomaly_type.upper(), anomaly_type)

            return {
                "type": anomaly_type.upper(),
                "type_description": type_desc,
                "total": len(anomalies),
                "period_days": days,
                "anomalies": anomalies,
                "summary": f"Encontradas {len(anomalies)} anomalias do tipo '{type_desc}' nos últimos {days} dias."
            }
    except Exception as e:
        logger.error("Erro ao buscar anomalias por tipo", error=str(e))
        return {"error": str(e), "anomalies": []}


def _build_anomaly_summary(severity_counts: dict, days: int) -> str:
    """Constrói resumo das anomalias."""
    critical = severity_counts.get("CRITICAL", 0)
    high = severity_counts.get("HIGH", 0)
    medium = severity_counts.get("MEDIUM", 0)
    low = severity_counts.get("LOW", 0)
    total = critical + high + medium + low

    if total == 0:
        return f"Nenhuma anomalia detectada nos últimos {days} dias. Excelente!"

    parts = []
    if critical > 0:
        parts.append(f"🔴 {critical} crítica(s)")
    if high > 0:
        parts.append(f"🟠 {high} alta(s)")
    if medium > 0:
        parts.append(f"🟡 {medium} média(s)")
    if low > 0:
        parts.append(f"🟢 {low} baixa(s)")

    return f"Total de {total} anomalias nos últimos {days} dias: {', '.join(parts)}."


def _get_suggested_action(anomaly_type: str, severity: AnomalySeverity) -> str:
    """Retorna ação sugerida baseada no tipo de anomalia."""
    actions = {
        "CPL_SPIKE": "Verificar segmentação e criativos. Considere pausar se CPL continuar alto.",
        "SPEND_DROP": "Verificar status da campanha, pagamento e aprovação de anúncios.",
        "LEADS_DROP": "Analisar funil de conversão e landing page. Verificar formulário de lead.",
        "CTR_DROP": "Renovar criativos e testar novos copies. Público pode estar saturado.",
        "FREQUENCY_HIGH": "Expandir público ou pausar temporariamente para evitar fadiga.",
        "PERFORMANCE_DROP": "Revisar todos os elementos da campanha. Considere recriar do zero.",
        "ZERO_IMPRESSIONS": "Verificar budget, bid e status de aprovação dos anúncios.",
    }

    default_action = "Investigar a causa e tomar ação corretiva."

    if severity == AnomalySeverity.CRITICAL:
        return f"URGENTE: {actions.get(anomaly_type, default_action)}"

    return actions.get(anomaly_type, default_action)
