"""
Tools para acesso a previsões de métricas.
"""

from typing import Optional
from langchain_core.tools import tool
from sqlalchemy import select, desc, and_
from datetime import datetime, timedelta

from app.db.session import get_db_session
from app.db.models import MLForecast, PredictionType
from app.core.logging import get_logger
from app.agent.tools.base import format_currency, format_number

logger = get_logger(__name__)


@tool
def get_forecasts(
    config_id: int,
    forecast_type: Optional[str] = None,
    days_ahead: int = 7,
    limit: int = 30
) -> dict:
    """
    Lista previsões de métricas para as campanhas.

    Use esta ferramenta para obter previsões de CPL, leads ou spend
    para os próximos dias.

    Args:
        config_id: ID da configuração Facebook Ads
        forecast_type: Tipo de previsão (CPL_FORECAST, LEADS_FORECAST, SPEND_FORECAST) ou None para todos
        days_ahead: Dias à frente para considerar (padrão: 7)
        limit: Número máximo de previsões (padrão: 30)

    Returns:
        Lista de previsões com intervalos de confiança
    """
    try:
        with get_db_session() as db:
            query = select(MLForecast).where(
                MLForecast.config_id == config_id,
                MLForecast.forecast_date <= datetime.utcnow() + timedelta(days=days_ahead),
                MLForecast.forecast_date >= datetime.utcnow()
            )

            if forecast_type:
                try:
                    pred_type = PredictionType(forecast_type.upper())
                    query = query.where(MLForecast.prediction_type == pred_type)
                except ValueError:
                    return {
                        "error": f"Tipo de previsão inválido: {forecast_type}",
                        "valid_types": ["CPL_FORECAST", "LEADS_FORECAST", "SPEND_FORECAST"],
                        "forecasts": []
                    }

            query = query.order_by(
                MLForecast.forecast_date,
                desc(MLForecast.created_at)
            ).limit(limit)

            results = db.execute(query).scalars().all()

            forecasts = []
            type_summaries = {}

            for f in results:
                pred_type = f.prediction_type.value

                # Acumular para sumário
                if pred_type not in type_summaries:
                    type_summaries[pred_type] = {
                        "count": 0,
                        "values": [],
                        "campaigns": set()
                    }
                type_summaries[pred_type]["count"] += 1
                type_summaries[pred_type]["values"].append(f.predicted_value)
                type_summaries[pred_type]["campaigns"].add(f.entity_id)

                forecasts.append({
                    "id": f.id,
                    "campaign_id": f.entity_id,
                    "type": pred_type,
                    "forecast_date": f.forecast_date.strftime("%Y-%m-%d") if f.forecast_date else None,
                    "predicted_value": round(f.predicted_value, 2),
                    "lower_bound": round(f.lower_bound, 2) if f.lower_bound else None,
                    "upper_bound": round(f.upper_bound, 2) if f.upper_bound else None,
                    "confidence": round(f.confidence_interval * 100, 1) if f.confidence_interval else 95,
                    "model_version": f.model_version,
                })

            # Construir sumário por tipo
            summary_parts = []
            for pred_type, data in type_summaries.items():
                avg_value = sum(data["values"]) / len(data["values"]) if data["values"] else 0
                if pred_type == "CPL_FORECAST":
                    summary_parts.append(f"CPL médio previsto: {format_currency(avg_value)}")
                elif pred_type == "LEADS_FORECAST":
                    summary_parts.append(f"Leads previstos: {format_number(sum(data['values']))}")
                elif pred_type == "SPEND_FORECAST":
                    summary_parts.append(f"Spend previsto: {format_currency(sum(data['values']))}")

            return {
                "total": len(forecasts),
                "days_ahead": days_ahead,
                "forecasts": forecasts,
                "summary": "; ".join(summary_parts) if summary_parts else "Sem previsões disponíveis."
            }
    except Exception as e:
        logger.error("Erro ao buscar previsões", error=str(e))
        return {"error": str(e), "forecasts": []}


@tool
def predict_campaign_cpl(
    config_id: int,
    campaign_id: str,
    days_ahead: int = 7
) -> dict:
    """
    Retorna previsão de CPL para uma campanha específica.

    Use esta ferramenta para saber qual será o CPL estimado
    de uma campanha nos próximos dias.

    Args:
        config_id: ID da configuração Facebook Ads
        campaign_id: ID da campanha
        days_ahead: Dias à frente para prever (padrão: 7)

    Returns:
        Previsões de CPL dia a dia com intervalos de confiança
    """
    try:
        with get_db_session() as db:
            results = db.execute(
                select(MLForecast).where(
                    MLForecast.config_id == config_id,
                    MLForecast.entity_id == campaign_id,
                    MLForecast.prediction_type == PredictionType.CPL_FORECAST,
                    MLForecast.forecast_date >= datetime.utcnow(),
                    MLForecast.forecast_date <= datetime.utcnow() + timedelta(days=days_ahead)
                ).order_by(MLForecast.forecast_date)
            ).scalars().all()

            if not results:
                return {
                    "found": False,
                    "campaign_id": campaign_id,
                    "message": f"Sem previsões de CPL disponíveis para a campanha {campaign_id}."
                }

            predictions = []
            total_cpl = 0

            for f in results:
                predictions.append({
                    "date": f.forecast_date.strftime("%Y-%m-%d"),
                    "predicted_cpl": format_currency(f.predicted_value),
                    "range": f"{format_currency(f.lower_bound)} - {format_currency(f.upper_bound)}"
                    if f.lower_bound and f.upper_bound else "N/A",
                    "raw_value": round(f.predicted_value, 2),
                })
                total_cpl += f.predicted_value

            avg_cpl = total_cpl / len(predictions) if predictions else 0
            trend = "estável"
            if len(predictions) >= 2:
                first_val = predictions[0]["raw_value"]
                last_val = predictions[-1]["raw_value"]
                if last_val > first_val * 1.1:
                    trend = "subindo 📈"
                elif last_val < first_val * 0.9:
                    trend = "caindo 📉"

            return {
                "found": True,
                "campaign_id": campaign_id,
                "days_ahead": days_ahead,
                "predictions": predictions,
                "average_cpl": format_currency(avg_cpl),
                "trend": trend,
                "summary": f"CPL médio previsto para os próximos {days_ahead} dias: {format_currency(avg_cpl)} ({trend})."
            }
    except Exception as e:
        logger.error("Erro ao prever CPL", error=str(e), campaign_id=campaign_id)
        return {"error": str(e), "found": False}


@tool
def predict_campaign_leads(
    config_id: int,
    campaign_id: str,
    days_ahead: int = 7
) -> dict:
    """
    Retorna previsão de leads para uma campanha específica.

    Use esta ferramenta para estimar quantos leads uma campanha
    deve gerar nos próximos dias.

    Args:
        config_id: ID da configuração Facebook Ads
        campaign_id: ID da campanha
        days_ahead: Dias à frente para prever (padrão: 7)

    Returns:
        Previsões de leads dia a dia com total estimado
    """
    try:
        with get_db_session() as db:
            results = db.execute(
                select(MLForecast).where(
                    MLForecast.config_id == config_id,
                    MLForecast.entity_id == campaign_id,
                    MLForecast.prediction_type == PredictionType.LEADS_FORECAST,
                    MLForecast.forecast_date >= datetime.utcnow(),
                    MLForecast.forecast_date <= datetime.utcnow() + timedelta(days=days_ahead)
                ).order_by(MLForecast.forecast_date)
            ).scalars().all()

            if not results:
                return {
                    "found": False,
                    "campaign_id": campaign_id,
                    "message": f"Sem previsões de leads disponíveis para a campanha {campaign_id}."
                }

            predictions = []
            total_leads = 0
            total_lower = 0
            total_upper = 0

            for f in results:
                leads = int(round(f.predicted_value))
                predictions.append({
                    "date": f.forecast_date.strftime("%Y-%m-%d"),
                    "predicted_leads": leads,
                    "range": f"{int(f.lower_bound)} - {int(f.upper_bound)}"
                    if f.lower_bound and f.upper_bound else "N/A",
                })
                total_leads += leads
                total_lower += int(f.lower_bound) if f.lower_bound else leads
                total_upper += int(f.upper_bound) if f.upper_bound else leads

            return {
                "found": True,
                "campaign_id": campaign_id,
                "days_ahead": days_ahead,
                "predictions": predictions,
                "total_predicted": total_leads,
                "total_range": f"{total_lower} - {total_upper}",
                "daily_average": round(total_leads / len(predictions), 1) if predictions else 0,
                "summary": f"Previsão de {total_leads} leads nos próximos {days_ahead} dias "
                          f"(média de {round(total_leads / days_ahead, 1)} leads/dia)."
            }
    except Exception as e:
        logger.error("Erro ao prever leads", error=str(e), campaign_id=campaign_id)
        return {"error": str(e), "found": False}
