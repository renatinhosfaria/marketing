"""
Tools para acesso a previsões de métricas.
"""

from typing import Optional
from langchain_core.tools import tool
from sqlalchemy import select, desc, and_
from datetime import datetime, timedelta

from shared.db.session import get_db_session
from shared.db.models.ml_readonly import MLForecast, PredictionType
from shared.core.logging import get_logger
from projects.agent.tools.base import format_currency, format_number

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
        forecast_type: Tipo de métrica (cpl, leads, spend) ou None para todos
        days_ahead: Dias à frente para considerar (padrão: 7)
        limit: Número máximo de forecasts (padrão: 30)

    Returns:
        Lista de previsões com intervalos de confiança
    """
    try:
        with get_db_session() as db:
            today = datetime.utcnow().date()
            end_date = today + timedelta(days=days_ahead)

            query = select(MLForecast).where(
                MLForecast.config_id == config_id,
                MLForecast.forecast_date >= today,
                MLForecast.forecast_date <= end_date
            )

            # Filtrar por métrica se especificado
            if forecast_type:
                metric = forecast_type.lower().replace("_forecast", "")
                query = query.where(MLForecast.target_metric == metric)

            query = query.order_by(
                MLForecast.forecast_date,
                desc(MLForecast.created_at)
            ).limit(limit)

            results = db.execute(query).scalars().all()

            forecasts = []
            metric_summaries = {}

            for forecast_row in results:
                metric = forecast_row.target_metric

                # Processar predictions JSON
                if forecast_row.predictions:
                    for pred in forecast_row.predictions:
                        pred_value = pred.get("predicted_value", 0)

                        # Acumular para sumário
                        if metric not in metric_summaries:
                            metric_summaries[metric] = {
                                "count": 0,
                                "values": [],
                                "campaigns": set()
                            }
                        metric_summaries[metric]["count"] += 1
                        metric_summaries[metric]["values"].append(pred_value)
                        metric_summaries[metric]["campaigns"].add(forecast_row.entity_id)

                        forecasts.append({
                            "id": forecast_row.id,
                            "campaign_id": forecast_row.entity_id,
                            "type": metric.upper(),
                            "forecast_date": pred.get("date", "")[:10] if pred.get("date") else None,
                            "predicted_value": round(pred_value, 2),
                            "lower_bound": round(pred.get("confidence_lower", 0), 2),
                            "upper_bound": round(pred.get("confidence_upper", 0), 2),
                            "method": forecast_row.method,
                            "model_version": forecast_row.model_version,
                        })

            # Construir sumário por métrica
            summary_parts = []
            for metric, data in metric_summaries.items():
                avg_value = sum(data["values"]) / len(data["values"]) if data["values"] else 0
                if metric == "cpl":
                    summary_parts.append(f"CPL médio previsto: {format_currency(avg_value)}")
                elif metric == "leads":
                    summary_parts.append(f"Leads previstos: {format_number(sum(data['values']))}")
                elif metric == "spend":
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
            today = datetime.utcnow().date()
            end_date = today + timedelta(days=days_ahead)

            forecast_row = db.execute(
                select(MLForecast).where(
                    MLForecast.config_id == config_id,
                    MLForecast.entity_id == campaign_id,
                    MLForecast.target_metric == "cpl",
                    MLForecast.forecast_date >= today,
                    MLForecast.forecast_date <= end_date
                ).order_by(MLForecast.forecast_date)
            ).scalar()

            if not forecast_row or not forecast_row.predictions:
                return {
                    "found": False,
                    "campaign_id": campaign_id,
                    "message": f"Sem previsões de CPL disponíveis para a campanha {campaign_id}."
                }

            predictions = []
            total_cpl = 0

            # Processar cada dia do JSON predictions
            for pred in forecast_row.predictions:
                pred_value = pred.get("predicted_value", 0)
                lower = pred.get("confidence_lower", 0)
                upper = pred.get("confidence_upper", 0)

                predictions.append({
                    "date": pred.get("date", "")[:10] if pred.get("date") else "",
                    "predicted_cpl": format_currency(pred_value),
                    "range": f"{format_currency(lower)} - {format_currency(upper)}" if upper > 0 else "N/A",
                    "raw_value": round(pred_value, 2),
                })
                total_cpl += pred_value

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
            today = datetime.utcnow().date()
            end_date = today + timedelta(days=days_ahead)

            forecast_row = db.execute(
                select(MLForecast).where(
                    MLForecast.config_id == config_id,
                    MLForecast.entity_id == campaign_id,
                    MLForecast.target_metric == "leads",
                    MLForecast.forecast_date >= today,
                    MLForecast.forecast_date <= end_date
                ).order_by(MLForecast.forecast_date)
            ).scalar()

            if not forecast_row or not forecast_row.predictions:
                return {
                    "found": False,
                    "campaign_id": campaign_id,
                    "message": f"Sem previsões de leads disponíveis para a campanha {campaign_id}."
                }

            predictions = []
            total_leads = 0
            total_lower = 0
            total_upper = 0

            # Processar cada dia do JSON predictions
            for pred in forecast_row.predictions:
                leads = int(round(pred.get("predicted_value", 0)))
                lower = int(round(pred.get("confidence_lower", 0)))
                upper = int(round(pred.get("confidence_upper", 0)))

                predictions.append({
                    "date": pred.get("date", "")[:10] if pred.get("date") else "",
                    "predicted_leads": leads,
                    "range": f"{lower} - {upper}" if upper > 0 else "N/A",
                })
                total_leads += leads
                total_lower += lower if lower > 0 else leads
                total_upper += upper if upper > 0 else leads

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
