"""
Task Celery para calcular impacto de acoes do Agent.

calculate_action_impact: busca acoes executadas ha 7+ dias e calcula
impacto percentual comparando metricas antes/depois.

Agendado: diariamente as 06:00 AM via Celery Beat.
"""

import json
from datetime import datetime, timezone, timedelta

from sqlalchemy import text

from shared.db.session import sync_session_maker
from app.celery import celery_app
from projects.agent.config import agent_settings

import structlog

logger = structlog.get_logger()


def _parse_executed_at(raw_executed_at) -> datetime:
    """Converte executed_at (ISO string) para datetime timezone-aware UTC."""
    if isinstance(raw_executed_at, datetime):
        executed_at = raw_executed_at
    elif isinstance(raw_executed_at, str):
        executed_at = datetime.fromisoformat(raw_executed_at.replace("Z", "+00:00"))
    else:
        raise ValueError("executed_at ausente ou invalido")

    if executed_at.tzinfo is None:
        executed_at = executed_at.replace(tzinfo=timezone.utc)
    return executed_at.astimezone(timezone.utc)


@celery_app.task(queue="default")
def calculate_action_impact():
    """Busca acoes com after_metrics=None e executed_at > 7 dias atras.

    Coleta metricas atuais e calcula impacto percentual.

    Fluxo:
      1. Busca acoes do Store que ainda nao tem after_metrics
      2. Para cada acao com executed_at > 7 dias atras:
         a. Busca metricas medias dos ultimos 7 dias (after)
         b. Calcula impact_pct = (after - before) / before * 100
         c. Atualiza registro no Store
      3. Emite metrica Prometheus agent_action_impact_pct

    Agendado: diariamente as 06:00 AM via Celery Beat.
    """
    logger.info("agent.impact.calculate.start")
    if not agent_settings.enable_agent_jobs:
        logger.info("agent.impact.calculate.skipped", reason="AGENT_ENABLE_AGENT_JOBS=false")
        return {"skipped": True}

    # Nota: Este job precisa acessar o PostgresStore do LangGraph.
    # Como o Store e async e os jobs Celery sao sync, usamos query
    # direta ao banco para buscar/atualizar registros.

    now_utc = datetime.now(timezone.utc)
    cutoff = now_utc - timedelta(days=7)

    try:
        with sync_session_maker() as session:
            # Buscar acoes pendentes de calculo de impacto
            result = session.execute(text(
                "SELECT key, value FROM store "
                "WHERE namespace LIKE '%%action_history%%' "
                "AND value->>'after_metrics' IS NULL"
            ))

            pending_actions = result.all()

            if not pending_actions:
                logger.info("agent.impact.calculate.no_pending")
                return {"calculated": 0}

            calculated = 0
            for row in pending_actions:
                key = row[0]
                action = row[1]

                if not isinstance(action, dict):
                    continue

                campaign_id = action.get("campaign_id")
                before_metrics = action.get("before_metrics", {})
                raw_executed_at = action.get("executed_at")

                if not campaign_id or not before_metrics:
                    continue

                try:
                    executed_at = _parse_executed_at(raw_executed_at)
                except Exception:
                    logger.warning(
                        "agent.impact.calculate.skip_invalid_executed_at",
                        key=key,
                        executed_at=raw_executed_at,
                    )
                    continue

                # Aguarda 7 dias completos apos execucao para calcular impacto.
                if executed_at > cutoff:
                    continue

                after_start = executed_at
                after_end = executed_at + timedelta(days=7)

                # Buscar metricas no periodo relativo a acao [executed_at, +7 dias]
                after_result = session.execute(text(
                    "SELECT "
                    "  AVG(cost_per_lead) as avg_cpl, "
                    "  AVG(ctr) as avg_ctr, "
                    "  SUM(leads) as total_leads, "
                    "  SUM(spend) as total_spend "
                    "FROM sistema_facebook_ads_insights_history "
                    "WHERE campaign_id = :cid "
                    "AND date >= :start_date "
                    "AND date <= :end_date"
                ), {
                    "cid": campaign_id,
                    "start_date": after_start.date(),
                    "end_date": after_end.date(),
                })
                after_row = after_result.one_or_none()

                if not after_row:
                    continue

                after_metrics = {
                    "cpl": float(after_row.avg_cpl or 0),
                    "ctr": float(after_row.avg_ctr or 0),
                    "leads": int(after_row.total_leads or 0),
                    "spend": float(after_row.total_spend or 0),
                }

                # Calcular impacto percentual
                impact_pct = {}
                for metric_name in ["cpl", "ctr", "leads", "spend"]:
                    before_val = before_metrics.get(metric_name, 0)
                    after_val = after_metrics.get(metric_name, 0)
                    if before_val and before_val != 0:
                        impact_pct[metric_name] = round(
                            ((after_val - before_val) / before_val) * 100, 1,
                        )
                    else:
                        impact_pct[metric_name] = None

                # Atualizar registro no Store
                session.execute(text(
                    "UPDATE store SET value = value || (:updates)::jsonb "
                    "WHERE key = :key "
                    "AND namespace LIKE '%%action_history%%'"
                ), {
                    "key": key,
                    "updates": json.dumps({
                        "after_metrics": after_metrics,
                        "impact_pct": impact_pct,
                        "impact_calculated_at": datetime.now(
                            timezone.utc,
                        ).isoformat(),
                    }),
                })

                calculated += 1

            session.commit()

        logger.info("agent.impact.calculate.done", calculated=calculated)
        return {"calculated": calculated}

    except Exception as e:
        logger.error("agent.impact.calculate.error", error=str(e))
        return {"error": str(e)}
