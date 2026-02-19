"""
Tasks Celery para limpeza de dados do Agent.

cleanup_agent_checkpoints: remove checkpoints mais antigos que 30 dias.
cleanup_agent_store: remove memorias expiradas por namespace.

Agendamento via Celery Beat:
  - cleanup_agent_checkpoints: todo domingo as 03:00 AM
  - cleanup_agent_store: primeiro domingo do mes as 04:00 AM
"""

from sqlalchemy import text

from shared.db.session import sync_session_maker
from app.celery import celery_app
from projects.agent.config import agent_settings

import structlog

logger = structlog.get_logger()

# Nomes de tabelas do LangGraph (validar antes de executar DELETE):
# - AsyncPostgresSaver: "checkpoints", "checkpoint_blobs", "checkpoint_writes"
# - PostgresStore: "store"
EXPECTED_TABLES = {"checkpoints", "store"}


def _validate_tables_exist(session) -> None:
    """Verifica que as tabelas do LangGraph existem antes do cleanup.

    Previne erros silenciosos se migrations nao rodaram.
    """
    result = session.execute(text(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema = 'public' AND table_name = ANY(:names)"
    ), {"names": list(EXPECTED_TABLES)})
    existing = {row[0] for row in result}
    missing = EXPECTED_TABLES - existing
    if missing:
        raise RuntimeError(
            f"Tabelas do LangGraph nao encontradas: {missing}. "
            "Execute as migrations antes do cleanup."
        )


@celery_app.task(queue="default")
def cleanup_agent_checkpoints():
    """Remove checkpoints mais antigos que 30 dias.

    Agendado: todo domingo as 03:00 AM via Celery Beat.
    """
    logger.info("agent.cleanup_checkpoints.start")
    if not agent_settings.enable_agent_jobs:
        logger.info("agent.cleanup_checkpoints.skipped", reason="AGENT_ENABLE_AGENT_JOBS=false")
        return {"skipped": True}

    with sync_session_maker() as session:
        _validate_tables_exist(session)

        result = session.execute(text(
            "DELETE FROM checkpoints WHERE updated_at < NOW() - INTERVAL '30 days'"
        ))
        deleted = result.rowcount
        session.commit()

    logger.info("agent.cleanup_checkpoints.done", deleted=deleted)
    return {"deleted_checkpoints": deleted}


@celery_app.task(queue="default")
def cleanup_agent_store():
    """Remove memorias expiradas do Store por namespace.

    Retencao:
      - patterns: indefinido (valor permanente)
      - insights: 90 dias
      - action_history: 180 dias

    Agendado: primeiro domingo do mes as 04:00 AM via Celery Beat.
    """
    logger.info("agent.cleanup_store.start")
    if not agent_settings.enable_agent_jobs:
        logger.info("agent.cleanup_store.skipped", reason="AGENT_ENABLE_AGENT_JOBS=false")
        return {"skipped": True}

    with sync_session_maker() as session:
        _validate_tables_exist(session)

        # Insights: 90 dias
        insights_result = session.execute(text(
            "DELETE FROM store WHERE namespace LIKE '%%insights%%' "
            "AND updated_at < NOW() - INTERVAL '90 days'"
        ))
        deleted_insights = insights_result.rowcount

        # Action history: 180 dias
        actions_result = session.execute(text(
            "DELETE FROM store WHERE namespace LIKE '%%action_history%%' "
            "AND updated_at < NOW() - INTERVAL '180 days'"
        ))
        deleted_actions = actions_result.rowcount

        session.commit()

    logger.info(
        "agent.cleanup_store.done",
        deleted_insights=deleted_insights,
        deleted_actions=deleted_actions,
    )
    return {
        "deleted_insights": deleted_insights,
        "deleted_actions": deleted_actions,
    }
