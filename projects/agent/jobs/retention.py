"""
Tasks Celery para limpeza de dados do Agent.

cleanup_agent_checkpoints: remove checkpoints mais antigos que 30 dias.
cleanup_agent_store: remove memorias expiradas por namespace.
reap_orphan_sse_sessions: conta sessoes SSE ativas no Redis e atualiza metrica.

Agendamento via Celery Beat:
  - cleanup_agent_checkpoints: todo domingo as 03:00 AM
  - cleanup_agent_store: primeiro domingo do mes as 04:00 AM
  - reap_orphan_sse_sessions: a cada 5 minutos
"""

import redis as sync_redis

from sqlalchemy import text

from shared.db.session import sync_session_maker
from app.celery import celery_app
from projects.agent.config import agent_settings

import structlog

logger = structlog.get_logger()

# Prefixo das meta-keys de sessao SSE no Redis
_SESSION_META_PATTERN = "agent:sse:meta:*"

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


@celery_app.task(queue="default")
def reap_orphan_sse_sessions():
    """Conta sessoes SSE ativas no Redis e atualiza metrica Prometheus.

    Uma sessao "ativa" (status=active ou meta-key ainda presente) que nao foi
    encerrada explicitamente e considerada orfã — cliente desconectou antes
    do close_session() ser chamado. O TTL do Redis as remove automaticamente,
    mas o gauge session_orphan_count permite monitorar a taxa.

    Agendado: a cada 5 minutos via Celery Beat.
    """
    logger.info("agent.reap_orphan_sse_sessions.start")
    if not agent_settings.enable_agent_jobs:
        logger.info("agent.reap_orphan_sse_sessions.skipped", reason="AGENT_ENABLE_AGENT_JOBS=false")
        return {"skipped": True}

    try:
        r = sync_redis.from_url(
            agent_settings.agent_redis_url,
            decode_responses=True,
            socket_connect_timeout=5,
        )

        # Conta todas as meta-keys presentes (cada uma = sessao ativa ou recentemente encerrada)
        count = sum(1 for _ in r.scan_iter(_SESSION_META_PATTERN, count=100))
        r.close()

        from projects.agent.observability.metrics import session_orphan_count
        session_orphan_count.set(count)

        logger.info("agent.reap_orphan_sse_sessions.done", active_sessions=count)
        return {"active_sessions": count}

    except Exception as exc:
        logger.warning("agent.reap_orphan_sse_sessions.failed", error=str(exc))
        return {"error": str(exc)}
