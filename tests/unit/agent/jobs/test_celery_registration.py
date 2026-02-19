"""Testes de registro de jobs do Agent no Celery."""

from app.celery import celery_app


def test_agent_jobs_modules_included():
    """Celery inclui modulos de jobs do Agent para discover de tasks."""
    includes = celery_app.conf.include or []
    assert "projects.agent.jobs.impact" in includes
    assert "projects.agent.jobs.retention" in includes


def test_agent_jobs_present_in_beat_schedule():
    """Beat schedule contem entradas dos jobs de impacto e retencao do Agent."""
    schedule = celery_app.conf.beat_schedule or {}
    task_names = {entry.get("task") for entry in schedule.values()}
    assert "projects.agent.jobs.impact.calculate_action_impact" in task_names
    assert "projects.agent.jobs.retention.cleanup_agent_checkpoints" in task_names
    assert "projects.agent.jobs.retention.cleanup_agent_store" in task_names
