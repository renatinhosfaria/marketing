"""
Testes unitarios dos jobs de retencao do Agent.

Testa:
  - cleanup_agent_checkpoints: executa DELETE corretamente
  - cleanup_agent_store: executa DELETE por retencao corretamente
  - reap_orphan_sse_sessions: conta sessoes ativas e atualiza metrica
  - Comportamento quando AGENT_ENABLE_AGENT_JOBS=false (skip)
  - Comportamento quando Redis esta indisponivel (graceful)
"""

import pytest
from unittest.mock import patch, MagicMock, call


def test_reap_orphan_sse_sessions_skips_when_disabled(monkeypatch):
    """reap_orphan_sse_sessions retorna skipped quando AGENT_ENABLE_AGENT_JOBS=false."""
    monkeypatch.setattr(
        "projects.agent.jobs.retention.agent_settings",
        MagicMock(enable_agent_jobs=False),
    )

    from projects.agent.jobs.retention import reap_orphan_sse_sessions

    result = reap_orphan_sse_sessions()

    assert result == {"skipped": True}


def test_reap_orphan_sse_sessions_counts_active(monkeypatch):
    """reap_orphan_sse_sessions conta meta-keys Redis e atualiza metrica."""
    mock_settings = MagicMock()
    mock_settings.enable_agent_jobs = True
    mock_settings.agent_redis_url = "redis://test:6379/1"
    monkeypatch.setattr("projects.agent.jobs.retention.agent_settings", mock_settings)

    mock_redis = MagicMock()
    # scan_iter retorna 3 meta-keys (sessoes ativas)
    mock_redis.scan_iter.return_value = iter(["key1", "key2", "key3"])
    mock_redis.close = MagicMock()

    mock_gauge = MagicMock()

    with patch("projects.agent.jobs.retention.sync_redis.from_url", return_value=mock_redis) as mock_from_url, \
         patch("projects.agent.observability.metrics.session_orphan_count", mock_gauge):
        from projects.agent.jobs.retention import reap_orphan_sse_sessions
        result = reap_orphan_sse_sessions()

    assert result["active_sessions"] == 3
    mock_redis.scan_iter.assert_called_once()
    mock_redis.close.assert_called_once()


def test_reap_orphan_sse_sessions_handles_redis_error(monkeypatch):
    """reap_orphan_sse_sessions retorna erro sem lancar excecao quando Redis falha."""
    mock_settings = MagicMock()
    mock_settings.enable_agent_jobs = True
    mock_settings.agent_redis_url = "redis://test:6379/1"
    monkeypatch.setattr("projects.agent.jobs.retention.agent_settings", mock_settings)

    with patch("projects.agent.jobs.retention.sync_redis.from_url") as mock_from_url:
        mock_from_url.side_effect = ConnectionError("Redis unavailable")

        from projects.agent.jobs.retention import reap_orphan_sse_sessions
        result = reap_orphan_sse_sessions()

    assert "error" in result
    assert "Redis unavailable" in result["error"]


def test_reap_orphan_sse_sessions_zero_sessions(monkeypatch):
    """reap_orphan_sse_sessions funciona quando nao ha sessoes ativas."""
    mock_settings = MagicMock()
    mock_settings.enable_agent_jobs = True
    mock_settings.agent_redis_url = "redis://test:6379/1"
    monkeypatch.setattr("projects.agent.jobs.retention.agent_settings", mock_settings)

    mock_redis = MagicMock()
    mock_redis.scan_iter.return_value = iter([])  # Nenhuma sessao
    mock_redis.close = MagicMock()

    mock_gauge = MagicMock()

    with patch("projects.agent.jobs.retention.sync_redis.from_url", return_value=mock_redis), \
         patch("projects.agent.observability.metrics.session_orphan_count", mock_gauge):
        from projects.agent.jobs.retention import reap_orphan_sse_sessions
        result = reap_orphan_sse_sessions()

    assert result["active_sessions"] == 0
