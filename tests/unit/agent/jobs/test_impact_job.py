"""Testes do job de impacto do Agent."""

from datetime import date
from unittest.mock import MagicMock, patch

from projects.agent.jobs.impact import calculate_action_impact


def _mock_sync_session(mock_session):
    """Cria context manager sync para patch de sync_session_maker."""
    mock_ctx = MagicMock()
    mock_ctx.__enter__.return_value = mock_session
    mock_ctx.__exit__.return_value = False
    return mock_ctx


def test_calculate_action_impact_uses_action_relative_window():
    """Job usa janela [executed_at, executed_at+7d] por acao."""
    pending = (
        "k1",
        {
            "campaign_id": "c1",
            "executed_at": "2026-01-10T12:00:00+00:00",
            "before_metrics": {"cpl": 10.0, "ctr": 1.0, "leads": 100, "spend": 1000.0},
            "after_metrics": None,
        },
    )

    pending_result = MagicMock()
    pending_result.all.return_value = [pending]

    after_row = MagicMock()
    after_row.avg_cpl = 12.0
    after_row.avg_ctr = 1.2
    after_row.total_leads = 120
    after_row.total_spend = 1300.0
    after_result = MagicMock()
    after_result.one_or_none.return_value = after_row

    update_result = MagicMock()

    mock_session = MagicMock()
    mock_session.execute.side_effect = [pending_result, after_result, update_result]
    mock_session.commit = MagicMock()

    with patch("projects.agent.jobs.impact.sync_session_maker", return_value=_mock_sync_session(mock_session)), \
         patch("projects.agent.jobs.impact.agent_settings") as mock_settings:
        mock_settings.enable_agent_jobs = True
        result = calculate_action_impact()

    assert result == {"calculated": 1}
    assert mock_session.execute.call_count == 3
    after_params = mock_session.execute.call_args_list[1][0][1]
    assert after_params["start_date"] == date(2026, 1, 10)
    assert after_params["end_date"] == date(2026, 1, 17)


def test_calculate_action_impact_skips_invalid_executed_at_without_failing_batch():
    """Job ignora acao com executed_at invalido e continua lote."""
    pending = (
        "k1",
        {
            "campaign_id": "c1",
            "executed_at": "data-invalida",
            "before_metrics": {"cpl": 10.0},
            "after_metrics": None,
        },
    )

    pending_result = MagicMock()
    pending_result.all.return_value = [pending]

    mock_session = MagicMock()
    mock_session.execute.side_effect = [pending_result]
    mock_session.commit = MagicMock()

    with patch("projects.agent.jobs.impact.sync_session_maker", return_value=_mock_sync_session(mock_session)), \
         patch("projects.agent.jobs.impact.agent_settings") as mock_settings, \
         patch("projects.agent.jobs.impact.logger") as mock_logger:
        mock_settings.enable_agent_jobs = True
        result = calculate_action_impact()

    assert result == {"calculated": 0}
    assert mock_session.execute.call_count == 1
    mock_logger.warning.assert_called()
