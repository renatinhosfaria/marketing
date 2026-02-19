from datetime import date

import projects.facebook_ads.api.campaigns as campaigns_api
import projects.facebook_ads.api.insights as insights_api


def test_insights_last_7d_excludes_today(monkeypatch):
    monkeypatch.setattr(insights_api, "get_today_sao_paulo", lambda: date(2026, 2, 9))

    since, until = insights_api._parse_date_params(None, None, "last_7d")

    assert since.strftime("%Y-%m-%d %H:%M:%S") == "2026-02-02 00:00:00"
    assert until.strftime("%Y-%m-%d %H:%M:%S") == "2026-02-08 23:59:59"


def test_campaigns_last_7d_excludes_today(monkeypatch):
    monkeypatch.setattr(campaigns_api, "get_today_sao_paulo", lambda: date(2026, 2, 9))

    since, until = campaigns_api._parse_date_params("last_7d")

    assert since.strftime("%Y-%m-%d %H:%M:%S") == "2026-02-02 00:00:00"
    assert until.strftime("%Y-%m-%d %H:%M:%S") == "2026-02-08 23:59:59"


def test_campaigns_last_7d_does_not_include_today_table():
    assert campaigns_api._should_use_today_table("last_7d") is False
