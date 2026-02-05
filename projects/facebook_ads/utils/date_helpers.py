"""
Utilitários de data e hora para o módulo Facebook Ads.
Todas as datas são tratadas no fuso horário de São Paulo (America/Sao_Paulo).
"""

from datetime import date, datetime, timedelta, timezone

import pytz

SAO_PAULO_TZ = pytz.timezone("America/Sao_Paulo")


def get_today_sao_paulo() -> date:
    """Get today's date in São Paulo timezone."""
    return datetime.now(SAO_PAULO_TZ).date()


def is_today(check_date: date | datetime) -> bool:
    """Check if a date is today (São Paulo timezone)."""
    if isinstance(check_date, datetime):
        check_date = check_date.date()
    return check_date == get_today_sao_paulo()


def get_date_range(days_back: int) -> tuple[str, str]:
    """Get date range as (since, until) strings in YYYY-MM-DD format."""
    today = get_today_sao_paulo()
    since = today - timedelta(days=days_back)
    return since.strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d")


def parse_facebook_datetime(dt_str: str | None) -> datetime | None:
    """Parse Facebook API datetime string (ISO 8601).

    Returns a naive (timezone-unaware) datetime in UTC,
    compatible with TIMESTAMP WITHOUT TIME ZONE columns.
    """
    if not dt_str:
        return None
    try:
        dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
        # Convert to UTC and strip tzinfo for naive TIMESTAMP columns
        if dt.tzinfo is not None:
            dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
        return dt
    except (ValueError, AttributeError):
        return None
