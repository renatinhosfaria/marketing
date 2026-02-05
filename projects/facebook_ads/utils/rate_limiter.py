"""
Rate limiter proativo para a API do Facebook.

Monitora os headers de resposta da API (X-Business-Use-Case-Usage, X-Ad-Account-Usage,
X-App-Usage) e aplica delays quando o uso se aproxima dos limites.
"""

import asyncio
import json
import time

from shared.core.logging import get_logger
from projects.facebook_ads.config import fb_settings

logger = get_logger(__name__)


class FacebookRateLimiter:
    """Proactive rate limiter based on Facebook API response headers."""

    def __init__(self):
        self._usage: dict[str, float] = {}  # account_id -> usage percentage
        self._last_update: dict[str, float] = {}
        self._lock = asyncio.Lock()

    def update_from_headers(self, headers: dict, account_id: str = "default") -> None:
        """Parse rate limit headers from Facebook API response."""
        usage = 0.0

        # X-Business-Use-Case-Usage: {"act_123": [{"call_count": 28, ...}]}
        buc_usage = headers.get("X-Business-Use-Case-Usage") or headers.get(
            "x-business-use-case-usage"
        )
        if buc_usage:
            try:
                parsed = (
                    json.loads(buc_usage) if isinstance(buc_usage, str) else buc_usage
                )
                for acc_id, metrics_list in parsed.items():
                    if metrics_list and isinstance(metrics_list, list):
                        m = metrics_list[0]
                        usage = max(
                            m.get("call_count", 0),
                            m.get("total_cputime", 0),
                            m.get("total_time", 0),
                        )
                        account_id = acc_id
            except (json.JSONDecodeError, KeyError, IndexError):
                pass

        # X-Ad-Account-Usage: {"acc_id_util_pct": 9.67}
        aa_usage = headers.get("X-Ad-Account-Usage") or headers.get(
            "x-ad-account-usage"
        )
        if aa_usage:
            try:
                parsed = (
                    json.loads(aa_usage) if isinstance(aa_usage, str) else aa_usage
                )
                pct = parsed.get("acc_id_util_pct", 0)
                usage = max(usage, float(pct))
            except (json.JSONDecodeError, KeyError, ValueError):
                pass

        # X-App-Usage: {"call_count": 10, "total_cputime": 5, "total_time": 5}
        app_usage = headers.get("X-App-Usage") or headers.get("x-app-usage")
        if app_usage:
            try:
                parsed = (
                    json.loads(app_usage) if isinstance(app_usage, str) else app_usage
                )
                app_max = max(
                    parsed.get("call_count", 0),
                    parsed.get("total_cputime", 0),
                    parsed.get("total_time", 0),
                )
                usage = max(usage, app_max)
            except (json.JSONDecodeError, KeyError):
                pass

        self._usage[account_id] = usage
        self._last_update[account_id] = time.time()

        if usage > fb_settings.facebook_rate_limit_threshold:
            logger.warning(
                "Rate limit elevado",
                account_id=account_id,
                usage_pct=usage,
                threshold=fb_settings.facebook_rate_limit_threshold,
            )

    async def check_and_wait(self, account_id: str = "default") -> None:
        """Wait if rate limit is approaching threshold."""
        async with self._lock:
            usage = self._usage.get(account_id, 0)

            if usage >= fb_settings.facebook_rate_limit_pause_threshold:
                wait_time = 30.0
                logger.warning(
                    "Rate limit crítico - pausando",
                    account_id=account_id,
                    usage_pct=usage,
                    wait_seconds=wait_time,
                )
                await asyncio.sleep(wait_time)
            elif usage >= fb_settings.facebook_rate_limit_threshold:
                wait_time = 5.0
                logger.info(
                    "Rate limit elevado - delay",
                    account_id=account_id,
                    usage_pct=usage,
                    wait_seconds=wait_time,
                )
                await asyncio.sleep(wait_time)

    def get_usage(self, account_id: str = "default") -> float:
        """Get current usage percentage for an account."""
        return self._usage.get(account_id, 0.0)


# Instância global
rate_limiter = FacebookRateLimiter()
