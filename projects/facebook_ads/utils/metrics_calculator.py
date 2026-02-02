"""
Calculadora de métricas derivadas para Facebook Ads.

Todas as divisões são seguras (retornam None em vez de ZeroDivisionError)
e usam Decimal para precisão financeira.
"""

from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Optional


def safe_divide(
    numerator: int | float | Decimal,
    denominator: int | float | Decimal,
    precision: int = 4,
) -> Optional[Decimal]:
    """Safely divide two numbers, returning None if denominator is 0."""
    try:
        if not denominator or denominator == 0:
            return None
        result = Decimal(str(numerator)) / Decimal(str(denominator))
        return result.quantize(Decimal(f"0.{'0' * precision}"), rounding=ROUND_HALF_UP)
    except (InvalidOperation, ZeroDivisionError):
        return None


def calculate_ctr(clicks: int, impressions: int) -> Optional[Decimal]:
    """CTR = (clicks / impressions) * 100"""
    result = safe_divide(clicks, impressions)
    return result * 100 if result else None


def calculate_cpc(spend: Decimal | float, clicks: int) -> Optional[Decimal]:
    """CPC = spend / clicks"""
    return safe_divide(spend, clicks, 4)


def calculate_cpm(spend: Decimal | float, impressions: int) -> Optional[Decimal]:
    """CPM = (spend / impressions) * 1000"""
    result = safe_divide(spend, impressions, 6)
    return (
        (result * 1000).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)
        if result
        else None
    )


def calculate_cpl(spend: Decimal | float, leads: int) -> Optional[Decimal]:
    """CPL = spend / leads"""
    return safe_divide(spend, leads, 4)


def calculate_frequency(impressions: int, reach: int) -> Optional[Decimal]:
    """Frequency = impressions / reach"""
    return safe_divide(impressions, reach, 4)


def calculate_cpp(spend: Decimal | float, reach: int) -> Optional[Decimal]:
    """CPP = (spend / reach) * 1000 — custo por 1000 pessoas únicas."""
    result = safe_divide(spend, reach, 6)
    return (
        (result * 1000).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)
        if result
        else None
    )


def extract_roas_from_list(roas_list: list | None) -> Optional[Decimal]:
    """Extrai valor de ROAS de uma lista de AdsActionStats."""
    if not roas_list or not isinstance(roas_list, list):
        return None
    for item in roas_list:
        if isinstance(item, dict):
            try:
                val = Decimal(str(item.get("value", "0")))
                if val > 0:
                    return val.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)
            except (InvalidOperation, ValueError):
                pass
    return None


def extract_video_metric(insight: dict, field_name: str) -> int:
    """Extrai métrica de vídeo genérica de uma lista de AdsActionStats."""
    data = insight.get(field_name, [])
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                try:
                    return int(item.get("value", 0))
                except (ValueError, TypeError):
                    pass
    return 0


def extract_video_avg_time(insight: dict) -> Optional[Decimal]:
    """Extrai tempo médio de vídeo assistido."""
    data = insight.get("video_avg_time_watched_actions", [])
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                try:
                    return Decimal(str(item.get("value", "0"))).quantize(
                        Decimal("0.01"), rounding=ROUND_HALF_UP
                    )
                except (InvalidOperation, ValueError):
                    pass
    return None


def extract_action_stat_value(
    data: list | None, decimal_result: bool = False
) -> int | Optional[Decimal]:
    """Extrai valor de um campo que é List[AdsActionStats] (ex: cost_per_outbound_click)."""
    if not data or not isinstance(data, list):
        return Decimal("0") if decimal_result else 0
    for item in data:
        if isinstance(item, dict):
            try:
                val = item.get("value", "0")
                if decimal_result:
                    return Decimal(str(val)).quantize(
                        Decimal("0.0001"), rounding=ROUND_HALF_UP
                    )
                return int(val)
            except (InvalidOperation, ValueError, TypeError):
                pass
    return Decimal("0") if decimal_result else 0


def extract_landing_page_views(actions: list | None) -> int:
    """Extrai landing_page_view do array de actions."""
    if not actions or not isinstance(actions, list):
        return 0
    for action in actions:
        if isinstance(action, dict) and action.get("action_type") == "landing_page_view":
            try:
                return int(action.get("value", 0))
            except (ValueError, TypeError):
                pass
    return 0


def extract_leads_from_actions(actions: dict | list | None) -> int:
    """Extract lead count from Facebook actions array/dict.

    Uses 'lead' action type (the aggregate total shown in Ads Manager).
    Falls back to specific types only if 'lead' is not present.
    """
    if not actions:
        return 0

    action_list = actions if isinstance(actions, list) else [actions]

    # Build a map of action_type -> value
    action_map: dict[str, int] = {}
    for action in action_list:
        if isinstance(action, dict):
            action_type = action.get("action_type", "")
            try:
                action_map[action_type] = int(action.get("value", 0))
            except (ValueError, TypeError):
                pass

    # 'lead' is the aggregate total — use it if available
    if "lead" in action_map:
        return action_map["lead"]

    # Fallback: check specific types (pick first found, don't sum to avoid overlap)
    fallback_types = [
        "leadgen_grouped",
        "onsite_conversion.lead_grouped",
        "offsite_conversion.fb_pixel_lead",
    ]
    for lt in fallback_types:
        if lt in action_map:
            return action_map[lt]

    return 0
