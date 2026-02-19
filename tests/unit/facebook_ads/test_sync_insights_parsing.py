from datetime import datetime
from decimal import Decimal

from shared.db.models.famachat_readonly import SistemaFacebookAdsInsightsHistory
from projects.facebook_ads.services.sync_insights import SyncInsightsService


def _sample_insight_with_list_ctr() -> dict:
    return {
        "ad_id": "120233894376070206",
        "adset_id": "120233894285100206",
        "campaign_id": "120233891992660206",
        "date_start": "2026-02-04",
        "impressions": "10",
        "reach": "8",
        "clicks": "2",
        "spend": "1.00",
        "actions": [
            {"action_type": "lead", "value": "5"},
            {"action_type": "outbound_click", "value": "1"},
        ],
        # Facebook pode retornar estes campos como list[AdsActionStats]
        "outbound_clicks_ctr": [
            {"action_type": "outbound_click", "value": "0.053821"},
        ],
        "unique_outbound_clicks_ctr": [
            {"action_type": "outbound_click", "value": "0.063735"},
        ],
    }


def test_parse_insight_history_handles_list_ctr_fields():
    service = SyncInsightsService(db=None)
    obj = service._parse_insight_to_history(14, _sample_insight_with_list_ctr())

    assert obj.leads == 5
    assert obj.outbound_clicks_ctr == Decimal("0.0538")
    assert obj.unique_outbound_clicks_ctr == Decimal("0.0637")


def test_update_insight_history_handles_list_ctr_fields():
    service = SyncInsightsService(db=None)
    obj = SistemaFacebookAdsInsightsHistory(
        config_id=14,
        ad_id="120233894376070206",
        adset_id="120233894285100206",
        campaign_id="120233891992660206",
        date=datetime(2026, 2, 4),
    )

    service._update_insight_history(obj, _sample_insight_with_list_ctr())

    assert obj.leads == 5
    assert obj.outbound_clicks_ctr == Decimal("0.0538")
    assert obj.unique_outbound_clicks_ctr == Decimal("0.0637")
