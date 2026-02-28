import pytest

from projects.facebook_ads.services.agent_prompt_to_sql import PromptToSQL


def test_translates_top_campaigns_prompt() -> None:
    sql = PromptToSQL.translate("listar top 5 campanhas por spend da config 1")
    normalized = " ".join(sql.upper().split())

    assert "FROM SISTEMA_FACEBOOK_ADS_INSIGHTS_HISTORY" in normalized
    assert "WHERE CONFIG_ID = 1" in normalized
    assert "LIMIT 5" in normalized


def test_rejects_ambiguous_prompt() -> None:
    with pytest.raises(ValueError, match="Nao consegui traduzir"):
        PromptToSQL.translate("faz alguma coisa ai")
