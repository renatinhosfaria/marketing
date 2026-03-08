"""Translator MVP de prompt em linguagem natural para SQL controlado."""

import re


class PromptToSQL:
    """Traduz prompts conhecidos para SQL seguro e deterministico."""

    @staticmethod
    def translate(prompt: str) -> str:
        normalized = prompt.lower().strip()

        if "top" in normalized and "campanh" in normalized and "spend" in normalized:
            config_match = re.search(r"config\s*(\d+)", normalized)
            config_id = int(config_match.group(1)) if config_match else 1

            return (
                "SELECT campaign_id, SUM(spend) AS spend "
                "FROM sistema_facebook_ads_insights_history "
                f"WHERE config_id = {config_id} "
                "GROUP BY campaign_id "
                "ORDER BY spend DESC "
                "LIMIT 5"
            )

        raise ValueError("Nao consegui traduzir o prompt para SQL com seguranca")
