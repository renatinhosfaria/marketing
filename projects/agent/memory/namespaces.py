"""
Contrato unico de namespaces do Store.

Todos os acessos ao Store (tools, nos e testes) usam estas constantes.
Evita fragmentacao e garante que save e recall usem o mesmo namespace.
"""


class StoreNamespace:
    """Contrato unico para namespaces do Store. Evita fragmentacao."""

    PROFILE = "profile"              # Preferencias globais do usuario
    PATTERNS = "patterns"            # Padroes recorrentes da conta
    INSIGHTS = "insights"            # Insights descobertos (por categoria)
    ACTION_HISTORY = "action_history"  # Historico de acoes executadas
    TITLES = "titles"                    # Titulos de conversas

    @staticmethod
    def user_profile(user_id: str) -> tuple:
        """Namespace para preferencias globais do usuario."""
        return (user_id, StoreNamespace.PROFILE)

    @staticmethod
    def account_patterns(user_id: str, account_id: str) -> tuple:
        """Namespace para padroes recorrentes da conta."""
        return (user_id, account_id, StoreNamespace.PATTERNS)

    @staticmethod
    def account_insights(user_id: str, account_id: str) -> tuple:
        """Namespace para insights descobertos da conta."""
        return (user_id, account_id, StoreNamespace.INSIGHTS)

    @staticmethod
    def account_actions(user_id: str, account_id: str) -> tuple:
        """Namespace para historico de acoes executadas."""
        return (user_id, account_id, StoreNamespace.ACTION_HISTORY)

    @staticmethod
    def campaign_insights(
        user_id: str, account_id: str, campaign_id: str,
    ) -> tuple:
        """Namespace para insights granulares por campanha."""
        return (user_id, account_id, campaign_id, StoreNamespace.INSIGHTS)

    @staticmethod
    def conversation_titles(user_id: str, account_id: str) -> tuple:
        """Namespace para titulos de conversas."""
        return (user_id, account_id, StoreNamespace.TITLES)
