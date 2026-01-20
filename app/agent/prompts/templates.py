"""
Templates de resposta formatada para o agente.
"""

from typing import Any, Dict, List, Optional


class ResponseTemplates:
    """Templates para formatação de respostas do agente."""

    @staticmethod
    def campaign_summary(
        name: str,
        tier: str,
        cpl: str,
        leads: int,
        spend: str,
        trend: str = "estável"
    ) -> str:
        """Template para resumo de campanha."""
        tier_emoji = {
            "HIGH_PERFORMER": "🌟",
            "MODERATE": "📊",
            "LOW": "⚠️",
            "UNDERPERFORMER": "🔴",
        }.get(tier, "📋")

        return f"""**{name}** {tier_emoji}
📊 Classificação: {tier}
💰 CPL: {cpl} | 📈 Leads: {leads}
💵 Investimento: {spend}
📉 Tendência: {trend}"""

    @staticmethod
    def anomaly_alert(
        campaign_name: str,
        anomaly_type: str,
        severity: str,
        description: str,
        suggested_action: str
    ) -> str:
        """Template para alerta de anomalia."""
        severity_emoji = {
            "CRITICAL": "🔴",
            "HIGH": "🟠",
            "MEDIUM": "🟡",
            "LOW": "🟢",
        }.get(severity, "⚪")

        return f"""{severity_emoji} **{severity}** - {campaign_name}
📋 Tipo: {anomaly_type}
📝 {description}
💡 **Ação sugerida:** {suggested_action}"""

    @staticmethod
    def recommendation_card(
        campaign_name: str,
        rec_type: str,
        priority: int,
        title: str,
        description: str,
        confidence: float
    ) -> str:
        """Template para card de recomendação."""
        priority_emoji = "🔴" if priority >= 8 else "🟠" if priority >= 6 else "🟡"

        type_action = {
            "SCALE_UP": "Escalar 📈",
            "BUDGET_INCREASE": "Aumentar Budget 💰",
            "BUDGET_DECREASE": "Reduzir Budget 📉",
            "PAUSE_CAMPAIGN": "Pausar ⏸️",
            "CREATIVE_REFRESH": "Renovar Criativos 🎨",
            "AUDIENCE_REVIEW": "Revisar Público 👥",
            "REACTIVATE": "Reativar ▶️",
        }.get(rec_type, rec_type)

        return f"""{priority_emoji} **{title}**
🎯 Campanha: {campaign_name}
📋 Ação: {type_action}
📝 {description}
🎲 Confiança: {confidence:.0f}%"""

    @staticmethod
    def comparison_table(comparisons: List[Dict[str, Any]]) -> str:
        """Template para tabela de comparação."""
        if not comparisons:
            return "Sem dados para comparar."

        # Header
        table = "| Campanha | CPL | Leads | Spend | Status |\n"
        table += "|----------|-----|-------|-------|--------|\n"

        # Rows
        for c in comparisons:
            best_cpl = "✅" if c.get("is_best_cpl") else ""
            best_leads = "🏆" if c.get("is_best_leads") else ""

            table += f"| {c.get('name', 'N/A')[:20]} | {c.get('cpl', 'N/A')} {best_cpl} | "
            table += f"{c.get('leads', 0)} {best_leads} | {c.get('spend', 'N/A')} | "
            table += f"{c.get('status', 'N/A')} |\n"

        return table

    @staticmethod
    def metrics_summary(
        total_spend: str,
        total_leads: int,
        avg_cpl: str,
        period_days: int
    ) -> str:
        """Template para resumo de métricas."""
        return f"""📊 **Resumo dos últimos {period_days} dias**

💵 Investimento Total: {total_spend}
📈 Total de Leads: {total_leads}
💰 CPL Médio: {avg_cpl}
📅 Período: {period_days} dias"""

    @staticmethod
    def roi_analysis(
        total_investment: str,
        total_leads: int,
        estimated_sales: float,
        estimated_revenue: str,
        roi: str
    ) -> str:
        """Template para análise de ROI."""
        return f"""💰 **Análise de ROI**

📊 Investimento: {total_investment}
📈 Leads Gerados: {total_leads}
🎯 Vendas Estimadas: {estimated_sales:.1f}
💵 Receita Projetada: {estimated_revenue}
📊 **ROI: {roi}**"""

    @staticmethod
    def action_list(actions: List[Dict[str, str]]) -> str:
        """Template para lista de ações."""
        if not actions:
            return "Nenhuma ação pendente."

        result = "📋 **Lista de Ações Recomendadas**\n\n"

        for i, action in enumerate(actions, 1):
            priority = action.get("priority", "")
            emoji = "🔴" if priority == "alta" else "🟠" if priority == "média" else "🟢"
            result += f"{i}. {emoji} **{action.get('title', 'Ação')}**\n"
            result += f"   📝 {action.get('description', '')}\n\n"

        return result

    @staticmethod
    def error_message(error: str, suggestion: str = "") -> str:
        """Template para mensagem de erro."""
        msg = f"❌ **Erro:** {error}"
        if suggestion:
            msg += f"\n💡 **Sugestão:** {suggestion}"
        return msg

    @staticmethod
    def no_data_message(entity: str, suggestion: str = "") -> str:
        """Template para quando não há dados."""
        msg = f"📭 Não encontrei dados de {entity} para exibir."
        if suggestion:
            msg += f"\n💡 {suggestion}"
        return msg

    @staticmethod
    def success_message(message: str) -> str:
        """Template para mensagem de sucesso."""
        return f"✅ {message}"

    @staticmethod
    def warning_message(message: str) -> str:
        """Template para mensagem de aviso."""
        return f"⚠️ {message}"
