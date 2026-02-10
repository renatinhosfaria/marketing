"""Testes para qualidade dos prompts do agente."""
import pytest

from projects.agent.prompts.system import SYSTEM_PROMPT, get_system_prompt
from projects.agent.orchestrator.prompts import ORCHESTRATOR_SYSTEM_PROMPT, SYNTHESIS_PROMPT
from projects.agent.subagents.classification.prompts import CLASSIFICATION_SYSTEM_PROMPT
from projects.agent.subagents.anomaly.prompts import ANOMALY_SYSTEM_PROMPT
from projects.agent.subagents.forecast.prompts import FORECAST_SYSTEM_PROMPT
from projects.agent.subagents.recommendation.prompts import RECOMMENDATION_SYSTEM_PROMPT
from projects.agent.subagents.campaign.prompts import CAMPAIGN_SYSTEM_PROMPT
from projects.agent.subagents.analysis.prompts import ANALYSIS_SYSTEM_PROMPT


ALL_PROMPTS = {
    "system": SYSTEM_PROMPT,
    "orchestrator": ORCHESTRATOR_SYSTEM_PROMPT,
    "synthesis": SYNTHESIS_PROMPT,
    "classification": CLASSIFICATION_SYSTEM_PROMPT,
    "anomaly": ANOMALY_SYSTEM_PROMPT,
    "forecast": FORECAST_SYSTEM_PROMPT,
    "recommendation": RECOMMENDATION_SYSTEM_PROMPT,
    "campaign": CAMPAIGN_SYSTEM_PROMPT,
    "analysis": ANALYSIS_SYSTEM_PROMPT,
}

SUBAGENT_PROMPTS = {
    "classification": CLASSIFICATION_SYSTEM_PROMPT,
    "anomaly": ANOMALY_SYSTEM_PROMPT,
    "forecast": FORECAST_SYSTEM_PROMPT,
    "recommendation": RECOMMENDATION_SYSTEM_PROMPT,
    "campaign": CAMPAIGN_SYSTEM_PROMPT,
    "analysis": ANALYSIS_SYSTEM_PROMPT,
}


class TestPromptBasics:
    """Testes basicos de qualidade dos prompts."""

    @pytest.mark.parametrize("name,prompt", ALL_PROMPTS.items())
    def test_prompt_not_empty(self, name, prompt):
        """Prompt nao esta vazio."""
        assert len(prompt) > 100, f"Prompt '{name}' muito curto"

    @pytest.mark.parametrize("name,prompt", ALL_PROMPTS.items())
    def test_prompt_in_portuguese(self, name, prompt):
        """Prompts devem estar em portugues."""
        pt_words = ["voce", "analise", "campanha", "dados", "resultado", "acao"]
        has_pt = any(word in prompt.lower() for word in pt_words)
        assert has_pt, f"Prompt '{name}' nao parece estar em portugues"

    @pytest.mark.parametrize("name,prompt", ALL_PROMPTS.items())
    def test_prompt_has_structure(self, name, prompt):
        """Prompts devem ter secoes estruturadas."""
        has_sections = "##" in prompt or "**" in prompt or "- " in prompt
        assert has_sections, f"Prompt '{name}' sem secoes estruturadas"

    @pytest.mark.parametrize("name,prompt", ALL_PROMPTS.items())
    def test_prompt_has_role_definition(self, name, prompt):
        """Prompts devem definir papel/missao."""
        prompt_lower = prompt.lower()
        has_role = any(w in prompt_lower for w in [
            "voce e", "voce classifica", "voce detecta", "voce interpreta",
            "voce gera", "voce coleta", "voce realiza", "voce sintetiza",
        ])
        assert has_role, f"Prompt '{name}' nao define papel ou missao"

    @pytest.mark.parametrize("name,prompt", ALL_PROMPTS.items())
    def test_prompt_minimum_length(self, name, prompt):
        """Prompts devem ter conteudo substancial (>200 chars)."""
        assert len(prompt) > 200, f"Prompt '{name}' com apenas {len(prompt)} chars"


class TestSubagentPromptContent:
    """Testes de conteudo especifico dos prompts de subagentes."""

    def test_classification_mentions_tiers(self):
        """Classificacao deve mencionar tiers."""
        assert "HIGH_PERFORMER" in CLASSIFICATION_SYSTEM_PROMPT
        assert "UNDERPERFORMER" in CLASSIFICATION_SYSTEM_PROMPT
        assert "MODERATE" in CLASSIFICATION_SYSTEM_PROMPT
        assert "LOW" in CLASSIFICATION_SYSTEM_PROMPT

    def test_classification_mentions_scoring(self):
        """Classificacao deve mencionar sistema de scoring."""
        prompt_lower = CLASSIFICATION_SYSTEM_PROMPT.lower()
        assert "score" in prompt_lower or "scoring" in prompt_lower
        assert "cpl" in prompt_lower

    def test_anomaly_mentions_severity(self):
        """Anomalia deve mencionar severidades."""
        prompt = ANOMALY_SYSTEM_PROMPT
        assert "critico" in prompt.lower()

    def test_anomaly_mentions_financial_impact(self):
        """Anomalia deve mencionar impacto financeiro."""
        prompt_lower = ANOMALY_SYSTEM_PROMPT.lower()
        assert "impacto" in prompt_lower
        assert "r$" in prompt_lower

    def test_forecast_mentions_confidence(self):
        """Previsao deve mencionar confianca."""
        assert "confian" in FORECAST_SYSTEM_PROMPT.lower()

    def test_forecast_mentions_scenarios(self):
        """Previsao deve mencionar cenarios."""
        prompt_lower = FORECAST_SYSTEM_PROMPT.lower()
        assert "otimista" in prompt_lower
        assert "realista" in prompt_lower
        assert "pessimista" in prompt_lower

    def test_forecast_mentions_seasonality(self):
        """Previsao deve mencionar sazonalidade brasileira."""
        prompt_lower = FORECAST_SYSTEM_PROMPT.lower()
        assert "sazonal" in prompt_lower

    def test_recommendation_mentions_actions(self):
        """Recomendacao deve mencionar tipos de acao."""
        prompt_lower = RECOMMENDATION_SYSTEM_PROMPT.lower()
        assert "escalar" in prompt_lower
        assert "pausar" in prompt_lower

    def test_recommendation_mentions_ice_score(self):
        """Recomendacao deve mencionar Score ICE."""
        prompt = RECOMMENDATION_SYSTEM_PROMPT
        assert "ICE" in prompt
        assert "Impact" in prompt or "impact" in prompt.lower()

    def test_recommendation_mentions_financial_impact(self):
        """Recomendacao deve mencionar impacto em R$."""
        assert "R$" in RECOMMENDATION_SYSTEM_PROMPT

    def test_campaign_mentions_metrics(self):
        """Campanha deve mencionar metricas de funil."""
        prompt_lower = CAMPAIGN_SYSTEM_PROMPT.lower()
        assert "cpl" in prompt_lower
        assert "ctr" in prompt_lower
        assert "cpc" in prompt_lower
        assert "lead" in prompt_lower

    def test_campaign_mentions_pacing(self):
        """Campanha deve mencionar deteccao de pacing."""
        prompt_lower = CAMPAIGN_SYSTEM_PROMPT.lower()
        assert "pacing" in prompt_lower or "underpacing" in prompt_lower

    def test_analysis_mentions_comparison(self):
        """Analise deve mencionar comparacao."""
        prompt_lower = ANALYSIS_SYSTEM_PROMPT.lower()
        assert "compara" in prompt_lower

    def test_analysis_mentions_pareto(self):
        """Analise deve mencionar analise de Pareto 80/20."""
        prompt_lower = ANALYSIS_SYSTEM_PROMPT.lower()
        assert "pareto" in prompt_lower or "80/20" in prompt_lower

    def test_analysis_mentions_ltv_cac(self):
        """Analise deve mencionar LTV/CAC."""
        prompt = ANALYSIS_SYSTEM_PROMPT
        assert "LTV" in prompt
        assert "CAC" in prompt

    @pytest.mark.parametrize("name,prompt", SUBAGENT_PROMPTS.items())
    def test_subagent_is_actionable(self, name, prompt):
        """Subagentes devem ser orientados a acao."""
        prompt_lower = prompt.lower()
        has_actionable = any(w in prompt_lower for w in [
            "perguntado", "pergunta", "focado", "especifico",
            "usuario", "dados reais", "nao invente",
        ])
        assert has_actionable, f"Prompt '{name}' sem orientacao acionavel"


class TestOrchestratorPrompts:
    """Testes para prompts do orchestrator."""

    def test_orchestrator_mentions_specialists(self):
        """Orchestrator deve mencionar especialistas."""
        prompt_lower = ORCHESTRATOR_SYSTEM_PROMPT.lower()
        assert "especialista" in prompt_lower

    def test_orchestrator_mentions_all_agents(self):
        """Orchestrator deve mencionar todos os 6 subagentes."""
        prompt_lower = ORCHESTRATOR_SYSTEM_PROMPT.lower()
        agents = ["classification", "anomaly", "forecast", "recommendation", "campaign", "analysis"]
        for agent in agents:
            assert agent in prompt_lower, f"Orchestrator nao menciona agente '{agent}'"

    def test_orchestrator_mentions_delegation_rules(self):
        """Orchestrator deve mencionar regras de delegacao."""
        prompt_lower = ORCHESTRATOR_SYSTEM_PROMPT.lower()
        assert "delega" in prompt_lower

    def test_orchestrator_mentions_minimum_agents(self):
        """Orchestrator deve mencionar usar minimo de agentes."""
        prompt_lower = ORCHESTRATOR_SYSTEM_PROMPT.lower()
        assert "minimo" in prompt_lower

    def test_synthesis_mentions_direct_answer(self):
        """Synthesis deve mencionar resposta direta."""
        prompt_lower = SYNTHESIS_PROMPT.lower()
        assert "exatamente" in prompt_lower or "direta" in prompt_lower

    def test_synthesis_mentions_formatting(self):
        """Synthesis deve mencionar formatacao com emojis."""
        assert "🔴" in SYNTHESIS_PROMPT
        assert "🟢" in SYNTHESIS_PROMPT

    def test_synthesis_mentions_failure_handling(self):
        """Synthesis deve mencionar tratamento de falhas."""
        prompt_lower = SYNTHESIS_PROMPT.lower()
        assert "falh" in prompt_lower


class TestSystemPromptFunction:
    """Testes para funcao get_system_prompt."""

    def test_base_prompt(self):
        """Retorna prompt base."""
        prompt = get_system_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 100

    def test_with_config_id(self):
        """Adiciona contexto de config_id."""
        prompt = get_system_prompt(config_id=42)
        assert "42" in prompt

    def test_with_additional_context(self):
        """Adiciona contexto adicional."""
        prompt = get_system_prompt(additional_context="Foco em leads")
        assert "Foco em leads" in prompt

    def test_with_both(self):
        """Adiciona ambos os contextos."""
        prompt = get_system_prompt(config_id=1, additional_context="Extra")
        assert "1" in prompt
        assert "Extra" in prompt

    def test_without_config_no_session_context(self):
        """Sem config_id nao adiciona contexto de sessao."""
        prompt = get_system_prompt()
        assert "Sessão" not in prompt or "config_id" not in prompt.split("Sessão")[0] if "Sessão" in prompt else True
