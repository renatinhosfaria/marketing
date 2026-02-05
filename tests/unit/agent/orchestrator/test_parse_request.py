"""Testes para parse_request node."""
import pytest
from unittest.mock import Mock, patch, AsyncMock


class TestParseRequest:
    """Testes para o no parse_request."""

    def test_parse_request_import(self):
        """parse_request deve ser importavel."""
        from projects.agent.orchestrator.nodes.parse_request import parse_request
        assert parse_request is not None

    def test_detect_intent_analyze(self):
        """detect_intent deve identificar intencao de analise."""
        from projects.agent.orchestrator.nodes.parse_request import detect_intent

        assert detect_intent("Como esta a performance?") == "analyze_performance"
        assert detect_intent("Analise minhas campanhas") == "analyze_performance"
        assert detect_intent("Como estao os resultados?") == "analyze_performance"

    def test_detect_intent_problems(self):
        """detect_intent deve identificar busca por problemas."""
        from projects.agent.orchestrator.nodes.parse_request import detect_intent

        assert detect_intent("Tem algum problema?") == "find_problems"
        assert detect_intent("Quais anomalias existem?") == "find_problems"
        assert detect_intent("O que esta errado?") == "find_problems"

    def test_detect_intent_recommendations(self):
        """detect_intent deve identificar pedido de recomendacoes."""
        from projects.agent.orchestrator.nodes.parse_request import detect_intent

        assert detect_intent("O que devo fazer?") == "get_recommendations"
        assert detect_intent("Quais sao suas recomendacoes?") == "get_recommendations"
        assert detect_intent("Qual campanha escalar?") == "get_recommendations"

    def test_detect_intent_forecast(self):
        """detect_intent deve identificar pedido de previsao."""
        from projects.agent.orchestrator.nodes.parse_request import detect_intent

        assert detect_intent("Qual a previsao para semana?") == "predict_future"
        assert detect_intent("Como vai ser o CPL?") == "predict_future"
        assert detect_intent("Forecast de leads") == "predict_future"

    def test_detect_intent_compare(self):
        """detect_intent deve identificar comparacao."""
        from projects.agent.orchestrator.nodes.parse_request import detect_intent

        assert detect_intent("Compare campanha A com B") == "compare_campaigns"
        assert detect_intent("Qual e melhor entre X e Y?") == "compare_campaigns"

    def test_detect_intent_full_report(self):
        """detect_intent deve identificar relatorio completo."""
        from projects.agent.orchestrator.nodes.parse_request import detect_intent

        assert detect_intent("Me de um relatorio completo") == "full_report"
        assert detect_intent("Resumo geral de tudo") == "full_report"

    def test_detect_intent_fallback(self):
        """detect_intent deve retornar general para mensagens genericas."""
        from projects.agent.orchestrator.nodes.parse_request import detect_intent

        assert detect_intent("Ola") == "general"
        assert detect_intent("Obrigado") == "general"

    def test_parse_request_sets_user_intent(self):
        """parse_request deve retornar user_intent no estado."""
        from projects.agent.orchestrator.nodes.parse_request import parse_request
        from langchain_core.messages import HumanMessage

        state = {
            "messages": [HumanMessage(content="Como esta a performance?")]
        }

        result = parse_request(state)

        assert result.get("user_intent") == "analyze_performance"
