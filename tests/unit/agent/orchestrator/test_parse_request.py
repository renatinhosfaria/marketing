"""Testes para o nó parse_request do Orchestrator.

Testa a detecção de intenção do usuário e extração de referências a campanhas.
"""
import os
import sys
import importlib.util

import pytest
from langchain_core.messages import HumanMessage, AIMessage


# Carregar o módulo parse_request diretamente sem passar pelo __init__.py
# Isso evita dependências de outros módulos que podem ter requirements não instalados
parse_request_path = os.path.join(
    os.path.dirname(__file__),
    '..', '..', '..', '..',
    'app', 'agent', 'orchestrator', 'nodes', 'parse_request.py'
)
parse_request_path = os.path.abspath(parse_request_path)

spec = importlib.util.spec_from_file_location("parse_request_module", parse_request_path)
parse_request_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parse_request_module)

# Exportar os símbolos do módulo
INTENT_PATTERNS = parse_request_module.INTENT_PATTERNS
detect_intent = parse_request_module.detect_intent
extract_campaign_references = parse_request_module.extract_campaign_references
parse_request = parse_request_module.parse_request


class TestImports:
    """Testes de importação do módulo."""

    def test_import_parse_request_module(self):
        """Verifica que o módulo pode ser importado."""
        assert parse_request_module is not None

    def test_import_intent_patterns(self):
        """Verifica que INTENT_PATTERNS está disponível."""
        assert isinstance(INTENT_PATTERNS, dict)
        assert len(INTENT_PATTERNS) > 0

    def test_import_detect_intent(self):
        """Verifica que detect_intent está disponível."""
        assert callable(detect_intent)

    def test_import_extract_campaign_references(self):
        """Verifica que extract_campaign_references está disponível."""
        assert callable(extract_campaign_references)

    def test_import_parse_request_function(self):
        """Verifica que parse_request está disponível."""
        assert callable(parse_request)


class TestIntentPatterns:
    """Testes para a estrutura de INTENT_PATTERNS."""

    def test_intent_patterns_has_required_intents(self):
        """Verifica que todos os intents necessários estão definidos."""
        required_intents = [
            "analyze_performance",
            "find_problems",
            "get_recommendations",
            "predict_future",
            "compare_campaigns",
            "full_report",
        ]

        for intent in required_intents:
            assert intent in INTENT_PATTERNS, f"Intent '{intent}' não encontrado"

    def test_intent_patterns_values_are_lists(self):
        """Verifica que cada intent tem uma lista de padrões."""
        for intent, patterns in INTENT_PATTERNS.items():
            assert isinstance(patterns, list), f"Padrões de '{intent}' não é lista"
            assert len(patterns) > 0, f"Intent '{intent}' não tem padrões"


class TestDetectIntent:
    """Testes para a função detect_intent."""

    def test_detect_analyze_performance_como_esta(self):
        """Detecta intent de performance com 'como está'."""
        assert detect_intent("como está minha campanha?") == "analyze_performance"

    def test_detect_analyze_performance_analise(self):
        """Detecta intent de performance com 'analise'."""
        assert detect_intent("analise os resultados") == "analyze_performance"

    def test_detect_analyze_performance_metricas(self):
        """Detecta intent de performance com 'métricas'."""
        assert detect_intent("quero ver as métricas") == "analyze_performance"

    def test_detect_analyze_performance_desempenho(self):
        """Detecta intent de performance com 'desempenho'."""
        assert detect_intent("qual o desempenho?") == "analyze_performance"

    def test_detect_find_problems_problema(self):
        """Detecta intent de problemas com 'problema'."""
        assert detect_intent("tem algum problema?") == "find_problems"

    def test_detect_find_problems_anomalia(self):
        """Detecta intent de problemas com 'anomalia'."""
        assert detect_intent("detectou alguma anomalia?") == "find_problems"

    def test_detect_find_problems_errado(self):
        """Detecta intent de problemas com 'errado'."""
        assert detect_intent("o que está errado?") == "find_problems"

    def test_detect_get_recommendations_recomenda(self):
        """Detecta intent de recomendações com 'recomenda'."""
        assert detect_intent("o que você recomenda?") == "get_recommendations"

    def test_detect_get_recommendations_sugestao(self):
        """Detecta intent de recomendações com 'sugestão'."""
        assert detect_intent("alguma sugestão?") == "get_recommendations"

    def test_detect_get_recommendations_o_que_fazer(self):
        """Detecta intent de recomendações com 'o que fazer'."""
        assert detect_intent("o que fazer para melhorar?") == "get_recommendations"

    def test_detect_get_recommendations_otimizar(self):
        """Detecta intent de recomendações com 'otimizar'."""
        assert detect_intent("como otimizar a campanha?") == "get_recommendations"

    def test_detect_predict_future_previsao(self):
        """Detecta intent de previsão com 'previsão'."""
        assert detect_intent("qual a previsão para o mês?") == "predict_future"

    def test_detect_predict_future_forecast(self):
        """Detecta intent de previsão com 'forecast'."""
        assert detect_intent("faça um forecast") == "predict_future"

    def test_detect_predict_future_futuro(self):
        """Detecta intent de previsão com 'futuro'."""
        assert detect_intent("como será no futuro?") == "predict_future"

    def test_detect_predict_future_proximo(self):
        """Detecta intent de previsão com 'próximo'."""
        assert detect_intent("o que esperar no próximo mês?") == "predict_future"

    def test_detect_compare_campaigns_compare(self):
        """Detecta intent de comparação com 'compare'."""
        assert detect_intent("compare as campanhas") == "compare_campaigns"

    def test_detect_compare_campaigns_versus(self):
        """Detecta intent de comparação com 'versus'."""
        assert detect_intent("campanha A versus campanha B") == "compare_campaigns"

    def test_detect_compare_campaigns_vs(self):
        """Detecta intent de comparação com 'vs'."""
        assert detect_intent("campanha A vs campanha B") == "compare_campaigns"

    def test_detect_full_report_relatorio_completo(self):
        """Detecta intent de relatório completo."""
        assert detect_intent("quero um relatório completo") == "full_report"

    def test_detect_full_report_resumo_geral(self):
        """Detecta intent de relatório com 'resumo geral'."""
        assert detect_intent("me dê um resumo geral") == "full_report"

    def test_detect_full_report_visao_geral(self):
        """Detecta intent de relatório com 'visão geral'."""
        assert detect_intent("preciso de uma visão geral") == "full_report"

    def test_detect_general_fallback(self):
        """Retorna 'general' para mensagens sem padrão específico."""
        assert detect_intent("olá") == "general"
        assert detect_intent("obrigado") == "general"
        assert detect_intent("tudo bem?") == "general"

    def test_detect_intent_case_insensitive(self):
        """Detecta intent independente de maiúsculas/minúsculas."""
        assert detect_intent("COMO ESTÁ a campanha?") == "analyze_performance"
        assert detect_intent("Qual A PREVISÃO?") == "predict_future"


class TestExtractCampaignReferences:
    """Testes para a função extract_campaign_references."""

    def test_extract_campaign_with_keyword(self):
        """Extrai referência com palavra 'campanha'."""
        refs = extract_campaign_references("como está a campanha Black Friday?")
        assert "Black Friday" in refs

    def test_extract_campaign_quoted_single(self):
        """Extrai referência entre aspas simples."""
        refs = extract_campaign_references("analise a campanha 'Promoção Verão'")
        assert "Promoção Verão" in refs

    def test_extract_campaign_quoted_double(self):
        """Extrai referência entre aspas duplas."""
        refs = extract_campaign_references('compare "Campanha A" com "Campanha B"')
        assert "Campanha A" in refs
        assert "Campanha B" in refs

    def test_extract_no_campaigns(self):
        """Retorna lista vazia quando não há referências."""
        refs = extract_campaign_references("como estão as métricas?")
        assert refs == []

    def test_extract_removes_duplicates(self):
        """Remove duplicatas da lista de referências."""
        refs = extract_campaign_references(
            'analise a campanha "Promo" e compare com campanha "Promo"'
        )
        # Deve ter apenas uma referência
        assert refs.count("Promo") == 1


class TestParseRequest:
    """Testes para a função parse_request."""

    def test_parse_request_with_human_message(self):
        """Processa estado com HumanMessage."""
        state = {
            "messages": [HumanMessage(content="como está a performance?")],
            "thread_id": "test-123",
            "config_id": 1,
            "user_id": 1,
        }

        result = parse_request(state)

        assert result["intent"] == "analyze_performance"
        assert result["campaign_refs"] == []
        assert result["original_message"] == "como está a performance?"

    def test_parse_request_with_dict_message(self):
        """Processa estado com mensagem em formato dict."""
        state = {
            "messages": [
                {"role": "user", "content": "detectou algum problema?"}
            ],
            "thread_id": "test-123",
            "config_id": 1,
            "user_id": 1,
        }

        result = parse_request(state)

        assert result["intent"] == "find_problems"
        assert "original_message" in result

    def test_parse_request_uses_last_human_message(self):
        """Usa a última mensagem do usuário quando há múltiplas."""
        state = {
            "messages": [
                HumanMessage(content="olá"),
                AIMessage(content="Olá! Como posso ajudar?"),
                HumanMessage(content="qual a previsão?"),
            ],
            "thread_id": "test-123",
            "config_id": 1,
            "user_id": 1,
        }

        result = parse_request(state)

        assert result["intent"] == "predict_future"
        assert result["original_message"] == "qual a previsão?"

    def test_parse_request_extracts_campaign_refs(self):
        """Extrai referências a campanhas da mensagem."""
        state = {
            "messages": [
                HumanMessage(content='compare a campanha "Verão" com "Inverno"')
            ],
            "thread_id": "test-123",
            "config_id": 1,
            "user_id": 1,
        }

        result = parse_request(state)

        assert result["intent"] == "compare_campaigns"
        assert "Verão" in result["campaign_refs"]
        assert "Inverno" in result["campaign_refs"]

    def test_parse_request_empty_messages(self):
        """Retorna general quando não há mensagens."""
        state = {
            "messages": [],
            "thread_id": "test-123",
            "config_id": 1,
            "user_id": 1,
        }

        result = parse_request(state)

        assert result["intent"] == "general"
        assert result["campaign_refs"] == []

    def test_parse_request_no_user_message(self):
        """Retorna general quando não há mensagem do usuário."""
        state = {
            "messages": [AIMessage(content="Como posso ajudar?")],
            "thread_id": "test-123",
            "config_id": 1,
            "user_id": 1,
        }

        result = parse_request(state)

        assert result["intent"] == "general"
        assert result["campaign_refs"] == []

    def test_parse_request_returns_dict(self):
        """Verifica que retorna um dicionário com as chaves esperadas."""
        state = {
            "messages": [HumanMessage(content="teste")],
            "thread_id": "test-123",
            "config_id": 1,
            "user_id": 1,
        }

        result = parse_request(state)

        assert isinstance(result, dict)
        assert "intent" in result
        assert "campaign_refs" in result
        assert "original_message" in result
