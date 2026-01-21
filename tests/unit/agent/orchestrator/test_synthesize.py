"""Testes para synthesize node."""
import os
import importlib.util

import pytest


# Carregar o modulo diretamente para evitar dependencias de __init__.py
def load_module_direct(module_name: str, relative_path: str):
    """Carrega modulo diretamente pelo path."""
    module_path = os.path.join(
        os.path.dirname(__file__),
        '..', '..', '..', '..',
        relative_path
    )
    module_path = os.path.abspath(module_path)

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestCalculateConfidenceScore:
    """Testes para a funcao calculate_confidence_score."""

    def test_confidence_all_success(self):
        """Todos os resultados com sucesso devem retornar 1.0."""
        synthesize_module = load_module_direct(
            "synthesize",
            "app/agent/orchestrator/nodes/synthesize.py"
        )

        agent_results = {
            "classification": {
                "agent_name": "classification",
                "success": True,
                "data": {"category": "performance"},
                "error": None,
                "duration_ms": 1000,
                "tool_calls": []
            },
            "anomaly": {
                "agent_name": "anomaly",
                "success": True,
                "data": {"anomalies": []},
                "error": None,
                "duration_ms": 1500,
                "tool_calls": []
            },
            "recommendation": {
                "agent_name": "recommendation",
                "success": True,
                "data": {"recommendations": ["optimize"]},
                "error": None,
                "duration_ms": 2000,
                "tool_calls": []
            }
        }

        confidence = synthesize_module.calculate_confidence_score(agent_results)

        assert confidence == 1.0

    def test_confidence_all_failure(self):
        """Todos os resultados com falha devem retornar 0.0."""
        synthesize_module = load_module_direct(
            "synthesize",
            "app/agent/orchestrator/nodes/synthesize.py"
        )

        agent_results = {
            "classification": {
                "agent_name": "classification",
                "success": False,
                "data": None,
                "error": "API Error",
                "duration_ms": 1000,
                "tool_calls": []
            },
            "anomaly": {
                "agent_name": "anomaly",
                "success": False,
                "data": None,
                "error": "Timeout",
                "duration_ms": 5000,
                "tool_calls": []
            }
        }

        confidence = synthesize_module.calculate_confidence_score(agent_results)

        assert confidence == 0.0

    def test_confidence_mixed(self):
        """Resultados mistos devem retornar score proporcional."""
        synthesize_module = load_module_direct(
            "synthesize",
            "app/agent/orchestrator/nodes/synthesize.py"
        )

        agent_results = {
            "classification": {
                "agent_name": "classification",
                "success": True,
                "data": {"category": "performance"},
                "error": None,
                "duration_ms": 1000,
                "tool_calls": []
            },
            "anomaly": {
                "agent_name": "anomaly",
                "success": False,
                "data": None,
                "error": "API Error",
                "duration_ms": 1500,
                "tool_calls": []
            },
            "recommendation": {
                "agent_name": "recommendation",
                "success": True,
                "data": {"recommendations": ["optimize"]},
                "error": None,
                "duration_ms": 2000,
                "tool_calls": []
            },
            "forecast": {
                "agent_name": "forecast",
                "success": False,
                "data": None,
                "error": "Timeout",
                "duration_ms": 5000,
                "tool_calls": []
            }
        }

        confidence = synthesize_module.calculate_confidence_score(agent_results)

        # 2 sucessos / 4 total = 0.5
        assert confidence == 0.5

    def test_confidence_empty(self):
        """Sem resultados deve retornar 0.0."""
        synthesize_module = load_module_direct(
            "synthesize",
            "app/agent/orchestrator/nodes/synthesize.py"
        )

        agent_results = {}

        confidence = synthesize_module.calculate_confidence_score(agent_results)

        assert confidence == 0.0

    def test_confidence_single_success(self):
        """Unico resultado com sucesso deve retornar 1.0."""
        synthesize_module = load_module_direct(
            "synthesize",
            "app/agent/orchestrator/nodes/synthesize.py"
        )

        agent_results = {
            "classification": {
                "agent_name": "classification",
                "success": True,
                "data": {"category": "performance"},
                "error": None,
                "duration_ms": 1000,
                "tool_calls": []
            }
        }

        confidence = synthesize_module.calculate_confidence_score(agent_results)

        assert confidence == 1.0

    def test_confidence_single_failure(self):
        """Unico resultado com falha deve retornar 0.0."""
        synthesize_module = load_module_direct(
            "synthesize",
            "app/agent/orchestrator/nodes/synthesize.py"
        )

        agent_results = {
            "classification": {
                "agent_name": "classification",
                "success": False,
                "data": None,
                "error": "Error",
                "duration_ms": 1000,
                "tool_calls": []
            }
        }

        confidence = synthesize_module.calculate_confidence_score(agent_results)

        assert confidence == 0.0


class TestFormatAgentSection:
    """Testes para a funcao format_agent_section."""

    def test_format_success_result(self):
        """Resultado bem-sucedido com dados deve formatar conteudo corretamente."""
        synthesize_module = load_module_direct(
            "synthesize",
            "app/agent/orchestrator/nodes/synthesize.py"
        )

        result = {
            "agent_name": "anomaly",
            "success": True,
            "data": {"anomalies": ["high_cpc", "low_ctr"], "severity": "high"},
            "error": None,
            "duration_ms": 1500,
            "tool_calls": []
        }

        section = synthesize_module.format_agent_section("anomaly", result)

        # Deve conter o template de alertas
        assert "Alertas e Problemas" in section
        # Deve conter os dados
        assert "high_cpc" in section or "anomalies" in section

    def test_format_error_result(self):
        """Resultado com erro deve mostrar mensagem de erro."""
        synthesize_module = load_module_direct(
            "synthesize",
            "app/agent/orchestrator/nodes/synthesize.py"
        )

        result = {
            "agent_name": "forecast",
            "success": False,
            "data": None,
            "error": "API timeout after 5000ms",
            "duration_ms": 5000,
            "tool_calls": []
        }

        section = synthesize_module.format_agent_section("forecast", result)

        # Deve indicar erro
        assert "erro" in section.lower() or "error" in section.lower()
        # Deve conter a mensagem de erro
        assert "API timeout" in section or "5000ms" in section

    def test_format_uses_template(self):
        """Deve usar o template correto para cada agente."""
        synthesize_module = load_module_direct(
            "synthesize",
            "app/agent/orchestrator/nodes/synthesize.py"
        )

        result = {
            "agent_name": "recommendation",
            "success": True,
            "data": {"recommendations": ["Aumentar budget", "Pausar ad"]},
            "error": None,
            "duration_ms": 2000,
            "tool_calls": []
        }

        section = synthesize_module.format_agent_section("recommendation", result)

        # Deve usar o template de recomendacoes
        assert "Recomendacoes" in section

    def test_format_unknown_agent(self):
        """Agente desconhecido deve usar formato padrao."""
        synthesize_module = load_module_direct(
            "synthesize",
            "app/agent/orchestrator/nodes/synthesize.py"
        )

        result = {
            "agent_name": "custom_agent",
            "success": True,
            "data": {"custom": "data"},
            "error": None,
            "duration_ms": 1000,
            "tool_calls": []
        }

        section = synthesize_module.format_agent_section("custom_agent", result)

        # Deve retornar algum conteudo formatado
        assert len(section) > 0
        # Deve conter dados ou nome do agente
        assert "custom" in section.lower() or "agent" in section.lower()


class TestOrderResultsByPriority:
    """Testes para a funcao order_results_by_priority."""

    def test_order_by_priority(self):
        """Resultados devem ser ordenados por prioridade."""
        synthesize_module = load_module_direct(
            "synthesize",
            "app/agent/orchestrator/nodes/synthesize.py"
        )

        agent_results = {
            "forecast": {
                "agent_name": "forecast",
                "success": True,
                "data": {},
                "error": None,
                "duration_ms": 1000,
                "tool_calls": []
            },
            "anomaly": {
                "agent_name": "anomaly",
                "success": True,
                "data": {},
                "error": None,
                "duration_ms": 1000,
                "tool_calls": []
            },
            "classification": {
                "agent_name": "classification",
                "success": True,
                "data": {},
                "error": None,
                "duration_ms": 1000,
                "tool_calls": []
            },
            "recommendation": {
                "agent_name": "recommendation",
                "success": True,
                "data": {},
                "error": None,
                "duration_ms": 1000,
                "tool_calls": []
            }
        }

        ordered = synthesize_module.order_results_by_priority(agent_results)

        # Extrair nomes na ordem
        ordered_names = [name for name, _ in ordered]

        # Verificar que anomaly vem antes de recommendation
        assert ordered_names.index("anomaly") < ordered_names.index("recommendation")
        # Verificar que recommendation vem antes de classification
        assert ordered_names.index("recommendation") < ordered_names.index("classification")
        # Verificar que classification vem antes de forecast
        assert ordered_names.index("classification") < ordered_names.index("forecast")

    def test_order_unknown_agents_last(self):
        """Agentes desconhecidos devem ficar no final."""
        synthesize_module = load_module_direct(
            "synthesize",
            "app/agent/orchestrator/nodes/synthesize.py"
        )

        agent_results = {
            "unknown_agent": {
                "agent_name": "unknown_agent",
                "success": True,
                "data": {},
                "error": None,
                "duration_ms": 1000,
                "tool_calls": []
            },
            "anomaly": {
                "agent_name": "anomaly",
                "success": True,
                "data": {},
                "error": None,
                "duration_ms": 1000,
                "tool_calls": []
            },
            "custom_agent": {
                "agent_name": "custom_agent",
                "success": True,
                "data": {},
                "error": None,
                "duration_ms": 1000,
                "tool_calls": []
            }
        }

        ordered = synthesize_module.order_results_by_priority(agent_results)

        # Extrair nomes na ordem
        ordered_names = [name for name, _ in ordered]

        # Anomaly deve vir primeiro (prioridade 1)
        assert ordered_names[0] == "anomaly"
        # Agentes desconhecidos devem vir depois
        assert ordered_names.index("unknown_agent") > ordered_names.index("anomaly")
        assert ordered_names.index("custom_agent") > ordered_names.index("anomaly")

    def test_order_empty(self):
        """Resultados vazios devem retornar lista vazia."""
        synthesize_module = load_module_direct(
            "synthesize",
            "app/agent/orchestrator/nodes/synthesize.py"
        )

        agent_results = {}

        ordered = synthesize_module.order_results_by_priority(agent_results)

        assert ordered == []


class TestSynthesizeResponse:
    """Testes para a funcao synthesize_response."""

    def test_synthesize_single_result(self):
        """Unico resultado deve ser formatado corretamente."""
        synthesize_module = load_module_direct(
            "synthesize",
            "app/agent/orchestrator/nodes/synthesize.py"
        )

        agent_results = {
            "classification": {
                "agent_name": "classification",
                "success": True,
                "data": {"category": "performance", "score": 0.85},
                "error": None,
                "duration_ms": 1000,
                "tool_calls": []
            }
        }

        response = synthesize_module.synthesize_response(agent_results)

        # Deve retornar string nao vazia
        assert isinstance(response, str)
        assert len(response) > 0
        # Deve conter informacoes sobre performance
        assert "Performance" in response or "classification" in response.lower()

    def test_synthesize_multiple_ordered(self):
        """Multiplos resultados devem aparecer na ordem de prioridade."""
        synthesize_module = load_module_direct(
            "synthesize",
            "app/agent/orchestrator/nodes/synthesize.py"
        )

        agent_results = {
            "recommendation": {
                "agent_name": "recommendation",
                "success": True,
                "data": {"recommendations": ["Optimize budget"]},
                "error": None,
                "duration_ms": 2000,
                "tool_calls": []
            },
            "anomaly": {
                "agent_name": "anomaly",
                "success": True,
                "data": {"anomalies": ["High CPC"]},
                "error": None,
                "duration_ms": 1500,
                "tool_calls": []
            }
        }

        response = synthesize_module.synthesize_response(agent_results)

        # Alertas (anomaly) deve vir antes de Recomendacoes
        alertas_pos = response.find("Alertas") if "Alertas" in response else response.find("anomaly")
        recomendacoes_pos = response.find("Recomendacoes") if "Recomendacoes" in response else response.find("recommendation")

        # Se ambos estao presentes, alertas deve vir primeiro
        if alertas_pos >= 0 and recomendacoes_pos >= 0:
            assert alertas_pos < recomendacoes_pos

    def test_synthesize_empty(self):
        """Resultados vazios devem retornar string vazia."""
        synthesize_module = load_module_direct(
            "synthesize",
            "app/agent/orchestrator/nodes/synthesize.py"
        )

        agent_results = {}

        response = synthesize_module.synthesize_response(agent_results)

        assert response == ""

    def test_synthesize_includes_all_sections(self):
        """Todos os agentes devem estar incluidos na resposta."""
        synthesize_module = load_module_direct(
            "synthesize",
            "app/agent/orchestrator/nodes/synthesize.py"
        )

        agent_results = {
            "classification": {
                "agent_name": "classification",
                "success": True,
                "data": {"category": "performance"},
                "error": None,
                "duration_ms": 1000,
                "tool_calls": []
            },
            "anomaly": {
                "agent_name": "anomaly",
                "success": True,
                "data": {"anomalies": ["High CPC"]},
                "error": None,
                "duration_ms": 1500,
                "tool_calls": []
            },
            "recommendation": {
                "agent_name": "recommendation",
                "success": True,
                "data": {"recommendations": ["Optimize"]},
                "error": None,
                "duration_ms": 2000,
                "tool_calls": []
            },
            "forecast": {
                "agent_name": "forecast",
                "success": True,
                "data": {"prediction": 1500},
                "error": None,
                "duration_ms": 1800,
                "tool_calls": []
            }
        }

        response = synthesize_module.synthesize_response(agent_results)

        # Deve conter secoes de todos os agentes (com templates ou nomes)
        # Verificar que cada agente contribuiu para a resposta
        assert len(response) > 100  # Resposta substancial com 4 agentes


class TestSynthesizeNode:
    """Testes para a funcao synthesize (no principal)."""

    def test_synthesize_node_returns_dict(self):
        """Deve retornar dict com formato correto."""
        synthesize_module = load_module_direct(
            "synthesize",
            "app/agent/orchestrator/nodes/synthesize.py"
        )
        state_module = load_module_direct(
            "state",
            "app/agent/orchestrator/state.py"
        )

        state = state_module.create_initial_orchestrator_state(
            config_id=1, user_id=1, thread_id="test"
        )
        state["agent_results"] = {
            "classification": {
                "agent_name": "classification",
                "success": True,
                "data": {"category": "performance"},
                "error": None,
                "duration_ms": 1000,
                "tool_calls": []
            }
        }

        result = synthesize_module.synthesize(state)

        # Deve retornar dict
        assert isinstance(result, dict)
        # Deve conter synthesized_response
        assert "synthesized_response" in result
        # Deve conter confidence_score
        assert "confidence_score" in result

    def test_synthesize_node_calculates_confidence(self):
        """Deve calcular confidence_score corretamente."""
        synthesize_module = load_module_direct(
            "synthesize",
            "app/agent/orchestrator/nodes/synthesize.py"
        )
        state_module = load_module_direct(
            "state",
            "app/agent/orchestrator/state.py"
        )

        state = state_module.create_initial_orchestrator_state(
            config_id=1, user_id=1, thread_id="test"
        )
        state["agent_results"] = {
            "classification": {
                "agent_name": "classification",
                "success": True,
                "data": {"category": "performance"},
                "error": None,
                "duration_ms": 1000,
                "tool_calls": []
            },
            "anomaly": {
                "agent_name": "anomaly",
                "success": False,
                "data": None,
                "error": "Timeout",
                "duration_ms": 5000,
                "tool_calls": []
            }
        }

        result = synthesize_module.synthesize(state)

        # 1 sucesso / 2 total = 0.5
        assert result["confidence_score"] == 0.5

    def test_synthesize_node_empty_results(self):
        """Deve tratar agent_results vazio corretamente."""
        synthesize_module = load_module_direct(
            "synthesize",
            "app/agent/orchestrator/nodes/synthesize.py"
        )
        state_module = load_module_direct(
            "state",
            "app/agent/orchestrator/state.py"
        )

        state = state_module.create_initial_orchestrator_state(
            config_id=1, user_id=1, thread_id="test"
        )
        # agent_results vazio (padrao do estado inicial)

        result = synthesize_module.synthesize(state)

        # Deve retornar dict valido
        assert isinstance(result, dict)
        # Deve ter synthesized_response vazio
        assert result["synthesized_response"] == ""
        # Deve ter confidence_score 0.0
        assert result["confidence_score"] == 0.0


class TestSynthesizeImport:
    """Testes de importacao do modulo synthesize."""

    def test_synthesize_import(self):
        """synthesize deve ser importavel."""
        synthesize_module = load_module_direct(
            "synthesize",
            "app/agent/orchestrator/nodes/synthesize.py"
        )
        assert synthesize_module.synthesize is not None

    def test_calculate_confidence_score_import(self):
        """calculate_confidence_score deve ser importavel."""
        synthesize_module = load_module_direct(
            "synthesize",
            "app/agent/orchestrator/nodes/synthesize.py"
        )
        assert synthesize_module.calculate_confidence_score is not None

    def test_format_agent_section_import(self):
        """format_agent_section deve ser importavel."""
        synthesize_module = load_module_direct(
            "synthesize",
            "app/agent/orchestrator/nodes/synthesize.py"
        )
        assert synthesize_module.format_agent_section is not None

    def test_order_results_by_priority_import(self):
        """order_results_by_priority deve ser importavel."""
        synthesize_module = load_module_direct(
            "synthesize",
            "app/agent/orchestrator/nodes/synthesize.py"
        )
        assert synthesize_module.order_results_by_priority is not None

    def test_synthesize_response_import(self):
        """synthesize_response deve ser importavel."""
        synthesize_module = load_module_direct(
            "synthesize",
            "app/agent/orchestrator/nodes/synthesize.py"
        )
        assert synthesize_module.synthesize_response is not None

    def test_constants_import(self):
        """Constantes SYNTHESIS_PRIORITY e SECTION_TEMPLATES devem existir."""
        synthesize_module = load_module_direct(
            "synthesize",
            "app/agent/orchestrator/nodes/synthesize.py"
        )
        assert hasattr(synthesize_module, "SYNTHESIS_PRIORITY")
        assert hasattr(synthesize_module, "SECTION_TEMPLATES")

    def test_all_functions_are_callable(self):
        """Todas as funcoes exportadas devem ser callable."""
        synthesize_module = load_module_direct(
            "synthesize",
            "app/agent/orchestrator/nodes/synthesize.py"
        )
        assert callable(synthesize_module.synthesize)
        assert callable(synthesize_module.calculate_confidence_score)
        assert callable(synthesize_module.format_agent_section)
        assert callable(synthesize_module.order_results_by_priority)
        assert callable(synthesize_module.synthesize_response)
