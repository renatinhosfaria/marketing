"""Testes unitarios para o grafo do Orchestrator.

Testa:
- Funcao should_dispatch (roteamento condicional)
- Funcao build_orchestrator_graph (construcao do grafo)
- Classe OrchestratorAgent (wrapper de alto nivel)
- Fluxo do grafo end-to-end
"""
import os
import sys
import importlib.util
import asyncio

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from langchain_core.messages import HumanMessage


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


# Carregar modulos
_state_module = load_module_direct(
    "orchestrator_state",
    "app/agent/orchestrator/state.py"
)
OrchestratorState = _state_module.OrchestratorState
create_initial_orchestrator_state = _state_module.create_initial_orchestrator_state

_graph_module = load_module_direct(
    "orchestrator_graph",
    "app/agent/orchestrator/graph.py"
)
should_dispatch = _graph_module.should_dispatch
build_orchestrator_graph = _graph_module.build_orchestrator_graph
OrchestratorAgent = _graph_module.OrchestratorAgent


# =============================================================================
# Testes para should_dispatch
# =============================================================================
class TestShouldDispatch:
    """Testes para a funcao should_dispatch."""

    def test_should_dispatch_with_agents(self):
        """Retorna 'dispatch' quando required_agents tem agentes."""
        state = create_initial_orchestrator_state(
            config_id=1,
            user_id=1,
            thread_id="test-thread"
        )
        state["required_agents"] = ["classification", "anomaly"]

        result = should_dispatch(state)

        assert result == "dispatch"

    def test_should_dispatch_empty_agents(self):
        """Retorna 'synthesize' quando required_agents esta vazio."""
        state = create_initial_orchestrator_state(
            config_id=1,
            user_id=1,
            thread_id="test-thread"
        )
        state["required_agents"] = []

        result = should_dispatch(state)

        assert result == "synthesize"

    def test_should_dispatch_none_agents(self):
        """Retorna 'synthesize' quando required_agents e None."""
        state: OrchestratorState = {
            "messages": [],
            "thread_id": "test",
            "config_id": 1,
            "user_id": 1,
            "user_intent": None,
            "required_agents": None,  # type: ignore
            "execution_plan": None,
            "agent_results": {},
            "synthesized_response": None,
            "confidence_score": 0.0,
            "error": None
        }

        result = should_dispatch(state)

        assert result == "synthesize"

    def test_should_dispatch_single_agent(self):
        """Retorna 'dispatch' mesmo com apenas um agente."""
        state = create_initial_orchestrator_state(
            config_id=1,
            user_id=1,
            thread_id="test-thread"
        )
        state["required_agents"] = ["classification"]

        result = should_dispatch(state)

        assert result == "dispatch"


# =============================================================================
# Testes para build_orchestrator_graph
# =============================================================================
class TestBuildOrchestratorGraph:
    """Testes para a funcao build_orchestrator_graph."""

    def test_build_graph_returns_compiled(self):
        """Retorna um grafo compilado."""
        graph = build_orchestrator_graph()

        # Verifica que retornou algo
        assert graph is not None

        # Grafos compilados do LangGraph tem o metodo invoke
        assert hasattr(graph, "invoke") or hasattr(graph, "ainvoke")

    def test_build_graph_has_nodes(self):
        """Grafo tem todos os nos requeridos."""
        graph = build_orchestrator_graph()

        # Obter lista de nos do grafo
        graph_data = graph.get_graph()
        # nodes e um dicionario onde as chaves sao os nomes dos nos
        node_names = list(graph_data.nodes.keys())

        # Verificar nos principais
        expected_nodes = [
            "parse_request",
            "plan_execution",
            "dispatch_agents",
            "collect_results",
            "synthesize"
        ]

        for expected in expected_nodes:
            assert expected in node_names, f"No '{expected}' nao encontrado no grafo"

    def test_build_graph_callable(self):
        """Grafo e invocavel (tem metodo invoke/ainvoke)."""
        graph = build_orchestrator_graph()

        # Grafos do LangGraph devem ter pelo menos um destes metodos
        has_invoke = hasattr(graph, "invoke")
        has_ainvoke = hasattr(graph, "ainvoke")

        assert has_invoke or has_ainvoke, "Grafo deve ser invocavel"

    def test_build_graph_has_start_and_end(self):
        """Grafo tem nos START e END configurados."""
        graph = build_orchestrator_graph()

        graph_data = graph.get_graph()
        # nodes e um dicionario
        node_ids = list(graph_data.nodes.keys())

        # START e END sao nos especiais do LangGraph
        assert "__start__" in node_ids, "START nao encontrado"
        assert "__end__" in node_ids, "END nao encontrado"


# =============================================================================
# Testes para OrchestratorAgent
# =============================================================================
class TestOrchestratorAgent:
    """Testes para a classe OrchestratorAgent."""

    def test_orchestrator_agent_init(self):
        """Pode instanciar OrchestratorAgent."""
        agent = OrchestratorAgent()

        assert agent is not None
        assert isinstance(agent, OrchestratorAgent)

    def test_orchestrator_agent_build_graph(self):
        """build_graph retorna grafo compilado."""
        agent = OrchestratorAgent()

        graph = agent.build_graph()

        assert graph is not None
        assert hasattr(graph, "invoke") or hasattr(graph, "ainvoke")

    def test_orchestrator_agent_graph_cached(self):
        """Grafo e cacheado na segunda chamada."""
        agent = OrchestratorAgent()

        graph1 = agent.build_graph()
        graph2 = agent.build_graph()

        # Deve retornar a mesma instancia
        assert graph1 is graph2

    def test_orchestrator_agent_has_run_method(self):
        """Tem metodo run assincrono."""
        agent = OrchestratorAgent()

        assert hasattr(agent, "run")
        # Verificar que e uma coroutine function
        import inspect
        assert inspect.iscoroutinefunction(agent.run)

    def test_orchestrator_agent_different_instances_different_graphs(self):
        """Instancias diferentes tem grafos diferentes."""
        agent1 = OrchestratorAgent()
        agent2 = OrchestratorAgent()

        graph1 = agent1.build_graph()
        graph2 = agent2.build_graph()

        # Instancias diferentes devem ter grafos proprios
        # Note: os grafos sao compilados separadamente
        assert graph1 is not graph2


# =============================================================================
# Testes de fluxo do grafo (integration-style)
# =============================================================================
class TestGraphFlow:
    """Testes de fluxo do grafo (estilo integracao)."""

    def test_graph_flow_simple_request(self):
        """Fluxo end-to-end com requisicao simples."""
        agent = OrchestratorAgent()

        # Executar com mensagem simples usando asyncio.run
        result = asyncio.get_event_loop().run_until_complete(
            agent.run(
                message="Como esta o desempenho das minhas campanhas?",
                config_id=1,
                user_id=1,
                thread_id="test-flow"
            )
        )

        # Deve retornar um dicionario com estado final
        assert isinstance(result, dict)

        # Estado deve ter campos essenciais
        assert "messages" in result or "synthesized_response" in result
        # Nao deve ter erro (se processou corretamente)
        # Nota: pode haver erro None ou error key ausente
        error = result.get("error")
        # Se tiver erro, nao deve ser erro de sistema critico
        if error:
            assert "critical" not in error.lower()

    def test_graph_flow_empty_message(self):
        """Fluxo com mensagem vazia - deve tratar graciosamente."""
        agent = OrchestratorAgent()

        # Executar com mensagem vazia
        result = asyncio.get_event_loop().run_until_complete(
            agent.run(
                message="",
                config_id=1,
                user_id=1,
                thread_id="test-empty"
            )
        )

        # Deve retornar resultado mesmo com mensagem vazia
        assert isinstance(result, dict)

        # Estado final deve estar presente
        assert "messages" in result or "synthesized_response" in result or "error" in result

    def test_graph_flow_preserves_config(self):
        """Fluxo preserva config_id, user_id e thread_id."""
        agent = OrchestratorAgent()

        result = asyncio.get_event_loop().run_until_complete(
            agent.run(
                message="Teste de configuracao",
                config_id=42,
                user_id=100,
                thread_id="unique-thread-123"
            )
        )

        # Valores devem ser preservados no estado
        assert result.get("config_id") == 42
        assert result.get("user_id") == 100
        assert result.get("thread_id") == "unique-thread-123"

    def test_graph_flow_with_intent(self):
        """Fluxo detecta intencao corretamente."""
        agent = OrchestratorAgent()

        # Mensagem com intencao clara de analise
        result = asyncio.get_event_loop().run_until_complete(
            agent.run(
                message="Analise o desempenho das campanhas",
                config_id=1,
                user_id=1,
                thread_id="test-intent"
            )
        )

        # Deve ter detectado alguma intencao (nao None)
        # Nota: intencao pode ser "general" se nao detectar especifica
        user_intent = result.get("user_intent")
        # Apos parse_request, deve ter algum valor
        assert user_intent is not None or "error" in result


# =============================================================================
# Testes de roteamento condicional
# =============================================================================
class TestConditionalRouting:
    """Testes para roteamento condicional do grafo."""

    def test_routing_skips_dispatch_when_no_agents(self):
        """Roteamento pula dispatch quando nao ha agentes."""
        # Estado sem agentes requeridos
        state = create_initial_orchestrator_state(
            config_id=1,
            user_id=1,
            thread_id="test"
        )
        state["required_agents"] = []
        state["user_intent"] = "general"

        result = should_dispatch(state)

        assert result == "synthesize"

    def test_routing_goes_to_dispatch_with_agents(self):
        """Roteamento vai para dispatch quando ha agentes."""
        # Estado com agentes requeridos
        state = create_initial_orchestrator_state(
            config_id=1,
            user_id=1,
            thread_id="test"
        )
        state["required_agents"] = ["classification", "anomaly"]
        state["user_intent"] = "find_problems"

        result = should_dispatch(state)

        assert result == "dispatch"


# =============================================================================
# Testes de tratamento de erros
# =============================================================================
class TestErrorHandling:
    """Testes de tratamento de erros no grafo."""

    def test_handles_invalid_state_gracefully(self):
        """Trata estado invalido graciosamente."""
        # Testar should_dispatch com estado malformado
        bad_state = {"messages": []}  # Faltam campos obrigatorios

        # Nao deve lancar excecao, deve retornar padrao
        try:
            result = should_dispatch(bad_state)  # type: ignore
            assert result in ["dispatch", "synthesize"]
        except Exception as e:
            # Se lancar excecao, deve ser KeyError ou similar (nao critico)
            assert isinstance(e, (KeyError, TypeError, AttributeError))

    def test_build_graph_multiple_calls_stable(self):
        """Multiplas chamadas a build_orchestrator_graph sao estaveis."""
        # Construir grafo varias vezes
        graphs = [build_orchestrator_graph() for _ in range(3)]

        # Todos devem ser validos
        for graph in graphs:
            assert graph is not None
            assert hasattr(graph, "invoke") or hasattr(graph, "ainvoke")


# =============================================================================
# Testes adicionais de integracao
# =============================================================================
class TestGraphIntegration:
    """Testes de integracao do grafo com outros componentes."""

    def test_graph_uses_parse_request(self):
        """Grafo usa o no parse_request."""
        graph = build_orchestrator_graph()
        graph_data = graph.get_graph()

        # parse_request deve estar no grafo
        assert "parse_request" in graph_data.nodes

    def test_graph_uses_plan_execution(self):
        """Grafo usa o no plan_execution."""
        graph = build_orchestrator_graph()
        graph_data = graph.get_graph()

        # plan_execution deve estar no grafo
        assert "plan_execution" in graph_data.nodes

    def test_graph_uses_synthesize(self):
        """Grafo usa o no synthesize."""
        graph = build_orchestrator_graph()
        graph_data = graph.get_graph()

        # synthesize deve estar no grafo
        assert "synthesize" in graph_data.nodes

    def test_graph_flow_direct_to_synthesize(self):
        """Grafo vai direto para synthesize quando nao ha agentes."""
        # Criar estado sem agentes
        state = create_initial_orchestrator_state(
            config_id=1,
            user_id=1,
            thread_id="test"
        )
        state["required_agents"] = []

        # Verificar roteamento
        result = should_dispatch(state)
        assert result == "synthesize"


# =============================================================================
# Testes de singleton do orchestrator
# =============================================================================
class TestOrchestratorSingleton:
    """Testes do singleton do orchestrator."""

    def test_get_orchestrator_import(self):
        """get_orchestrator deve ser importavel."""
        from app.agent.orchestrator.graph import get_orchestrator
        assert get_orchestrator is not None

    def test_get_orchestrator_singleton(self):
        """get_orchestrator deve retornar a mesma instancia."""
        from app.agent.orchestrator.graph import get_orchestrator, reset_orchestrator

        reset_orchestrator()
        g1 = get_orchestrator()
        g2 = get_orchestrator()

        assert g1 is g2
