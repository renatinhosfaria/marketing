"""Testes para OrchestratorState."""
import sys
import os
import importlib.util

import pytest
from typing import get_type_hints

# Carregar o modulo state diretamente sem passar pelo __init__.py
# Isso evita dependencias de outros modulos que podem ter requirements nao instalados
state_path = os.path.join(
    os.path.dirname(__file__),
    '..', '..', '..', '..',
    'app', 'agent', 'orchestrator', 'state.py'
)
state_path = os.path.abspath(state_path)

spec = importlib.util.spec_from_file_location("orchestrator_state", state_path)
orchestrator_state = importlib.util.module_from_spec(spec)
spec.loader.exec_module(orchestrator_state)

OrchestratorState = orchestrator_state.OrchestratorState
ExecutionPlan = orchestrator_state.ExecutionPlan
VALID_AGENTS = orchestrator_state.VALID_AGENTS
INTENT_TO_AGENTS = orchestrator_state.INTENT_TO_AGENTS
PRIORITY_ORDER = orchestrator_state.PRIORITY_ORDER
create_initial_orchestrator_state = orchestrator_state.create_initial_orchestrator_state
get_agents_for_intent = orchestrator_state.get_agents_for_intent


class TestOrchestratorState:
    """Testes para o estado do orchestrator."""

    def test_orchestrator_state_import(self):
        """OrchestratorState deve ser importavel."""
        assert OrchestratorState is not None

    def test_orchestrator_state_has_conversation_fields(self):
        """OrchestratorState deve ter campos de conversa."""
        hints = get_type_hints(OrchestratorState)

        assert 'messages' in hints
        assert 'thread_id' in hints
        assert 'config_id' in hints
        assert 'user_id' in hints

    def test_orchestrator_state_has_planning_fields(self):
        """OrchestratorState deve ter campos de planejamento."""
        hints = get_type_hints(OrchestratorState)

        assert 'user_intent' in hints
        assert 'required_agents' in hints
        assert 'execution_plan' in hints

    def test_orchestrator_state_has_result_fields(self):
        """OrchestratorState deve ter campos de resultado."""
        hints = get_type_hints(OrchestratorState)

        assert 'agent_results' in hints
        assert 'synthesized_response' in hints
        assert 'confidence_score' in hints

    def test_orchestrator_state_has_error_field(self):
        """OrchestratorState deve ter campo de erro."""
        hints = get_type_hints(OrchestratorState)

        assert 'error' in hints


class TestIntentToAgents:
    """Testes para constante INTENT_TO_AGENTS."""

    def test_intent_to_agents_mapping(self):
        """INTENT_TO_AGENTS deve ter mapeamentos corretos."""
        assert 'analyze_performance' in INTENT_TO_AGENTS
        assert 'find_problems' in INTENT_TO_AGENTS
        assert 'get_recommendations' in INTENT_TO_AGENTS
        assert 'predict_future' in INTENT_TO_AGENTS
        assert 'full_report' in INTENT_TO_AGENTS

    def test_intent_to_agents_values(self):
        """Valores de INTENT_TO_AGENTS devem ser listas de agentes validos."""
        for intent, agents in INTENT_TO_AGENTS.items():
            assert isinstance(agents, list), f"Valor para {intent} deve ser lista"
            for agent in agents:
                assert agent in VALID_AGENTS, f"Agente {agent} invalido para {intent}"

    def test_intent_to_agents_has_general_fallback(self):
        """INTENT_TO_AGENTS deve ter fallback 'general'."""
        assert 'general' in INTENT_TO_AGENTS
        assert len(INTENT_TO_AGENTS['general']) > 0


class TestCreateInitialOrchestratorState:
    """Testes para funcao create_initial_orchestrator_state."""

    def test_create_initial_orchestrator_state(self):
        """create_initial_orchestrator_state deve criar estado valido."""
        state = create_initial_orchestrator_state(
            config_id=1,
            user_id=1,
            thread_id="test-thread"
        )

        assert state['config_id'] == 1
        assert state['user_id'] == 1
        assert state['thread_id'] == "test-thread"
        assert state['messages'] == []
        assert state['agent_results'] == {}

    def test_create_initial_orchestrator_state_defaults(self):
        """create_initial_orchestrator_state deve ter valores padrao corretos."""
        state = create_initial_orchestrator_state(
            config_id=5,
            user_id=10,
            thread_id="thread-xyz"
        )

        assert state['user_intent'] is None
        assert state['required_agents'] == []
        assert state['execution_plan'] is None
        assert state['synthesized_response'] is None
        assert state['confidence_score'] == 0.0
        assert state['error'] is None

    def test_create_initial_orchestrator_state_with_messages(self):
        """create_initial_orchestrator_state deve aceitar mensagens iniciais."""
        from langchain_core.messages import HumanMessage

        initial_messages = [HumanMessage(content="Ola, como estao minhas campanhas?")]

        state = create_initial_orchestrator_state(
            config_id=1,
            user_id=1,
            thread_id="test-thread",
            messages=initial_messages
        )

        assert len(state['messages']) == 1
        assert state['messages'][0].content == "Ola, como estao minhas campanhas?"


class TestExecutionPlan:
    """Testes para ExecutionPlan TypedDict."""

    def test_execution_plan_import(self):
        """ExecutionPlan deve ser importavel."""
        assert ExecutionPlan is not None

    def test_execution_plan_has_required_fields(self):
        """ExecutionPlan deve ter campos obrigatorios."""
        hints = get_type_hints(ExecutionPlan)

        assert 'agents' in hints
        assert 'tasks' in hints
        assert 'parallel' in hints
        assert 'timeout' in hints


class TestGetAgentsForIntent:
    """Testes para funcao get_agents_for_intent."""

    def test_get_agents_for_known_intent(self):
        """get_agents_for_intent deve retornar agentes para intencao conhecida."""
        agents = get_agents_for_intent("analyze_performance")
        assert isinstance(agents, list)
        assert len(agents) > 0

    def test_get_agents_for_unknown_intent_returns_general(self):
        """get_agents_for_intent deve retornar general para intencao desconhecida."""
        agents = get_agents_for_intent("unknown_intent_xyz")
        assert agents == INTENT_TO_AGENTS["general"]

    def test_get_agents_for_all_known_intents(self):
        """get_agents_for_intent deve retornar agentes para todas intencoes conhecidas."""
        for intent in INTENT_TO_AGENTS:
            agents = get_agents_for_intent(intent)
            assert agents == INTENT_TO_AGENTS[intent]


class TestPriorityOrder:
    """Testes para constante PRIORITY_ORDER."""

    def test_priority_order_exists(self):
        """PRIORITY_ORDER deve existir e ter agentes validos."""
        assert PRIORITY_ORDER is not None
        for agent in PRIORITY_ORDER:
            assert agent in VALID_AGENTS, f"Agente {agent} invalido em PRIORITY_ORDER"

    def test_priority_order_values_are_integers(self):
        """Valores de PRIORITY_ORDER devem ser inteiros."""
        for agent, priority in PRIORITY_ORDER.items():
            assert isinstance(priority, int), f"Prioridade para {agent} deve ser int"
            assert priority >= 1, f"Prioridade para {agent} deve ser >= 1"

    def test_priority_order_has_all_agents(self):
        """PRIORITY_ORDER deve ter todos os agentes validos."""
        for agent in VALID_AGENTS:
            assert agent in PRIORITY_ORDER, f"Agente {agent} faltando em PRIORITY_ORDER"


class TestValidAgents:
    """Testes para constante VALID_AGENTS."""

    def test_valid_agents_has_expected_agents(self):
        """VALID_AGENTS deve ter agentes esperados."""
        expected = {'classification', 'anomaly', 'forecast', 'recommendation', 'campaign', 'analysis'}
        assert VALID_AGENTS == expected

    def test_valid_agents_is_frozenset(self):
        """VALID_AGENTS deve ser frozenset (imutavel)."""
        assert isinstance(VALID_AGENTS, frozenset)
