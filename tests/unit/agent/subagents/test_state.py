"""Testes para SubagentState."""
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
    'app', 'agent', 'subagents', 'state.py'
)
state_path = os.path.abspath(state_path)

spec = importlib.util.spec_from_file_location("subagent_state", state_path)
subagent_state = importlib.util.module_from_spec(spec)
spec.loader.exec_module(subagent_state)

SubagentState = subagent_state.SubagentState
AgentResult = subagent_state.AgentResult
SubagentTask = subagent_state.SubagentTask
create_initial_subagent_state = subagent_state.create_initial_subagent_state


class TestSubagentState:
    """Testes para o estado dos subagentes."""

    def test_subagent_state_import(self):
        """SubagentState deve ser importavel."""
        assert SubagentState is not None

    def test_subagent_state_has_required_fields(self):
        """SubagentState deve ter todos os campos obrigatorios."""
        hints = get_type_hints(SubagentState)

        required_fields = [
            'messages', 'task', 'config_id', 'user_id',
            'thread_id', 'result', 'error', 'tool_calls_count',
            'started_at', 'completed_at'
        ]
        for field in required_fields:
            assert field in hints, f"Campo {field} nao encontrado"

    def test_agent_result_import(self):
        """AgentResult deve ser importavel."""
        assert AgentResult is not None

    def test_agent_result_has_fields(self):
        """AgentResult deve ter campos necessarios."""
        hints = get_type_hints(AgentResult)

        required_fields = ['agent_name', 'success', 'data', 'error', 'duration_ms', 'tool_calls']
        for field in required_fields:
            assert field in hints, f"Campo {field} nao encontrado em AgentResult"

    def test_subagent_task_import(self):
        """SubagentTask deve ser importavel."""
        assert SubagentTask is not None

    def test_subagent_task_has_fields(self):
        """SubagentTask deve ter campos necessarios."""
        hints = get_type_hints(SubagentTask)

        required_fields = ['description', 'context', 'priority']
        for field in required_fields:
            assert field in hints, f"Campo {field} nao encontrado em SubagentTask"


class TestCreateInitialSubagentState:
    """Testes para a funcao create_initial_subagent_state."""

    def test_create_initial_state_basic(self):
        """Deve criar estado inicial com valores basicos."""
        task = SubagentTask(
            description="Analisar campanha",
            context={"campaign_id": 123},
            priority=1
        )

        state = create_initial_subagent_state(
            task=task,
            config_id=1,
            user_id=42,
            thread_id="thread-123"
        )

        assert state['task'] == task
        assert state['config_id'] == 1
        assert state['user_id'] == 42
        assert state['thread_id'] == "thread-123"
        assert state['messages'] == []
        assert state['result'] is None
        assert state['error'] is None
        assert state['tool_calls_count'] == 0
        assert state['started_at'] is not None
        assert state['completed_at'] is None

    def test_create_initial_state_with_messages(self):
        """Deve aceitar mensagens iniciais."""
        from langchain_core.messages import HumanMessage

        task = SubagentTask(
            description="Tarefa teste",
            context={},
            priority=2
        )
        initial_messages = [HumanMessage(content="Contexto inicial")]

        state = create_initial_subagent_state(
            task=task,
            config_id=1,
            user_id=1,
            thread_id="test-thread",
            messages=initial_messages
        )

        assert len(state['messages']) == 1
        assert state['messages'][0].content == "Contexto inicial"
