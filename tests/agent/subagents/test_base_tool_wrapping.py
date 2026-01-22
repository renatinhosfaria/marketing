"""Testes para tool wrapping no BaseSubagent."""
import pytest
from unittest.mock import Mock
from langchain_core.tools import BaseTool

from app.agent.subagents.base import BaseSubagent


class MockSubagent(BaseSubagent):
    """Subagente mock para testes."""

    AGENT_NAME = "test_agent"
    AGENT_DESCRIPTION = "Test agent"

    def get_tools(self):
        return []

    def get_system_prompt(self):
        return "Test prompt"


def test_wrap_tool_does_not_mutate_original():
    """Testa que wrapping não modifica a tool original."""
    # Criar tool mock
    original_func = Mock(return_value="original result")
    tool = Mock(spec=BaseTool)
    tool.name = "test_tool"
    tool.func = original_func

    # Salvar referência ao func original
    func_before = tool.func

    # Criar subagent e wrappear tool
    agent = MockSubagent()
    wrapped_tool = agent._wrap_tool_with_logging(tool)

    # Verificar que a tool original NÃO foi modificada
    assert tool.func is func_before, "Tool original foi modificada (mutação)"

    # Verificar que wrapped_tool é diferente
    assert wrapped_tool is not tool, "Wrapped tool deve ser uma nova instância"
    assert wrapped_tool.func is not func_before, "Wrapped func deve ser diferente"


def test_wrap_tool_preserves_tool_properties():
    """Testa que wrapping preserva propriedades da tool."""
    # Criar tool mock
    original_func = Mock(return_value="result")
    tool = Mock(spec=BaseTool)
    tool.name = "my_tool"
    tool.description = "My tool description"
    tool.func = original_func

    # Wrappear
    agent = MockSubagent()
    wrapped_tool = agent._wrap_tool_with_logging(tool)

    # Verificar que propriedades foram preservadas
    assert wrapped_tool.name == "my_tool"
    assert wrapped_tool.description == "My tool description"


def test_multiple_wraps_are_independent():
    """Testa que múltiplos wraps da mesma tool são independentes."""
    # Criar tool
    original_func = Mock(return_value="result")
    tool = Mock(spec=BaseTool)
    tool.name = "test_tool"
    tool.func = original_func

    # Wrappear duas vezes
    agent = MockSubagent()
    wrapped1 = agent._wrap_tool_with_logging(tool)
    wrapped2 = agent._wrap_tool_with_logging(tool)

    # Verificar que são instâncias independentes
    assert wrapped1 is not wrapped2
    assert wrapped1.func is not wrapped2.func

    # Verificar que original não foi modificada
    assert tool.func is original_func
