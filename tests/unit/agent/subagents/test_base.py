"""Testes para BaseSubagent."""
import sys
import os
import importlib.util
import types

import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime


# Adicionar o diretorio raiz ao path para permitir imports
root_path = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    '..', '..', '..', '..'
))
if root_path not in sys.path:
    sys.path.insert(0, root_path)


# Carregar o modulo state diretamente para evitar dependencias
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


# Carregar o modulo base diretamente
base_path = os.path.join(
    os.path.dirname(__file__),
    '..', '..', '..', '..',
    'app', 'agent', 'subagents', 'base.py'
)
base_path = os.path.abspath(base_path)


def _create_mock_config_module():
    """Cria um modulo de configuracao mock."""
    config_mod = types.ModuleType('app.agent.config')

    class MockAgentSettings:
        timeout_test = 30
        timeout_classification = 30
        timeout_anomaly = 30
        timeout_forecast = 45
        timeout_recommendation = 30
        timeout_campaign = 20
        timeout_analysis = 45

    def get_agent_settings():
        return MockAgentSettings()

    config_mod.get_agent_settings = get_agent_settings
    config_mod.AgentSettings = MockAgentSettings
    return config_mod


def load_base_module():
    """Carrega o modulo base dinamicamente evitando imports circulares."""
    # Criar um modulo fake para .state que sera usado pelo base.py
    # Isso evita que o import relativo tente carregar app.agent

    # Primeiro, criar o pacote app.agent.subagents fake se necessario
    if 'app' not in sys.modules:
        sys.modules['app'] = types.ModuleType('app')
    if 'app.agent' not in sys.modules:
        agent_mod = types.ModuleType('app.agent')
        sys.modules['app.agent'] = agent_mod
    if 'app.agent.subagents' not in sys.modules:
        subagents_mod = types.ModuleType('app.agent.subagents')
        sys.modules['app.agent.subagents'] = subagents_mod

    # Registrar o modulo state como app.agent.subagents.state
    sys.modules['app.agent.subagents.state'] = subagent_state

    # Criar e registrar modulo config mock
    config_mod = _create_mock_config_module()
    sys.modules['app.agent.config'] = config_mod

    # Verificar se ja foi carregado (evitar recarregamento)
    if 'app.agent.subagents.base' in sys.modules:
        return sys.modules['app.agent.subagents.base']

    # Agora carregar o modulo base
    spec = importlib.util.spec_from_file_location(
        "app.agent.subagents.base",
        base_path,
        submodule_search_locations=[os.path.dirname(base_path)]
    )
    base_module = importlib.util.module_from_spec(spec)

    # Configurar o parent package
    base_module.__package__ = 'app.agent.subagents'

    spec.loader.exec_module(base_module)

    # Registrar no sys.modules
    sys.modules['app.agent.subagents.base'] = base_module

    return base_module


class TestBaseSubagent:
    """Testes para a classe base dos subagentes."""

    def test_base_subagent_import(self):
        """BaseSubagent deve ser importavel."""
        base_module = load_base_module()
        assert hasattr(base_module, 'BaseSubagent')
        assert base_module.BaseSubagent is not None

    def test_base_subagent_is_abstract(self):
        """BaseSubagent deve ser classe abstrata."""
        import abc
        base_module = load_base_module()
        BaseSubagent = base_module.BaseSubagent
        assert issubclass(BaseSubagent, abc.ABC)

    def test_base_subagent_requires_name(self):
        """Subclasses devem definir AGENT_NAME."""
        base_module = load_base_module()
        BaseSubagent = base_module.BaseSubagent

        class InvalidAgent(BaseSubagent):
            """Agente invalido sem AGENT_NAME."""
            AGENT_DESCRIPTION = "Test"

            def get_tools(self):
                return []

            def get_system_prompt(self) -> str:
                return "Test prompt"

        with pytest.raises(TypeError):
            InvalidAgent()

    def test_base_subagent_has_build_graph_method(self):
        """BaseSubagent deve ter metodo build_graph."""
        base_module = load_base_module()
        BaseSubagent = base_module.BaseSubagent
        assert hasattr(BaseSubagent, 'build_graph')

    def test_base_subagent_has_get_tools_method(self):
        """BaseSubagent deve ter metodo get_tools."""
        base_module = load_base_module()
        BaseSubagent = base_module.BaseSubagent
        assert hasattr(BaseSubagent, 'get_tools')

    def test_base_subagent_has_get_system_prompt_method(self):
        """BaseSubagent deve ter metodo get_system_prompt."""
        base_module = load_base_module()
        BaseSubagent = base_module.BaseSubagent
        assert hasattr(BaseSubagent, 'get_system_prompt')

    def test_base_subagent_has_run_method(self):
        """BaseSubagent deve ter metodo run."""
        base_module = load_base_module()
        BaseSubagent = base_module.BaseSubagent
        assert hasattr(BaseSubagent, 'run')

    def test_base_subagent_has_get_timeout_method(self):
        """BaseSubagent deve ter metodo get_timeout."""
        base_module = load_base_module()
        BaseSubagent = base_module.BaseSubagent
        assert hasattr(BaseSubagent, 'get_timeout')


class TestConcreteSubagent:
    """Testes para implementacao concreta de subagente."""

    def test_concrete_subagent_creation(self):
        """Subagente concreto deve ser criavel."""
        base_module = load_base_module()
        BaseSubagent = base_module.BaseSubagent
        from langchain_core.tools import tool

        @tool
        def dummy_tool() -> str:
            """Ferramenta de teste."""
            return "result"

        class TestAgent(BaseSubagent):
            AGENT_NAME = "test"
            AGENT_DESCRIPTION = "Test agent"

            def get_tools(self):
                return [dummy_tool]

            def get_system_prompt(self) -> str:
                return "You are a test agent."

        agent = TestAgent()
        assert agent.AGENT_NAME == "test"
        assert agent.AGENT_DESCRIPTION == "Test agent"
        assert agent.get_system_prompt() == "You are a test agent."

    def test_concrete_subagent_get_tools(self):
        """Subagente concreto deve retornar tools."""
        base_module = load_base_module()
        BaseSubagent = base_module.BaseSubagent
        from langchain_core.tools import tool

        @tool
        def tool_a() -> str:
            """Tool A."""
            return "a"

        @tool
        def tool_b() -> str:
            """Tool B."""
            return "b"

        class TestAgent(BaseSubagent):
            AGENT_NAME = "test"
            AGENT_DESCRIPTION = "Test agent"

            def get_tools(self):
                return [tool_a, tool_b]

            def get_system_prompt(self) -> str:
                return "Test prompt"

        agent = TestAgent()
        tools = agent.get_tools()
        assert len(tools) == 2
        assert tools[0].name == "tool_a"
        assert tools[1].name == "tool_b"

    def test_subagent_graph_builds(self):
        """Grafo do subagente deve ser construido."""
        base_module = load_base_module()
        BaseSubagent = base_module.BaseSubagent
        from langchain_core.tools import tool

        @tool
        def dummy_tool() -> str:
            """Ferramenta de teste."""
            return "result"

        class TestAgent(BaseSubagent):
            AGENT_NAME = "test"
            AGENT_DESCRIPTION = "Test agent"

            def get_tools(self):
                return [dummy_tool]

            def get_system_prompt(self) -> str:
                return "You are a test agent."

        agent = TestAgent()
        graph = agent.build_graph()

        # Verificar que grafo foi construido
        assert graph is not None

    def test_subagent_get_timeout_returns_int(self):
        """get_timeout deve retornar inteiro."""
        base_module = load_base_module()
        BaseSubagent = base_module.BaseSubagent
        from langchain_core.tools import tool

        @tool
        def dummy_tool() -> str:
            """Ferramenta de teste."""
            return "result"

        class TestAgent(BaseSubagent):
            AGENT_NAME = "test"
            AGENT_DESCRIPTION = "Test agent"

            def get_tools(self):
                return [dummy_tool]

            def get_system_prompt(self) -> str:
                return "Test prompt"

        agent = TestAgent()
        timeout = agent.get_timeout()
        assert isinstance(timeout, int)
        assert timeout > 0


class TestSubagentHelperMethods:
    """Testes para metodos auxiliares do subagente."""

    def test_format_task_message(self):
        """_format_task_message deve formatar mensagem da tarefa."""
        base_module = load_base_module()
        BaseSubagent = base_module.BaseSubagent
        from langchain_core.tools import tool

        @tool
        def dummy_tool() -> str:
            """Ferramenta de teste."""
            return "result"

        class TestAgent(BaseSubagent):
            AGENT_NAME = "test"
            AGENT_DESCRIPTION = "Test agent"

            def get_tools(self):
                return [dummy_tool]

            def get_system_prompt(self) -> str:
                return "Test prompt"

        agent = TestAgent()

        task = SubagentTask(
            description="Analisar campanha XYZ",
            context={"campaign_id": 123, "metric": "CPC"},
            priority=1
        )

        state = create_initial_subagent_state(
            task=task,
            config_id=1,
            user_id=42,
            thread_id="thread-123"
        )

        formatted = agent._format_task_message(state)
        assert "Analisar campanha XYZ" in formatted
        assert "campaign_id" in formatted or "123" in formatted

    def test_extract_tool_calls_empty(self):
        """_extract_tool_calls deve retornar lista vazia se nao houver tool calls."""
        base_module = load_base_module()
        BaseSubagent = base_module.BaseSubagent
        from langchain_core.tools import tool
        from langchain_core.messages import HumanMessage, AIMessage

        @tool
        def dummy_tool() -> str:
            """Ferramenta de teste."""
            return "result"

        class TestAgent(BaseSubagent):
            AGENT_NAME = "test"
            AGENT_DESCRIPTION = "Test agent"

            def get_tools(self):
                return [dummy_tool]

            def get_system_prompt(self) -> str:
                return "Test prompt"

        agent = TestAgent()
        messages = [
            HumanMessage(content="Hello"),
            AIMessage(content="Hi there!")
        ]

        tool_calls = agent._extract_tool_calls(messages)
        assert tool_calls == []

    def test_extract_tool_calls_with_calls(self):
        """_extract_tool_calls deve extrair nomes das tools chamadas."""
        base_module = load_base_module()
        BaseSubagent = base_module.BaseSubagent
        from langchain_core.tools import tool
        from langchain_core.messages import HumanMessage, AIMessage

        @tool
        def dummy_tool() -> str:
            """Ferramenta de teste."""
            return "result"

        class TestAgent(BaseSubagent):
            AGENT_NAME = "test"
            AGENT_DESCRIPTION = "Test agent"

            def get_tools(self):
                return [dummy_tool]

            def get_system_prompt(self) -> str:
                return "Test prompt"

        agent = TestAgent()

        # AIMessage com tool_calls
        ai_message = AIMessage(
            content="",
            tool_calls=[
                {"name": "tool_a", "id": "1", "args": {}},
                {"name": "tool_b", "id": "2", "args": {"x": 1}}
            ]
        )
        messages = [
            HumanMessage(content="Execute tools"),
            ai_message
        ]

        tool_calls = agent._extract_tool_calls(messages)
        assert "tool_a" in tool_calls
        assert "tool_b" in tool_calls
        assert len(tool_calls) == 2


class TestSubagentRunMethod:
    """Testes para o metodo run do subagente."""

    def test_run_method_returns_agent_result(self):
        """run deve retornar AgentResult."""
        import asyncio

        base_module = load_base_module()
        BaseSubagent = base_module.BaseSubagent
        from langchain_core.tools import tool

        @tool
        def dummy_tool() -> str:
            """Ferramenta de teste."""
            return "result"

        class TestAgent(BaseSubagent):
            AGENT_NAME = "test"
            AGENT_DESCRIPTION = "Test agent"

            def get_tools(self):
                return [dummy_tool]

            def get_system_prompt(self) -> str:
                return "Test prompt"

        agent = TestAgent()

        task = SubagentTask(
            description="Test task",
            context={},
            priority=1
        )

        async def run_test():
            # Mock do grafo para evitar chamadas reais ao LLM
            with patch.object(agent, 'build_graph') as mock_build:
                mock_graph = Mock()
                mock_graph.ainvoke = AsyncMock(return_value={
                    'messages': [],
                    'result': {"test": "data"},
                    'error': None,
                    'tool_calls_count': 0
                })
                mock_build.return_value = mock_graph

                result = await agent.run(
                    task=task,
                    config_id=1,
                    user_id=42,
                    thread_id="thread-123"
                )

                assert result is not None
                assert 'agent_name' in result
                assert result['agent_name'] == "test"
                assert 'success' in result
                assert 'duration_ms' in result

        asyncio.run(run_test())

    def test_run_method_handles_timeout(self):
        """run deve tratar timeout corretamente."""
        import asyncio

        base_module = load_base_module()
        BaseSubagent = base_module.BaseSubagent
        from langchain_core.tools import tool

        @tool
        def dummy_tool() -> str:
            """Ferramenta de teste."""
            return "result"

        class TestAgent(BaseSubagent):
            AGENT_NAME = "test"
            AGENT_DESCRIPTION = "Test agent"

            def get_tools(self):
                return [dummy_tool]

            def get_system_prompt(self) -> str:
                return "Test prompt"

            def get_timeout(self) -> int:
                return 1  # 1 segundo de timeout

        agent = TestAgent()

        task = SubagentTask(
            description="Test task",
            context={},
            priority=1
        )

        async def run_test():
            # Mock do grafo que demora mais que o timeout
            async def slow_invoke(*args, **kwargs):
                await asyncio.sleep(5)  # Mais que o timeout
                return {'messages': [], 'result': None, 'error': None}

            with patch.object(agent, 'build_graph') as mock_build:
                mock_graph = Mock()
                mock_graph.ainvoke = slow_invoke
                mock_build.return_value = mock_graph

                result = await agent.run(
                    task=task,
                    config_id=1,
                    user_id=42,
                    thread_id="thread-123"
                )

                # Deve retornar erro de timeout
                assert result is not None
                assert result['success'] is False
                assert result['error'] is not None
                assert 'timeout' in result['error'].lower() or 'Timeout' in result['error']

        asyncio.run(run_test())

    def test_run_method_handles_error(self):
        """run deve tratar erros graciosamente."""
        import asyncio

        base_module = load_base_module()
        BaseSubagent = base_module.BaseSubagent
        from langchain_core.tools import tool

        @tool
        def dummy_tool() -> str:
            """Ferramenta de teste."""
            return "result"

        class TestAgent(BaseSubagent):
            AGENT_NAME = "test"
            AGENT_DESCRIPTION = "Test agent"

            def get_tools(self):
                return [dummy_tool]

            def get_system_prompt(self) -> str:
                return "Test prompt"

        agent = TestAgent()

        task = SubagentTask(
            description="Test task",
            context={},
            priority=1
        )

        async def run_test():
            # Mock do grafo que lanca excecao
            with patch.object(agent, 'build_graph') as mock_build:
                mock_graph = Mock()
                mock_graph.ainvoke = AsyncMock(side_effect=Exception("LLM Error"))
                mock_build.return_value = mock_graph

                result = await agent.run(
                    task=task,
                    config_id=1,
                    user_id=42,
                    thread_id="thread-123"
                )

                # Deve retornar erro
                assert result is not None
                assert result['success'] is False
                assert result['error'] is not None
                assert "LLM Error" in result['error'] or "error" in result['error'].lower()

        asyncio.run(run_test())
