"""Testes unitarios para streaming multi-agente do TrafficAgentService.

Testa:
- stream_chat_multi_agent emite eventos na ordem correta
- Eventos possuem estrutura esperada
- Tratamento de erros
- Geracao de thread_id
"""
import sys
import os
import types
import uuid
import asyncio

import pytest
from unittest.mock import MagicMock, AsyncMock, patch


# Adicionar o diretorio raiz ao path
root_path = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    '..', '..', '..'
))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

# Modulos que serao mockados durante o import do service.py
_MOCKED_MODULE_NAMES = [
    "app.core.logging",
    "app.core",
    "app.agent.config",
    "app.agent.graph.state",
    "app.agent.graph",
    "app.agent.graph.builder",
    "app.agent.memory.checkpointer",
    "app.agent.memory",
    "langchain_core.messages",
    "langchain_core",
    "app.agent.orchestrator.state",
]
_REAL_MODULES = {name: sys.modules.get(name) for name in _MOCKED_MODULE_NAMES}


def _restore_mocked_modules():
    """Restaura modulos reais apos importar o service.py."""
    for name, module in _REAL_MODULES.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


# =============================================================================
# Setup: Criar modulos fake para evitar imports circulares
# =============================================================================
def _setup_mock_modules():
    """Configura modulos mock para permitir import do service.py."""

    # Mock do logging
    if 'app.core.logging' not in sys.modules:
        logging_mod = types.ModuleType('app.core.logging')

        class MockLogger:
            def info(self, *args, **kwargs): pass
            def warning(self, *args, **kwargs): pass
            def error(self, *args, **kwargs): pass
            def debug(self, *args, **kwargs): pass

        def get_logger(name):
            return MockLogger()

        logging_mod.get_logger = get_logger
        sys.modules['app.core.logging'] = logging_mod
        sys.modules['app.core'] = types.ModuleType('app.core')

    # Mock do config
    if 'app.agent.config' not in sys.modules:
        config_mod = types.ModuleType('app.agent.config')

        class MockAgentSettings:
            multi_agent_enabled = False
            orchestrator_timeout = 120
            max_parallel_subagents = 4

        def get_agent_settings():
            return MockAgentSettings()

        config_mod.get_agent_settings = get_agent_settings
        config_mod.AgentSettings = MockAgentSettings
        sys.modules['app.agent.config'] = config_mod

    # Mock do graph state
    if 'app.agent.graph.state' not in sys.modules:
        state_mod = types.ModuleType('app.agent.graph.state')

        class MockAgentState(dict):
            pass

        def create_initial_state(**kwargs):
            return MockAgentState(kwargs)

        state_mod.AgentState = MockAgentState
        state_mod.create_initial_state = create_initial_state
        sys.modules['app.agent.graph.state'] = state_mod
        sys.modules['app.agent.graph'] = types.ModuleType('app.agent.graph')

    # Mock do graph builder
    if 'app.agent.graph.builder' not in sys.modules:
        builder_mod = types.ModuleType('app.agent.graph.builder')

        def build_agent_graph():
            mock_graph = MagicMock()
            mock_graph.compile = MagicMock(return_value=MagicMock())
            return mock_graph

        builder_mod.build_agent_graph = build_agent_graph
        sys.modules['app.agent.graph.builder'] = builder_mod

    # Mock do checkpointer
    if 'app.agent.memory.checkpointer' not in sys.modules:
        checkpointer_mod = types.ModuleType('app.agent.memory.checkpointer')

        class MockAgentCheckpointer:
            @staticmethod
            async def get_checkpointer():
                return MagicMock()

        async def get_agent_checkpointer():
            return MagicMock()

        checkpointer_mod.AgentCheckpointer = MockAgentCheckpointer
        checkpointer_mod.get_agent_checkpointer = get_agent_checkpointer
        sys.modules['app.agent.memory.checkpointer'] = checkpointer_mod
        sys.modules['app.agent.memory'] = types.ModuleType('app.agent.memory')

    # Mock do langchain
    if 'langchain_core.messages' not in sys.modules:
        messages_mod = types.ModuleType('langchain_core.messages')

        class HumanMessage:
            def __init__(self, content):
                self.content = content
                self.type = "human"

        class AIMessage:
            def __init__(self, content):
                self.content = content
                self.type = "ai"

        messages_mod.HumanMessage = HumanMessage
        messages_mod.AIMessage = AIMessage
        sys.modules['langchain_core.messages'] = messages_mod
        sys.modules['langchain_core'] = types.ModuleType('langchain_core')

    # Mock do orchestrator state
    if 'app.agent.orchestrator.state' not in sys.modules:
        orch_state_mod = types.ModuleType('app.agent.orchestrator.state')

        class MockOrchestratorState(dict):
            pass

        def create_initial_orchestrator_state(**kwargs):
            return MockOrchestratorState(kwargs)

        def get_agents_for_intent(intent):
            return ["classification"]

        orch_state_mod.OrchestratorState = MockOrchestratorState
        orch_state_mod.create_initial_orchestrator_state = create_initial_orchestrator_state
        orch_state_mod.get_agents_for_intent = get_agents_for_intent
        sys.modules['app.agent.orchestrator.state'] = orch_state_mod


# Configurar mocks antes de importar
_setup_mock_modules()

# Agora importar o modulo de servico
import importlib.util
service_path = os.path.join(root_path, 'app', 'agent', 'service.py')
spec = importlib.util.spec_from_file_location("agent_service_streaming_test", service_path)
service_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(service_module)
_restore_mocked_modules()

TrafficAgentService = service_module.TrafficAgentService
get_multi_agent_service = service_module.get_multi_agent_service
reset_services = service_module.reset_services


def run_async(coro):
    """Helper para executar coroutines em testes sincronos."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


async def collect_events(gen):
    """Helper para coletar eventos de um AsyncGenerator."""
    events = []
    async for event in gen:
        events.append(event)
    return events


def collect_events_sync(gen):
    """Helper sincrono para coletar eventos."""
    return run_async(collect_events(gen))


# =============================================================================
# Fixtures
# =============================================================================
@pytest.fixture(autouse=True)
def reset_singletons():
    """Reseta singletons antes e depois de cada teste."""
    reset_services()
    yield
    reset_services()


@pytest.fixture
def mock_graph():
    """Cria mock do grafo compilado com astream."""
    mock = MagicMock()

    # Simular astream que retorna updates de cada no
    async def mock_astream(state, stream_mode=None):
        # Simular sequencia de atualizacoes do grafo
        yield {
            "parse_request": {
                "user_intent": "analyze_performance",
                "messages": state.get("messages", [])
            }
        }
        yield {
            "plan_execution": {
                "required_agents": ["classification", "anomaly"],
                "execution_plan": {
                    "agents": ["classification", "anomaly"],
                    "parallel": True,
                    "timeout": 30
                }
            }
        }
        yield {
            "dispatch_agents": {
                "agent_results": {
                    "classification": {"success": True, "data": {}},
                    "anomaly": {"success": True, "data": {}},
                }
            }
        }
        yield {
            "collect_results": {
                "agent_results": {
                    "classification": {"success": True, "data": {}},
                    "anomaly": {"success": True, "data": {}},
                }
            }
        }
        yield {
            "synthesize": {
                "synthesized_response": "Analise completa das campanhas com deteccao de anomalias.",
                "confidence_score": 0.85
            }
        }

    mock.astream = mock_astream
    return mock


@pytest.fixture
def mock_orchestrator(mock_graph):
    """Cria mock do OrchestratorAgent."""
    mock = MagicMock()
    mock.build_graph = MagicMock(return_value=mock_graph)
    return mock


# =============================================================================
# TestStreamChatMultiAgent
# =============================================================================
class TestStreamChatMultiAgent:
    """Testes para o metodo stream_chat_multi_agent."""

    def test_stream_emits_orchestrator_start(self, mock_orchestrator):
        """Primeiro evento e orchestrator_start."""
        service = TrafficAgentService(multi_agent_mode=True)
        service._orchestrator = mock_orchestrator

        events = collect_events_sync(service.stream_chat_multi_agent(
            message="Analise minhas campanhas",
            config_id=1,
            user_id=1,
            thread_id="test-thread"
        ))

        # Primeiro evento deve ser orchestrator_start
        assert len(events) > 0
        assert events[0]["type"] == "orchestrator_start"
        assert events[0]["thread_id"] == "test-thread"
        assert "timestamp" in events[0]

    def test_stream_emits_intent_detected(self, mock_orchestrator):
        """Emite evento intent_detected apos parse."""
        service = TrafficAgentService(multi_agent_mode=True)
        service._orchestrator = mock_orchestrator

        events = collect_events_sync(service.stream_chat_multi_agent(
            message="Analise minhas campanhas",
            config_id=1,
            user_id=1,
            thread_id="test-thread"
        ))

        # Encontrar evento intent_detected
        intent_events = [e for e in events if e.get("type") == "intent_detected"]
        assert len(intent_events) == 1
        assert intent_events[0]["intent"] == "analyze_performance"
        assert intent_events[0]["thread_id"] == "test-thread"
        assert "timestamp" in intent_events[0]

    def test_stream_emits_plan_created(self, mock_orchestrator):
        """Emite evento plan_created com lista de agentes."""
        service = TrafficAgentService(multi_agent_mode=True)
        service._orchestrator = mock_orchestrator

        events = collect_events_sync(service.stream_chat_multi_agent(
            message="Analise minhas campanhas",
            config_id=1,
            user_id=1,
            thread_id="test-thread"
        ))

        # Encontrar evento plan_created
        plan_events = [e for e in events if e.get("type") == "plan_created"]
        assert len(plan_events) == 1
        assert plan_events[0]["agents"] == ["classification", "anomaly"]
        assert plan_events[0]["parallel"] is True
        assert plan_events[0]["thread_id"] == "test-thread"
        assert "timestamp" in plan_events[0]

    def test_stream_emits_agent_events(self, mock_orchestrator):
        """Emite eventos agent_start e agent_end para cada subagente."""
        service = TrafficAgentService(multi_agent_mode=True)
        service._orchestrator = mock_orchestrator

        events = collect_events_sync(service.stream_chat_multi_agent(
            message="Analise minhas campanhas",
            config_id=1,
            user_id=1,
            thread_id="test-thread"
        ))

        # Encontrar eventos agent_start
        agent_start_events = [e for e in events if e.get("type") == "agent_start"]
        assert len(agent_start_events) == 2

        agent_names_started = [e["agent"] for e in agent_start_events]
        assert "classification" in agent_names_started
        assert "anomaly" in agent_names_started

        # Verificar descricoes
        for event in agent_start_events:
            assert "description" in event
            assert event["thread_id"] == "test-thread"

        # Encontrar eventos agent_end
        agent_end_events = [e for e in events if e.get("type") == "agent_end"]
        assert len(agent_end_events) == 2

        for event in agent_end_events:
            assert "success" in event
            assert "duration_ms" in event
            assert event["thread_id"] == "test-thread"

    def test_stream_emits_synthesis_start(self, mock_orchestrator):
        """Emite evento synthesis_start antes da sintese."""
        service = TrafficAgentService(multi_agent_mode=True)
        service._orchestrator = mock_orchestrator

        events = collect_events_sync(service.stream_chat_multi_agent(
            message="Analise minhas campanhas",
            config_id=1,
            user_id=1,
            thread_id="test-thread"
        ))

        # Encontrar evento synthesis_start
        synthesis_events = [e for e in events if e.get("type") == "synthesis_start"]
        assert len(synthesis_events) == 1
        assert synthesis_events[0]["agents_completed"] == 2
        assert synthesis_events[0]["thread_id"] == "test-thread"
        assert "timestamp" in synthesis_events[0]

    def test_stream_emits_text(self, mock_orchestrator):
        """Emite eventos text com chunks da resposta."""
        service = TrafficAgentService(multi_agent_mode=True)
        service._orchestrator = mock_orchestrator

        events = collect_events_sync(service.stream_chat_multi_agent(
            message="Analise minhas campanhas",
            config_id=1,
            user_id=1,
            thread_id="test-thread"
        ))

        # Encontrar eventos text
        text_events = [e for e in events if e.get("type") == "text"]
        assert len(text_events) > 0

        # Juntar todos os chunks
        full_text = "".join(e["content"] for e in text_events)
        assert "Analise completa" in full_text or "campanhas" in full_text

        # Verificar estrutura
        for event in text_events:
            assert "content" in event
            assert event["thread_id"] == "test-thread"
            assert "timestamp" in event

    def test_stream_emits_done(self, mock_orchestrator):
        """Evento final e done com metadados."""
        service = TrafficAgentService(multi_agent_mode=True)
        service._orchestrator = mock_orchestrator

        events = collect_events_sync(service.stream_chat_multi_agent(
            message="Analise minhas campanhas",
            config_id=1,
            user_id=1,
            thread_id="test-thread"
        ))

        # Ultimo evento deve ser done
        assert events[-1]["type"] == "done"
        assert events[-1]["thread_id"] == "test-thread"
        assert "confidence_score" in events[-1]
        assert "total_duration_ms" in events[-1]
        assert "agents_used" in events[-1]
        assert "timestamp" in events[-1]

        # Verificar valores
        assert isinstance(events[-1]["confidence_score"], float)
        assert events[-1]["total_duration_ms"] >= 0
        assert isinstance(events[-1]["agents_used"], list)

    def test_stream_handles_errors(self, mock_orchestrator):
        """Emite evento error em caso de falha."""
        service = TrafficAgentService(multi_agent_mode=True)
        service._orchestrator = mock_orchestrator

        # Configurar mock para lancar excecao
        async def error_astream(*args, **kwargs):
            raise Exception("Erro simulado no grafo")
            yield  # Para tornar funcao um generator

        mock_orchestrator.build_graph().astream = error_astream

        events = collect_events_sync(service.stream_chat_multi_agent(
            message="Teste",
            config_id=1,
            user_id=1,
            thread_id="test-error"
        ))

        # Deve ter evento error
        error_events = [e for e in events if e.get("type") == "error"]
        assert len(error_events) == 1
        assert "Erro simulado" in error_events[0]["error"]
        assert error_events[0]["thread_id"] == "test-error"
        assert "timestamp" in error_events[0]

    def test_stream_generates_thread_id(self, mock_orchestrator):
        """Gera thread_id se nao fornecido."""
        service = TrafficAgentService(multi_agent_mode=True)
        service._orchestrator = mock_orchestrator

        events = collect_events_sync(service.stream_chat_multi_agent(
            message="Teste",
            config_id=1,
            user_id=1,
            thread_id=None  # Nao fornecer thread_id
        ))

        # Verificar que thread_id foi gerado
        assert len(events) > 0
        thread_id = events[0].get("thread_id")
        assert thread_id is not None
        assert len(thread_id) > 0

        # Todos eventos devem ter o mesmo thread_id
        for event in events:
            assert event.get("thread_id") == thread_id


# =============================================================================
# TestStreamChatMultiAgentNotEnabled
# =============================================================================
class TestStreamChatMultiAgentNotEnabled:
    """Testes para quando modo multi-agente nao esta habilitado."""

    def test_stream_returns_error_when_not_enabled(self):
        """Retorna erro se multi-agent mode nao esta habilitado."""
        service = TrafficAgentService(multi_agent_mode=False)

        events = collect_events_sync(service.stream_chat_multi_agent(
            message="Teste",
            config_id=1,
            user_id=1,
            thread_id="test-thread"
        ))

        # Deve retornar apenas evento de erro
        assert len(events) == 1
        assert events[0]["type"] == "error"
        assert "multi-agente" in events[0]["error"].lower() or "não está habilitado" in events[0]["error"]


# =============================================================================
# TestChunkText
# =============================================================================
class TestChunkText:
    """Testes para o metodo _chunk_text."""

    def test_chunk_text_empty(self):
        """Retorna lista vazia para texto vazio."""
        service = TrafficAgentService()

        result = service._chunk_text("")
        assert result == []

        result = service._chunk_text(None)
        assert result == []

    def test_chunk_text_short(self):
        """Texto curto retorna em um chunk."""
        service = TrafficAgentService()

        result = service._chunk_text("Hello world")
        assert len(result) == 1
        assert "Hello" in result[0]

    def test_chunk_text_long(self):
        """Texto longo e dividido em multiplos chunks."""
        service = TrafficAgentService()

        long_text = "Esta e uma frase muito longa que deve ser dividida em multiplos chunks para streaming suave ao usuario."
        result = service._chunk_text(long_text, chunk_size=30)

        assert len(result) > 1
        # Juntar e verificar texto completo
        full = "".join(result)
        assert "frase" in full
        assert "chunks" in full

    def test_chunk_text_preserves_words(self):
        """Chunks nao cortam palavras no meio."""
        service = TrafficAgentService()

        text = "word1 word2 word3 word4 word5"
        result = service._chunk_text(text, chunk_size=10)

        # Nenhum chunk deve ter palavra cortada
        for chunk in result:
            words = chunk.strip().split()
            for word in words:
                assert word in ["word1", "word2", "word3", "word4", "word5"]


# =============================================================================
# TestGetAgentDescription
# =============================================================================
class TestGetAgentDescription:
    """Testes para o metodo _get_agent_description."""

    def test_get_known_agent_description(self):
        """Retorna descricao para agentes conhecidos."""
        service = TrafficAgentService()

        assert "performance" in service._get_agent_description("classification").lower()
        assert "anomal" in service._get_agent_description("anomaly").lower()
        assert "previs" in service._get_agent_description("forecast").lower()
        assert "recomenda" in service._get_agent_description("recommendation").lower()
        assert "campanha" in service._get_agent_description("campaign").lower()
        # "análises" contains the accented character, check for "avan" which is unaccented
        assert "avan" in service._get_agent_description("analysis").lower()

    def test_get_unknown_agent_description(self):
        """Retorna descricao generica para agentes desconhecidos."""
        service = TrafficAgentService()

        result = service._get_agent_description("unknown_agent")
        assert "unknown_agent" in result.lower() or "executando" in result.lower()


# =============================================================================
# TestStreamEventsOrder
# =============================================================================
class TestStreamEventsOrder:
    """Testes para verificar ordem correta dos eventos."""

    def test_events_in_correct_order(self, mock_orchestrator):
        """Eventos sao emitidos na ordem esperada."""
        service = TrafficAgentService(multi_agent_mode=True)
        service._orchestrator = mock_orchestrator

        events = collect_events_sync(service.stream_chat_multi_agent(
            message="Analise minhas campanhas",
            config_id=1,
            user_id=1,
            thread_id="test-thread"
        ))

        # Extrair tipos de eventos
        event_types = [e["type"] for e in events]

        # Verificar ordem geral
        # orchestrator_start deve ser primeiro
        assert event_types[0] == "orchestrator_start"

        # done deve ser ultimo
        assert event_types[-1] == "done"

        # intent_detected deve vir antes de plan_created
        if "intent_detected" in event_types and "plan_created" in event_types:
            assert event_types.index("intent_detected") < event_types.index("plan_created")

        # plan_created deve vir antes de agent_start
        if "plan_created" in event_types and "agent_start" in event_types:
            assert event_types.index("plan_created") < event_types.index("agent_start")

        # synthesis_start deve vir antes de text
        if "synthesis_start" in event_types and "text" in event_types:
            assert event_types.index("synthesis_start") < event_types.index("text")

        # text deve vir antes de done
        if "text" in event_types:
            text_indices = [i for i, t in enumerate(event_types) if t == "text"]
            done_index = event_types.index("done")
            assert all(i < done_index for i in text_indices)


# =============================================================================
# TestStreamTimestamps
# =============================================================================
class TestStreamTimestamps:
    """Testes para verificar timestamps dos eventos."""

    def test_all_events_have_timestamps(self, mock_orchestrator):
        """Todos os eventos possuem timestamp."""
        service = TrafficAgentService(multi_agent_mode=True)
        service._orchestrator = mock_orchestrator

        events = collect_events_sync(service.stream_chat_multi_agent(
            message="Teste",
            config_id=1,
            user_id=1,
            thread_id="test-thread"
        ))

        for event in events:
            assert "timestamp" in event
            assert isinstance(event["timestamp"], (int, float))
            assert event["timestamp"] > 0

    def test_timestamps_are_increasing(self, mock_orchestrator):
        """Timestamps sao monotonicamente crescentes."""
        service = TrafficAgentService(multi_agent_mode=True)
        service._orchestrator = mock_orchestrator

        events = collect_events_sync(service.stream_chat_multi_agent(
            message="Teste",
            config_id=1,
            user_id=1,
            thread_id="test-thread"
        ))

        timestamps = [e["timestamp"] for e in events]

        # Timestamps devem ser crescentes ou iguais
        for i in range(1, len(timestamps)):
            assert timestamps[i] >= timestamps[i-1]
