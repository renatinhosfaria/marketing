"""Testes unitarios para o TrafficAgentService com suporte multi-agente.

Testa:
- Inicializacao em modo single-agent (padrao) e multi-agent
- Roteamento de chat() para modo correto
- Metodo chat_multi_agent
- Factory functions (get_agent_service, get_multi_agent_service)
- Compatibilidade com codigo existente
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


# Configurar mocks antes de importar
_setup_mock_modules()

# Agora importar o modulo de servico
import importlib.util
service_path = os.path.join(root_path, 'projects', 'agent', 'service.py')
spec = importlib.util.spec_from_file_location("agent_service_test", service_path)
service_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(service_module)
_restore_mocked_modules()

TrafficAgentService = service_module.TrafficAgentService
get_agent_service = service_module.get_agent_service
get_multi_agent_service = service_module.get_multi_agent_service
reset_services = service_module.reset_services


def run_async(coro):
    """Helper para executar coroutines em testes sincronos."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


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
def mock_orchestrator():
    """Cria mock do OrchestratorAgent."""
    mock = MagicMock()
    mock.build_graph = MagicMock()
    mock.run = AsyncMock(return_value={
        "synthesized_response": "Resposta sintetizada do multi-agente",
        "confidence_score": 0.85,
        "user_intent": "analyze_performance",
        "required_agents": ["classification", "anomaly"],
        "agent_results": {
            "classification": {"success": True, "data": {}},
            "anomaly": {"success": True, "data": {}},
        },
        "error": None,
    })
    return mock


# =============================================================================
# TestTrafficAgentServiceMultiAgent
# =============================================================================
class TestTrafficAgentServiceMultiAgent:
    """Testes para TrafficAgentService com suporte multi-agente."""

    def test_service_init_single_agent_default(self):
        """Default mode e single-agent."""
        service = TrafficAgentService()

        assert service.multi_agent_mode is False
        assert service._multi_agent_mode is False
        assert service._orchestrator is None
        assert service._agent is None

    def test_service_init_multi_agent_mode(self):
        """Pode inicializar em modo multi-agent."""
        service = TrafficAgentService(multi_agent_mode=True)

        assert service.multi_agent_mode is True
        assert service._multi_agent_mode is True
        assert service._orchestrator is None  # Ainda nao inicializado

    def test_service_init_explicit_single_agent(self):
        """Pode inicializar explicitamente em modo single-agent."""
        service = TrafficAgentService(multi_agent_mode=False)

        assert service.multi_agent_mode is False
        assert service._multi_agent_mode is False

    def test_service_initialize_creates_orchestrator(self):
        """initialize() cria orchestrator em modo multi-agent."""
        service = TrafficAgentService(multi_agent_mode=True)

        # Mock do OrchestratorAgent
        mock_orchestrator_class = MagicMock()
        mock_orchestrator_instance = MagicMock()
        mock_orchestrator_class.return_value = mock_orchestrator_instance

        orchestrator_mod = types.ModuleType('app.agent.orchestrator')
        orchestrator_mod.OrchestratorAgent = mock_orchestrator_class
        sys.modules['app.agent.orchestrator'] = orchestrator_mod

        try:
            run_async(service.initialize())

            # Verificar que orchestrator foi criado
            assert service._orchestrator is not None
            mock_orchestrator_instance.build_graph.assert_called_once()
        finally:
            # Limpar mock
            if 'app.agent.orchestrator' in sys.modules:
                del sys.modules['app.agent.orchestrator']

    def test_service_chat_routes_to_multi_agent(self, mock_orchestrator):
        """chat() roteia para chat_multi_agent quando em modo multi-agent."""
        service = TrafficAgentService(multi_agent_mode=True)
        service._orchestrator = mock_orchestrator

        result = run_async(service.chat(
            message="Analise minhas campanhas",
            config_id=1,
            user_id=1,
            thread_id="test-thread"
        ))

        # Verifica que orchestrator.run foi chamado
        mock_orchestrator.run.assert_called_once_with(
            message="Analise minhas campanhas",
            config_id=1,
            user_id=1,
            thread_id="test-thread"
        )

        # Verifica estrutura da resposta
        assert result["success"] is True
        assert "response" in result
        assert "confidence_score" in result

    def test_chat_multi_agent_returns_correct_structure(self, mock_orchestrator):
        """chat_multi_agent retorna estrutura esperada."""
        service = TrafficAgentService(multi_agent_mode=True)
        service._orchestrator = mock_orchestrator

        result = run_async(service.chat_multi_agent(
            message="Como estao minhas campanhas?",
            config_id=1,
            user_id=1,
            thread_id="test-thread"
        ))

        # Verificar campos obrigatorios
        assert "success" in result
        assert "thread_id" in result
        assert "response" in result
        assert "confidence_score" in result
        assert "intent" in result
        assert "agents_used" in result
        assert "agent_results" in result

        # Verificar valores
        assert result["success"] is True
        assert result["thread_id"] == "test-thread"
        assert result["response"] == "Resposta sintetizada do multi-agente"
        assert result["confidence_score"] == 0.85
        assert result["intent"] == "analyze_performance"

    def test_chat_multi_agent_includes_agents_used(self, mock_orchestrator):
        """Resposta inclui lista de agentes usados."""
        service = TrafficAgentService(multi_agent_mode=True)
        service._orchestrator = mock_orchestrator

        result = run_async(service.chat_multi_agent(
            message="Teste",
            config_id=1,
            user_id=1
        ))

        # Verifica que agents_used esta presente e correto
        assert "agents_used" in result
        assert isinstance(result["agents_used"], list)
        assert "classification" in result["agents_used"]
        assert "anomaly" in result["agents_used"]

    def test_chat_multi_agent_includes_confidence(self, mock_orchestrator):
        """Resposta inclui confidence_score."""
        service = TrafficAgentService(multi_agent_mode=True)
        service._orchestrator = mock_orchestrator

        result = run_async(service.chat_multi_agent(
            message="Teste",
            config_id=1,
            user_id=1
        ))

        assert "confidence_score" in result
        assert isinstance(result["confidence_score"], float)
        assert 0 <= result["confidence_score"] <= 1

    def test_chat_multi_agent_handles_errors(self, mock_orchestrator):
        """Trata erros graciosamente."""
        service = TrafficAgentService(multi_agent_mode=True)
        service._orchestrator = mock_orchestrator

        # Configurar mock para lancar excecao
        mock_orchestrator.run.side_effect = Exception("Erro de teste")

        result = run_async(service.chat_multi_agent(
            message="Teste",
            config_id=1,
            user_id=1,
            thread_id="test-error"
        ))

        # Deve retornar erro, nao lancar excecao
        assert result["success"] is False
        assert "error" in result
        assert "Erro de teste" in result["error"]
        assert result["thread_id"] == "test-error"
        assert result["confidence_score"] == 0.0
        assert result["agents_used"] == []


# =============================================================================
# TestGetMultiAgentService
# =============================================================================
class TestGetMultiAgentService:
    """Testes para a factory function get_multi_agent_service."""

    def test_get_multi_agent_service_returns_instance(self):
        """Retorna instancia inicializada."""
        # Mock do OrchestratorAgent
        mock_orchestrator_class = MagicMock()
        mock_orchestrator_instance = MagicMock()
        mock_orchestrator_class.return_value = mock_orchestrator_instance

        orchestrator_mod = types.ModuleType('app.agent.orchestrator')
        orchestrator_mod.OrchestratorAgent = mock_orchestrator_class
        sys.modules['app.agent.orchestrator'] = orchestrator_mod

        try:
            service = run_async(get_multi_agent_service())

            assert service is not None
            assert isinstance(service, TrafficAgentService)
        finally:
            if 'app.agent.orchestrator' in sys.modules:
                del sys.modules['app.agent.orchestrator']

    def test_get_multi_agent_service_singleton(self):
        """Retorna mesma instancia em chamadas multiplas."""
        mock_orchestrator_class = MagicMock()
        mock_orchestrator_instance = MagicMock()
        mock_orchestrator_class.return_value = mock_orchestrator_instance

        orchestrator_mod = types.ModuleType('app.agent.orchestrator')
        orchestrator_mod.OrchestratorAgent = mock_orchestrator_class
        sys.modules['app.agent.orchestrator'] = orchestrator_mod

        try:
            # Usar mesmo event loop para simular comportamento de singleton
            async def get_both():
                s1 = await get_multi_agent_service()
                s2 = await get_multi_agent_service()
                return s1, s2

            service1, service2 = run_async(get_both())
            assert service1 is service2
        finally:
            if 'app.agent.orchestrator' in sys.modules:
                del sys.modules['app.agent.orchestrator']

    def test_get_multi_agent_service_is_multi_agent_mode(self):
        """Instancia retornada esta em modo multi-agent."""
        mock_orchestrator_class = MagicMock()
        mock_orchestrator_instance = MagicMock()
        mock_orchestrator_class.return_value = mock_orchestrator_instance

        orchestrator_mod = types.ModuleType('app.agent.orchestrator')
        orchestrator_mod.OrchestratorAgent = mock_orchestrator_class
        sys.modules['app.agent.orchestrator'] = orchestrator_mod

        try:
            service = run_async(get_multi_agent_service())

            assert service.multi_agent_mode is True
        finally:
            if 'app.agent.orchestrator' in sys.modules:
                del sys.modules['app.agent.orchestrator']


# =============================================================================
# TestBackwardsCompatibility
# =============================================================================
class TestBackwardsCompatibility:
    """Testes de compatibilidade com codigo existente."""

    def test_existing_chat_signature_unchanged(self):
        """Assinatura do metodo chat nao mudou."""
        import inspect

        service = TrafficAgentService()
        sig = inspect.signature(service.chat)

        params = list(sig.parameters.keys())
        assert "message" in params
        assert "config_id" in params
        assert "user_id" in params
        assert "thread_id" in params

    def test_existing_get_agent_service_works(self):
        """get_agent_service() ainda funciona."""
        service = run_async(get_agent_service())

        assert service is not None
        assert isinstance(service, TrafficAgentService)
        assert service.multi_agent_mode is False

    def test_existing_get_agent_service_singleton(self):
        """get_agent_service() retorna singleton."""
        async def get_both():
            s1 = await get_agent_service()
            s2 = await get_agent_service()
            return s1, s2

        service1, service2 = run_async(get_both())
        assert service1 is service2

    def test_stream_chat_method_exists(self):
        """Metodo stream_chat ainda existe."""
        service = TrafficAgentService()

        assert hasattr(service, 'stream_chat')

    def test_get_conversation_history_method_exists(self):
        """Metodo get_conversation_history ainda existe."""
        service = TrafficAgentService()

        assert hasattr(service, 'get_conversation_history')

    def test_clear_conversation_method_exists(self):
        """Metodo clear_conversation ainda existe."""
        service = TrafficAgentService()

        assert hasattr(service, 'clear_conversation')

    def test_service_attributes_unchanged(self):
        """Atributos do servico nao mudaram."""
        service = TrafficAgentService()

        # Atributos existentes
        assert hasattr(service, '_agent')
        assert hasattr(service, '_checkpointer')

        # Novos atributos nao quebram API
        assert hasattr(service, '_multi_agent_mode')
        assert hasattr(service, '_orchestrator')


# =============================================================================
# TestResetServices
# =============================================================================
class TestResetServices:
    """Testes para a funcao reset_services."""

    def test_reset_services_clears_singletons(self):
        """reset_services limpa singletons."""
        # Criar singleton
        service1 = run_async(get_agent_service())
        assert service1 is not None

        # Reset
        reset_services()

        # Criar novamente - deve ser nova instancia
        service2 = run_async(get_agent_service())
        assert service2 is not None

        # Devem ser instancias diferentes
        assert service1 is not service2


# =============================================================================
# TestMultiAgentPropertyAccess
# =============================================================================
class TestMultiAgentPropertyAccess:
    """Testes para acesso a propriedades do servico."""

    def test_multi_agent_mode_property(self):
        """Property multi_agent_mode funciona."""
        service_single = TrafficAgentService(multi_agent_mode=False)
        service_multi = TrafficAgentService(multi_agent_mode=True)

        assert service_single.multi_agent_mode is False
        assert service_multi.multi_agent_mode is True

    def test_orchestrator_property_none_initially(self):
        """Property orchestrator e None inicialmente."""
        service = TrafficAgentService(multi_agent_mode=True)

        assert service.orchestrator is None

    def test_orchestrator_property_after_init(self):
        """Property orchestrator nao e None apos inicializacao."""
        service = TrafficAgentService(multi_agent_mode=True)

        mock_orchestrator = MagicMock()
        mock_orchestrator_class = MagicMock(return_value=mock_orchestrator)

        orchestrator_mod = types.ModuleType('app.agent.orchestrator')
        orchestrator_mod.OrchestratorAgent = mock_orchestrator_class
        sys.modules['app.agent.orchestrator'] = orchestrator_mod

        try:
            run_async(service.initialize())

            assert service.orchestrator is not None
            assert service.orchestrator is mock_orchestrator
        finally:
            if 'app.agent.orchestrator' in sys.modules:
                del sys.modules['app.agent.orchestrator']


# =============================================================================
# TestChatMultiAgentEdgeCases
# =============================================================================
class TestChatMultiAgentEdgeCases:
    """Testes para casos de borda do chat_multi_agent."""

    def test_empty_message(self, mock_orchestrator):
        """Trata mensagem vazia."""
        service = TrafficAgentService(multi_agent_mode=True)
        service._orchestrator = mock_orchestrator

        result = run_async(service.chat_multi_agent(
            message="",
            config_id=1,
            user_id=1
        ))

        # Deve processar mesmo com mensagem vazia
        assert "success" in result

    def test_empty_synthesized_response(self, mock_orchestrator):
        """Trata resposta sintetizada vazia."""
        service = TrafficAgentService(multi_agent_mode=True)
        service._orchestrator = mock_orchestrator

        # Configurar mock para retornar resposta vazia
        mock_orchestrator.run.return_value = {
            "synthesized_response": "",
            "confidence_score": 0.5,
            "user_intent": "general",
            "required_agents": [],
            "agent_results": {},
            "error": None,
        }

        result = run_async(service.chat_multi_agent(
            message="Teste",
            config_id=1,
            user_id=1
        ))

        # Deve fornecer mensagem padrao
        assert result["success"] is True
        assert result["response"] != ""
        assert "tente novamente" in result["response"].lower()

    def test_partial_agent_results(self, mock_orchestrator):
        """Trata resultados parciais de agentes."""
        service = TrafficAgentService(multi_agent_mode=True)
        service._orchestrator = mock_orchestrator

        # Configurar mock com um agente falhando
        mock_orchestrator.run.return_value = {
            "synthesized_response": "Resposta parcial",
            "confidence_score": 0.6,
            "user_intent": "analyze_performance",
            "required_agents": ["classification", "anomaly"],
            "agent_results": {
                "classification": {"success": True, "data": {}},
                "anomaly": {"success": False, "error": "Timeout"},
            },
            "error": None,
        }

        result = run_async(service.chat_multi_agent(
            message="Teste",
            config_id=1,
            user_id=1
        ))

        # Deve retornar sucesso com agentes parciais
        assert result["success"] is True
        # Apenas classification teve sucesso
        assert "classification" in result["agents_used"]
        assert "anomaly" not in result["agents_used"]

    def test_chat_multi_agent_not_enabled_error(self):
        """Retorna erro se chamado sem modo multi-agent habilitado."""
        service = TrafficAgentService(multi_agent_mode=False)

        result = run_async(service.chat_multi_agent(
            message="Teste",
            config_id=1,
            user_id=1
        ))

        assert result["success"] is False
        assert "error" in result

    def test_chat_multi_agent_generates_thread_id(self, mock_orchestrator):
        """Gera thread_id se nao fornecido."""
        service = TrafficAgentService(multi_agent_mode=True)
        service._orchestrator = mock_orchestrator

        result = run_async(service.chat_multi_agent(
            message="Teste",
            config_id=1,
            user_id=1,
            thread_id=None
        ))

        assert result["thread_id"] is not None
        assert len(result["thread_id"]) > 0

    def test_chat_multi_agent_handles_orchestrator_error(self, mock_orchestrator):
        """Trata erro retornado pelo orchestrator."""
        service = TrafficAgentService(multi_agent_mode=True)
        service._orchestrator = mock_orchestrator

        # Configurar mock para retornar erro
        mock_orchestrator.run.return_value = {
            "error": "Erro no orchestrator",
            "user_intent": "general",
            "required_agents": [],
            "agent_results": {},
        }

        result = run_async(service.chat_multi_agent(
            message="Teste",
            config_id=1,
            user_id=1
        ))

        assert result["success"] is False
        assert "error" in result
        assert result["error"] == "Erro no orchestrator"
