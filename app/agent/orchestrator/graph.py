"""Orchestrator Graph - Coordena o fluxo multi-agente.

Este modulo define o grafo principal do sistema multi-agente,
conectando todos os nos do orchestrator e subagentes.

O fluxo principal e:
START -> parse_request -> plan_execution -> [conditional]
                                             |
              (has agents) -> dispatch_agents -> collect_results -> synthesize -> END
              (no agents)  -> synthesize -> END
"""
import os
import importlib.util
from typing import Any, Literal

from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph


# Carregar state.py diretamente para evitar problemas de import circular
_state_path = os.path.join(
    os.path.dirname(__file__),
    'state.py'
)
_state_path = os.path.abspath(_state_path)

_spec = importlib.util.spec_from_file_location("orchestrator_state", _state_path)
_state_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_state_module)

OrchestratorState = _state_module.OrchestratorState
create_initial_orchestrator_state = _state_module.create_initial_orchestrator_state

# Carregar nodes diretamente
_nodes_init_path = os.path.join(
    os.path.dirname(__file__),
    'nodes', '__init__.py'
)
_nodes_init_path = os.path.abspath(_nodes_init_path)

# Carregar cada no individualmente para evitar problemas de import
_parse_request_path = os.path.join(
    os.path.dirname(__file__),
    'nodes', 'parse_request.py'
)
_parse_request_path = os.path.abspath(_parse_request_path)

_parse_spec = importlib.util.spec_from_file_location("parse_request_module", _parse_request_path)
_parse_module = importlib.util.module_from_spec(_parse_spec)
_parse_spec.loader.exec_module(_parse_module)
parse_request = _parse_module.parse_request

_plan_execution_path = os.path.join(
    os.path.dirname(__file__),
    'nodes', 'plan_execution.py'
)
_plan_execution_path = os.path.abspath(_plan_execution_path)

_plan_spec = importlib.util.spec_from_file_location("plan_execution_module", _plan_execution_path)
_plan_module = importlib.util.module_from_spec(_plan_spec)
_plan_spec.loader.exec_module(_plan_module)
plan_execution = _plan_module.plan_execution

_dispatch_path = os.path.join(
    os.path.dirname(__file__),
    'nodes', 'dispatch.py'
)
_dispatch_path = os.path.abspath(_dispatch_path)

_dispatch_spec = importlib.util.spec_from_file_location("dispatch_module", _dispatch_path)
_dispatch_module = importlib.util.module_from_spec(_dispatch_spec)
_dispatch_spec.loader.exec_module(_dispatch_module)
dispatch_agents = _dispatch_module.dispatch_agents

_collect_results_path = os.path.join(
    os.path.dirname(__file__),
    'nodes', 'collect_results.py'
)
_collect_results_path = os.path.abspath(_collect_results_path)

_collect_spec = importlib.util.spec_from_file_location("collect_results_module", _collect_results_path)
_collect_module = importlib.util.module_from_spec(_collect_spec)
_collect_spec.loader.exec_module(_collect_module)
collect_results = _collect_module.collect_results

_synthesize_path = os.path.join(
    os.path.dirname(__file__),
    'nodes', 'synthesize.py'
)
_synthesize_path = os.path.abspath(_synthesize_path)

_synthesize_spec = importlib.util.spec_from_file_location("synthesize_module", _synthesize_path)
_synthesize_module = importlib.util.module_from_spec(_synthesize_spec)
_synthesize_spec.loader.exec_module(_synthesize_module)
synthesize = _synthesize_module.synthesize


def should_dispatch(state: OrchestratorState) -> Literal["dispatch", "synthesize"]:
    """Decide se deve disparar subagentes ou ir direto para sintese.

    Esta funcao e usada como roteador condicional no grafo.
    Se houver agentes requeridos, vai para dispatch_agents.
    Se nao houver, vai direto para synthesize.

    Args:
        state: Estado atual do orchestrator

    Returns:
        "dispatch" se houver agentes requeridos
        "synthesize" se nao houver agentes
    """
    required_agents = state.get("required_agents")

    # Se required_agents for None ou lista vazia, vai para synthesize
    if required_agents is None or len(required_agents) == 0:
        return "synthesize"

    return "dispatch"


def build_orchestrator_graph() -> CompiledStateGraph:
    """Constroi e compila o grafo do Orchestrator.

    O grafo implementa o fluxo:
    1. parse_request: Analisa a mensagem e detecta intencao
    2. plan_execution: Cria plano de execucao com agentes necessarios
    3. [condicional]:
       - Se tem agentes: dispatch_agents -> collect_results -> synthesize
       - Se nao tem: vai direto para synthesize
    4. synthesize: Gera resposta final sintetizada

    Returns:
        Grafo compilado pronto para execucao
    """
    # Criar grafo de estado
    graph = StateGraph(OrchestratorState)

    # Adicionar nos principais
    graph.add_node("parse_request", parse_request)
    graph.add_node("plan_execution", plan_execution)
    graph.add_node("dispatch_agents", _dispatch_wrapper)
    graph.add_node("collect_results", _collect_wrapper)
    graph.add_node("synthesize", synthesize)

    # Definir fluxo: START -> parse_request -> plan_execution
    graph.add_edge(START, "parse_request")
    graph.add_edge("parse_request", "plan_execution")

    # Roteamento condicional apos plan_execution
    graph.add_conditional_edges(
        "plan_execution",
        should_dispatch,
        {
            "dispatch": "dispatch_agents",
            "synthesize": "synthesize"
        }
    )

    # Fluxo apos dispatch
    graph.add_edge("dispatch_agents", "collect_results")
    graph.add_edge("collect_results", "synthesize")

    # Fim do grafo
    graph.add_edge("synthesize", END)

    # Compilar e retornar
    return graph.compile()


def _dispatch_wrapper(state: OrchestratorState) -> dict[str, Any]:
    """Wrapper para dispatch_agents que lida com Send() de forma simplificada.

    O dispatch_agents original retorna uma lista de Send() para execucao paralela.
    Para o grafo simplificado, simulamos a execucao dos subagentes.

    Args:
        state: Estado atual do orchestrator

    Returns:
        Estado atualizado com marcacao de dispatch executado
    """
    # Chamar dispatch_agents original
    sends = dispatch_agents(state)

    # Para implementacao simplificada, apenas marca que dispatch foi chamado
    # Os subagentes reais seriam executados pelo LangGraph com Send()
    # Por enquanto, retornamos estado indicando que dispatch foi executado
    return {
        "agent_results": _simulate_subagent_execution(state)
    }


def _simulate_subagent_execution(state: OrchestratorState) -> dict[str, Any]:
    """Simula execucao dos subagentes para testes.

    Em producao, isso seria substituido por execucao real via Send().

    Args:
        state: Estado do orchestrator

    Returns:
        Dicionario com resultados simulados dos subagentes
    """
    from app.agent.orchestrator.state import AgentResult

    required_agents = state.get("required_agents", [])
    results = {}

    for agent_name in required_agents:
        results[agent_name] = AgentResult(
            agent_name=agent_name,
            success=True,
            data={"message": f"Analise do {agent_name} completada."},
            error=None,
            duration_ms=100,
            tool_calls=[]
        )

    return results


def _collect_wrapper(state: OrchestratorState) -> dict[str, Any]:
    """Wrapper para collect_results.

    Args:
        state: Estado atual do orchestrator

    Returns:
        Estado atualizado com resultados coletados
    """
    # Para implementacao simplificada, os resultados ja estao em agent_results
    # vindos do _dispatch_wrapper, entao apenas passamos adiante
    return collect_results(state, None)


class OrchestratorAgent:
    """Wrapper de alto nivel para o grafo do Orchestrator.

    Esta classe fornece uma interface simplificada para executar
    o sistema multi-agente. Ela cuida da criacao do grafo e do
    estado inicial.

    Attributes:
        _graph: Grafo compilado (cacheado)

    Example:
        agent = OrchestratorAgent()
        result = await agent.run(
            message="Analise minhas campanhas",
            config_id=1,
            user_id=1,
            thread_id="thread-123"
        )
    """

    def __init__(self):
        """Inicializa o OrchestratorAgent."""
        self._graph: CompiledStateGraph | None = None

    def build_graph(self) -> CompiledStateGraph:
        """Constroi e retorna o grafo compilado.

        O grafo e cacheado apos a primeira construcao.

        Returns:
            Grafo compilado do orchestrator
        """
        if self._graph is None:
            self._graph = build_orchestrator_graph()
        return self._graph

    async def run(
        self,
        message: str,
        config_id: int,
        user_id: int,
        thread_id: str
    ) -> dict[str, Any]:
        """Executa o orchestrator com a mensagem fornecida.

        Args:
            message: Mensagem do usuario
            config_id: ID da configuracao Facebook Ads
            user_id: ID do usuario autenticado
            thread_id: ID da thread para persistencia

        Returns:
            Estado final do orchestrator apos execucao completa
        """
        # Criar estado inicial
        initial_state = create_initial_orchestrator_state(
            config_id=config_id,
            user_id=user_id,
            thread_id=thread_id,
            messages=[HumanMessage(content=message)] if message else []
        )

        # Obter grafo
        graph = self.build_graph()

        # Executar grafo de forma assincrona
        try:
            final_state = await graph.ainvoke(initial_state)
            return final_state
        except Exception as e:
            # Em caso de erro, retornar estado com erro
            return {
                **initial_state,
                "error": f"Erro na execucao do orchestrator: {str(e)}"
            }


_orchestrator_graph: CompiledStateGraph | None = None


def get_orchestrator() -> CompiledStateGraph:
    """Retorna instância singleton do grafo do Orchestrator."""
    global _orchestrator_graph

    if _orchestrator_graph is None:
        _orchestrator_graph = build_orchestrator_graph()

    return _orchestrator_graph


def reset_orchestrator() -> None:
    """Reseta o singleton do orchestrator (para testes)."""
    global _orchestrator_graph
    _orchestrator_graph = None
